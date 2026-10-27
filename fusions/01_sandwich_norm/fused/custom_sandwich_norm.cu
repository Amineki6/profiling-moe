#include "custom_sandwich_norm.cuh"

#include <stdexcept>
#include <string>

namespace kolibri
{
namespace fusions
{

// Single-barrier block reduction: warp shuffle followed by a 5-warp broadcast read
template <int NUM_WARPS>
__device__ __forceinline__ float blockSumAll(float v, float* red)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float t = 0.f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        t += red[w]; // broadcast read
    }
    return t; // valid across every thread in the CTA
}

// =================================================================================================
// Production Fused Sandwich Norm Kernel (SM 89 Ada Lovelace Optimized)
//
// 1. Hoists all 4 global memory inputs up front (attn_out, residual_in, gamma1, gamma2).
// 2. 100% In-Register activation preservation (everything stays in ~32 thread registers, 0 B shmem).
// 3. Lightweight single-barrier reduction with warp-shuffle + broadcast (red1, red2 arrays).
// =================================================================================================
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedSandwichNormRegKernel(
    Vec128_t<T> const* __restrict__ attn_out,
    Vec128_t<T> const* __restrict__ residual_in,
    Vec128_t<T> const* __restrict__ gamma1,
    Vec128_t<T> const* __restrict__ gamma2,
    Vec128_t<T>* __restrict__ updated_residual_out,
    Vec128_t<T>* __restrict__ moe_in_out,
    float const eps)
{
    using V = Vec128_t<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    // Two separate shared buffers eliminate write-after-read hazards
    __shared__ float red1[NUM_WARPS];
    __shared__ float red2[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V a[ITERS];
    V r[ITERS];
    V g1[ITERS];
    V g2[ITERS];

    // Issue all global loads up front into registers to maximize memory-level parallelism
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        a[k] = attn_out[row + i];
        r[k] = residual_in[row + i];
    }

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        g1[k] = gamma1[i];
        g2[k] = gamma2[i];
    }

    // 1. Compute first RMSNorm variance (Post-Attention Sandwich Norm)
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += a[k].sum_of_squares();
    }
    float const inv1 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red1) * INV_HIDDEN_DIM + eps);

    // 2. Post-attention RMSNorm + Residual addition in-registers, store updated residual, accumulate pre-MoE variance
    ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        a[k] = a[k].rmsnorm(inv1, g1[k]).add(r[k]);
        updated_residual_out[row + tid + k * THREADS] = a[k];
        ss += a[k].sum_of_squares();
    }
    float const inv2 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red2) * INV_HIDDEN_DIM + eps);

    // 3. Pre-MoE RMSNorm in-registers, store moe_in
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_in_out[row + tid + k * THREADS] = a[k].rmsnorm(inv2, g2[k]);
    }
}

template <typename T>
void invokeFusedSandwichNormReg(
    T const* attn_out,
    T const* residual_in,
    T const* gamma1,
    T const* gamma2,
    T* updated_residual_out,
    T* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream)
{
    using V = Vec128_t<T>;
    constexpr int NUM_WARPS = 5;
    constexpr int ITERS = 2; // 5 * 32 * 2 = 320 vectors = 2560 elements
    constexpr int HIDDEN_DIM = 2560;

    if (hidden_dim != HIDDEN_DIM)
    {
        throw std::runtime_error("Fused sandwich norm reg requires hidden_dim=2560");
    }

    dim3 const grid(tokens);
    dim3 const block(NUM_WARPS * 32);

    fusedSandwichNormRegKernel<T, NUM_WARPS, ITERS, HIDDEN_DIM><<<grid, block, 0, stream>>>(
        reinterpret_cast<V const*>(attn_out),
        reinterpret_cast<V const*>(residual_in),
        reinterpret_cast<V const*>(gamma1),
        reinterpret_cast<V const*>(gamma2),
        reinterpret_cast<V*>(updated_residual_out),
        reinterpret_cast<V*>(moe_in_out),
        eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("FusedSandwichNormReg launch failed: ") + cudaGetErrorString(err));
    }
}

// Template explicit instantiations
template void invokeFusedSandwichNormReg<half>(
    half const* attn_out,
    half const* residual_in,
    half const* gamma1,
    half const* gamma2,
    half* updated_residual_out,
    half* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedSandwichNormReg<__nv_bfloat16>(
    __nv_bfloat16 const* attn_out,
    __nv_bfloat16 const* residual_in,
    __nv_bfloat16 const* gamma1,
    __nv_bfloat16 const* gamma2,
    __nv_bfloat16* updated_residual_out,
    __nv_bfloat16* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedSandwichNormReg<float>(
    float const* attn_out,
    float const* residual_in,
    float const* gamma1,
    float const* gamma2,
    float* updated_residual_out,
    float* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
