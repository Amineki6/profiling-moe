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
// Production Fused Sandwich Norm Kernel (Scalar In-Register Variant)
//
// 1. Hoists all 4 global memory inputs up front into scalar thread registers (attn_out, residual_in, gamma1, gamma2).
// 2. 100% In-Register activation preservation (everything stays in thread registers, 0 B shmem data caching).
// 3. Lightweight single-barrier reduction with warp-shuffle + broadcast (red1, red2 arrays).
// =================================================================================================
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedSandwichNormScalarRegKernel(
    T const* __restrict__ attn_out,
    T const* __restrict__ residual_in,
    T const* __restrict__ gamma1,
    T const* __restrict__ gamma2,
    T* __restrict__ updated_residual_out,
    T* __restrict__ moe_in_out,
    float const eps)
{
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_ELEM  = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    // Two separate shared buffers eliminate write-after-read hazards
    __shared__ float red1[NUM_WARPS];
    __shared__ float red2[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_ELEM;
    int const tid = threadIdx.x;

    T a[ITERS];
    T r[ITERS];
    T g1[ITERS];
    T g2[ITERS];

    // Issue all scalar global loads up front into registers
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
        float const f = to_float(a[k]);
        ss += f * f;
    }
    float const inv1 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red1) * INV_HIDDEN_DIM + eps);

    // 2. Post-attention RMSNorm + Residual addition in-registers, store updated residual, accumulate pre-MoE variance
    ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        float const a_f = to_float(a[k]);
        float const g1_f = to_float(g1[k]);
        float const r_f = to_float(r[k]);

        float const norm1 = a_f * inv1 * g1_f;
        float const res_val = r_f + norm1;

        a[k] = from_float<T>(res_val);
        updated_residual_out[row + tid + k * THREADS] = a[k];

        ss += res_val * res_val;
    }
    float const inv2 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red2) * INV_HIDDEN_DIM + eps);

    // 3. Pre-MoE RMSNorm in-registers, store moe_in
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        float const a_f = to_float(a[k]);
        float const g2_f = to_float(g2[k]);
        moe_in_out[row + tid + k * THREADS] = from_float<T>(a_f * inv2 * g2_f);
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
    constexpr int NUM_WARPS = 5;
    constexpr int ITERS = 16; // 5 * 32 * 16 = 2560 scalar elements
    constexpr int HIDDEN_DIM = 2560;

    if (hidden_dim != HIDDEN_DIM)
    {
        throw std::runtime_error("Fused sandwich norm scalar reg requires hidden_dim=2560");
    }

    dim3 const grid(tokens);
    dim3 const block(NUM_WARPS * 32);

    fusedSandwichNormScalarRegKernel<T, NUM_WARPS, ITERS, HIDDEN_DIM><<<grid, block, 0, stream>>>(
        attn_out,
        residual_in,
        gamma1,
        gamma2,
        updated_residual_out,
        moe_in_out,
        eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("FusedSandwichNormScalarReg launch failed: ") + cudaGetErrorString(err));
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
