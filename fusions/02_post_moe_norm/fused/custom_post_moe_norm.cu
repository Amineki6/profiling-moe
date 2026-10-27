#include "custom_post_moe_norm.cuh"

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
        t += red[w]; // broadcast read across CTA
    }
    return t; // valid across every thread in the CTA
}

// PTX Cache Modifiers: .cg (bypass L1 for streaming tokens) and .ca (cache in L1 for shared gamma)
template <typename T>
__device__ __forceinline__ Vec128_t<T> load_128_cg(Vec128_t<T> const* ptr)
{
    Vec128_t<T> ret;
    uint4* out = reinterpret_cast<uint4*>(&ret);
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out->x), "=r"(out->y), "=r"(out->z), "=r"(out->w)
                 : "l"(ptr));
    return ret;
}

template <typename T>
__device__ __forceinline__ Vec128_t<T> load_128_ca(Vec128_t<T> const* ptr)
{
    Vec128_t<T> ret;
    uint4* out = reinterpret_cast<uint4*>(&ret);
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out->x), "=r"(out->y), "=r"(out->z), "=r"(out->w)
                 : "l"(ptr));
    return ret;
}

// =================================================================================================
// Production Fused Add + Post-MoE Norm Kernel (SM 89 Ada Lovelace Optimized)
//
// 1. In-Register Vectorized Addition: Loads routed_out and shared_out via 128-bit vectors and adds them.
// 2. In-Register Variance Accumulation: Computes sum of squares of (routed + shared) in registers.
// 3. Single-Barrier CTA Reduction: Fast blockSumAll using warp shuffles and 32-byte static shared memory.
// 4. In-Register Normalization & 128-Bit Store: Multiplies by rsqrt(var + eps) and gamma, writing directly to DRAM.
//
// Completely eliminates:
//   - Intermediate DRAM write/read of (routed_out + shared_out).
//   - Separate elementwise add kernel launch.
//   - Dynamic shared memory allocations.
// =================================================================================================
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedAddPostMoeNormRegKernel(
    Vec128_t<T> const* __restrict__ routed_out,
    Vec128_t<T> const* __restrict__ shared_out,
    Vec128_t<T> const* __restrict__ gamma,
    Vec128_t<T>* __restrict__ moe_out,
    float const eps)
{
    using V = Vec128_t<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    __shared__ float red[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V c[ITERS];
    V g[ITERS];

    // Load routed and shared vectors via .cg, perform register addition, and load gamma via .ca
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        V routed = load_128_cg(&routed_out[row + i]);
        V shared = load_128_cg(&shared_out[row + i]);
        c[k] = routed.add(shared);
        g[k] = load_128_ca(&gamma[i]);
    }

    // Accumulate sum of squares in registers
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += c[k].sum_of_squares();
    }

    // Intra-block reduction
    float const sum = blockSumAll<NUM_WARPS>(ss, red);
    float const inv = rsqrtf(sum * INV_HIDDEN_DIM + eps);

    // Apply RMS scale and gamma weight, then store directly to global DRAM
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_out[row + tid + k * THREADS] = c[k].rmsnorm(inv, g[k]);
    }
}

// General fallback kernel for arbitrary hidden dimensions
template <typename T>
__global__ void fusedAddPostMoeNormGeneralKernel(
    T const* __restrict__ routed_out,
    T const* __restrict__ shared_out,
    T const* __restrict__ gamma,
    T* __restrict__ moe_out,
    int const hidden_dim,
    float const eps)
{
    extern __shared__ float s_red[];
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    size_t const row_offset = size_t(bid) * hidden_dim;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        float const r = float(routed_out[row_offset + i]);
        float const s = float(shared_out[row_offset + i]);
        float const sum = r + s;
        local_ss += sum * sum;
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_ss += __shfl_xor_sync(0xffffffffu, local_ss, offset);
    }

    int const lane = tid & 31;
    int const warp_id = tid >> 5;
    if (lane == 0)
    {
        s_red[warp_id] = local_ss;
    }
    __syncthreads();

    int const num_warps = blockDim.x >> 5;
    float block_sum = 0.0f;
    if (tid < num_warps)
    {
        block_sum = s_red[tid];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        block_sum += __shfl_xor_sync(0xffffffffu, block_sum, offset);
    }

    if (tid == 0)
    {
        s_red[0] = rsqrtf(block_sum / float(hidden_dim) + eps);
    }
    __syncthreads();

    float const inv = s_red[0];

    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        float const r = float(routed_out[row_offset + i]);
        float const s = float(shared_out[row_offset + i]);
        float const g = float(gamma[i]);
        float const norm_val = (r + s) * inv * g;
        moe_out[row_offset + i] = T(norm_val);
    }
}

// Host launcher function
template <typename T>
void invokeFusedAddPostMoeNorm(
    T const* routed_out,
    T const* shared_out,
    T const* gamma,
    T* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream)
{
    dim3 const grid(tokens);
    constexpr int vec128_elements = 16 / sizeof(T);

    // Fast-path for Kolibri hidden_dim=2560 (BF16 / FP16):
    // NUM_WARPS = 5 (160 threads), ITERS = 2, HIDDEN_DIM = 2560 (320 128-bit vectors)
    if (hidden_dim == 2560 && vec128_elements == 8)
    {
        using V128 = Vec128_t<T>;
        fusedAddPostMoeNormRegKernel<T, 5, 2, 2560><<<grid, 160, 0, stream>>>(
            reinterpret_cast<V128 const*>(routed_out),
            reinterpret_cast<V128 const*>(shared_out),
            reinterpret_cast<V128 const*>(gamma),
            reinterpret_cast<V128*>(moe_out),
            eps
        );
    }
    else
    {
        int const threads = 256;
        size_t const shmem_bytes = (threads / 32) * sizeof(float);
        fusedAddPostMoeNormGeneralKernel<T><<<grid, threads, shmem_bytes, stream>>>(
            routed_out, shared_out, gamma, moe_out, hidden_dim, eps
        );
    }
}

// Explicit template instantiations
template void invokeFusedAddPostMoeNorm<__nv_bfloat16>(
    __nv_bfloat16 const* routed_out,
    __nv_bfloat16 const* shared_out,
    __nv_bfloat16 const* gamma,
    __nv_bfloat16* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedAddPostMoeNorm<half>(
    half const* routed_out,
    half const* shared_out,
    half const* gamma,
    half* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedAddPostMoeNorm<float>(
    float const* routed_out,
    float const* shared_out,
    float const* gamma,
    float* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
