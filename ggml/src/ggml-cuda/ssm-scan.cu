#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB

#include "ssm-scan.cuh"


// Minimum number of tokens to use SSD (State Space Duality) matmul path instead of scan path.
// For n_tok <= this threshold, the scan kernel is used (lower overhead for short sequences).
#define SSM_SSD_MIN_TOKENS 128

// prepare_dt kernel dimensions: one block per (head, seq), each block handles DT_MAX_ITEMS items.
#define SSM_SSD_DT_BLOCK     256
#define SSM_SSD_DT_MAX_ITEMS  32

// Maximum tokens the SSD path supports, derived from the prepare_dt kernel block capacity.
#define SSM_SSD_MAX_TOKENS (SSM_SSD_DT_BLOCK * SSM_SSD_DT_MAX_ITEMS)

// Chunk size for chunked SSD. Caps matmul cost at O(chunk^2) per chunk.
#define SSM_SSD_CHUNK_SIZE 256

// We would like to keep pragma unroll for cases where L_template is not 0,
// so we suppress the clang transformation warning.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    ssm_scan_f32(const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
                 const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
                 const int32_t * src6_ptr, float * dst_ptr,
                 const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
                 const int64_t s_off, const int64_t d_inner, const int64_t L_param)
{
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
    const float   * GGML_CUDA_RESTRICT src4 = src4_ptr;
    const float   * GGML_CUDA_RESTRICT src5 = src5_ptr;
    const int32_t * GGML_CUDA_RESTRICT src6 = src6_ptr;
    float         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const size_t L = L_template == 0 ? L_param : L_template;
    ggml_cuda_pdl_sync();
    const float *s0_block = (const float *)((const char *)src0 + src6[blockIdx.x] * src0_nb3 + blockIdx.y * splitD * src0_nb2);
    const float *x_block = (const float *)((const char *)src1 + (blockIdx.x * src1_nb3) + blockIdx.y * splitD * sizeof(float));
    const float *dt_block = (const float *)((const char *)src2 + (blockIdx.x * src2_nb2) + blockIdx.y * splitD * sizeof(float));
    const float *A_block = (const float *)((const char *)src3 + blockIdx.y * splitD * src3_nb1);
    const float *B_block = (const float *)((const char *)src4 + (blockIdx.x * src4_nb3));
    const float *C_block = (const float *)((const char *)src5 + (blockIdx.x * src5_nb3));
    float *y_block = (float *)((char *)dst + (blockIdx.x * d_inner * L * sizeof(float)) + blockIdx.y * splitD * sizeof(float));
    float *s_block = (float *)((char *)dst + s_off + blockIdx.x * src0_nb3 + blockIdx.y * splitD * src0_nb2);

    const int stride_x = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B = src4_nb2 / sizeof(float);
    const int stride_C = src5_nb2 / sizeof(float);
    const int stride_y = d_inner;

    float regA[N];
    float regs0[N];

    __shared__ float smemB[N];
    __shared__ float smemC[N];

#ifdef USE_CUB
    using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    union CubTempStorage {
        typename BlockLoad::TempStorage load_temp;
        typename BlockStore::TempStorage store_temp;
    };
    __shared__ CubTempStorage cub_temp_storage;

    BlockLoad(cub_temp_storage.load_temp).Load(A_block, regA);
    __syncthreads();
    BlockLoad(cub_temp_storage.load_temp).Load(s0_block, regs0);
#else
    const int stride_s0 = src0_nb2 / sizeof(float);
    const int stride_A = src3_nb1 / sizeof(float);
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        regA[n] = A_block[threadIdx.x * stride_A + n];
        regs0[n] = s0_block[threadIdx.x * stride_s0 + n];
    }
#endif

#pragma unroll
    for (size_t i = 0; i < L; i++)
    {
        if (threadIdx.x < N)
        {
            smemB[threadIdx.x] = B_block[i * stride_B + threadIdx.x];
            smemC[threadIdx.x] = C_block[i * stride_C + threadIdx.x];
        }
        __syncthreads();

        float dt_soft_plus = dt_block[i * stride_dt + threadIdx.x];
        if (dt_soft_plus <= 20.0f)
        {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + threadIdx.x] * dt_soft_plus;

        float sumf = 0.0f;
#pragma unroll
        for (size_t n = 0; n < N; n++)
        {
            float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_block[i * stride_y + threadIdx.x] = sumf;
        __syncthreads();
    }

#ifdef USE_CUB
    BlockStore(cub_temp_storage.store_temp).Store(s_block, regs0);
#else
    const int stride_s = stride_s0;
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        s_block[threadIdx.x * stride_s + n] = regs0[n];
    }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// assumes as many threads as d_state
template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    ssm_scan_f32_group(
        const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
        const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
        const int32_t * src6_ptr, float * dst_ptr,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        char * s_base, const int64_t s_slot_bytes, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok, const int64_t K) {
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
    const float   * GGML_CUDA_RESTRICT src4 = src4_ptr;
    const float   * GGML_CUDA_RESTRICT src5 = src5_ptr;
    const int32_t * GGML_CUDA_RESTRICT src6 = src6_ptr;
    float         * GGML_CUDA_RESTRICT dst  = dst_ptr;

    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x  * c_factor + warp;

    ggml_cuda_pdl_sync();

    // the last block can have unused warps when n_head*d_head is not a multiple of c_factor
    if (warp_idx >= n_head * d_head) {
        return;
    }

    const int head_idx =  warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * sizeof(float);
    const int seq_idx  = blockIdx.y;

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    // TODO: refactor strides to be in elements/floats instead of bytes to be cleaner and consistent with the rest of the codebase
    const int source = src6 ? src6[seq_idx] : seq_idx;
    const float * s0_warp = (const float *) ((const char *) src0 + source * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
    const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    float *       s_warp  = (float *) (s_base + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    // strides across n_seq_tokens
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[c_factor];
    float state_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // NOTE: dt_soft_plus, dA and x_dt have the same value for a warp here.
        // Recalculation is intentional; sharing via shuffles/smem proved slower due to sync overhead.
        const float dt_soft_plus = (dt_warp[i * stride_dt] <= 20.0f ? log1pf(expf(dt_warp[i * stride_dt])) : dt_warp[i * stride_dt]);

        state_sum = 0.0f;
        const float dA   = expf(dt_soft_plus * A_warp[0]);
        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        // parallel accumulation for output
        state_sum = warp_reduce_sum(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        // Slot 0 is the final state written below; slots 1..K-1 are rollback snapshots.
        const int64_t slot = n_tok - 1 - i;
        if (K > 1 && slot > 0 && slot < K) {
            float * s_snapshot_warp = (float *) ((char *) s_warp + slot * s_slot_bytes);
#pragma unroll
            for (int j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

    // write back the state
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

static void ssm_scan_f32_cuda(const float * src0, const float * src1, const float * src2, const float * src3,
                              const float * src4, const float * src5, const int32_t * src6, float * dst,
                              const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1,
                              const int src2_nb2, const int src3_nb1, const int src4_nb2, const int src4_nb3, const int src5_nb2,
                              const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
                              const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
                              const int64_t K, const ggml_cuda_ssm_scan_fused_cache * cache, cudaStream_t stream) {
    // when fused, the states go straight into the recurrent cache and the dst tail is left alone
    char * const  s_base       = cache ? (char *) cache->data : (char *) dst + s_off;
    const int64_t s_slot_bytes = cache ? cache->slot_stride * (int64_t) sizeof(float) : n_seq * (int64_t) src0_nb3;

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    if (src3_nb1 == sizeof(float)) {
        // Mamba-2
        if (d_state == 64) { // Zamba2
            constexpr int threads   = 64;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<64/WARP_SIZE, 64>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_base, s_slot_bytes, n_head, head_dim, n_group, n_tok, K);
        } else if (d_state == 96) {
            constexpr int threads   = 96;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<96/WARP_SIZE, 96>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_base, s_slot_bytes, n_head, head_dim, n_group, n_tok, K);
        } else if (d_state == 128) {
            constexpr int threads   = 128;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<128/WARP_SIZE, 128>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_base, s_slot_bytes, n_head, head_dim, n_group, n_tok, K);
        } else if (d_state == 256) { // Falcon-H1
            constexpr int threads   = 256;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<256/WARP_SIZE, 256>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_base, s_slot_bytes, n_head, head_dim, n_group, n_tok, K);
        } else {
            GGML_ABORT("doesn't support d_state!=(64, 96, 128 or 256).");
        }
    } else {
        // Mamba-1
        GGML_ASSERT(K == 1);
        constexpr int threads = 128;
        GGML_ASSERT(n_head % threads == 0);
        GGML_ASSERT(head_dim == 1);
        GGML_ASSERT(n_group == 1);
        const dim3 blocks(n_seq, (n_head + threads - 1) / threads, 1);
        if (d_state == 16) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            switch (n_tok)
            {
            case 1:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 1>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 2:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 2>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 3:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 3>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 4:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 4>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 5:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 5>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 6:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 6>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 7:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 7>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 8:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 8>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            default:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 0>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            }
        } else {
            GGML_ABORT("doesn't support d_state!=16.");
        }
    }
}

#if !defined(GGML_USE_HIP)
// ============================================================================
// SSD (State Space Duality) kernels for Mamba-2 prefill (n_tok > SSM_SSD_MIN_TOKENS)
//
// Instead of a sequential scan, SSD reformulates the output as:
//   Y = (L (.) (C @ B^T)) @ (X * dt)  +  decay * C @ s_init
// where L is a causal decay mask derived from A and dt.
//
// This converts the O(T*N) sequential scan into parallel matmuls.
// ============================================================================
// Softplus(dt) and inclusive prefix sum per head using CUB BlockScan.
// Grid: (n_head, n_seqs)
template <int BLOCK_SIZE, int MAX_ITEMS>
__global__ void ssm_ssd_prepare_dt_kernel(
        const float * __restrict__ dt_raw,
        float * __restrict__ dt_sp_out,
        float * __restrict__ cs_out,
        const int n_head, const int n_tok,
        const int dt_stride_tok,   // elements between tokens in dt
        const int dt_stride_seq) { // elements between sequences in dt

    const int h = blockIdx.x;
    const int s = blockIdx.y;

    const float * dt_seq = dt_raw + s * dt_stride_seq;

    float * dt_sp_seq = dt_sp_out + s * n_tok * n_head;
    float * cs_seq    = cs_out    + s * n_tok * n_head;

    const int items_per_thread = (n_tok + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1: softplus with interleaved distribution (t = i*BLOCK_SIZE + threadIdx.x).
    // Each warp reads BLOCK_SIZE consecutive tokens, giving coalesced dt_raw loads
    // (stride n_head between threads vs. items_per_thread*n_head in blocked layout).
    float local_vals[MAX_ITEMS];
    for (int i = 0; i < items_per_thread; i++) {
        const int t = i * BLOCK_SIZE + threadIdx.x;
        if (t < n_tok) {
            float val = dt_seq[h + t * dt_stride_tok];
            float sp = (val <= 20.0f) ? log1pf(expf(val)) : val;
            local_vals[i] = sp;
            dt_sp_seq[t * n_head + h] = sp;
        } else {
            local_vals[i] = 0.0f;
        }
    }

    // Phase 2+3: per-step inclusive scan to build cs[] in token order.
    // With interleaved distribution the per-thread total scan would not give token-order
    // prefix sums, so we scan one BLOCK_SIZE slab at a time and carry a running total.
#ifdef USE_CUB
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    __shared__ float step_total;

    float running = 0.0f;
    for (int i = 0; i < items_per_thread; i++) {
        float inclusive;
        BlockScan(scan_temp).InclusiveSum(local_vals[i], inclusive);
        const int t = i * BLOCK_SIZE + threadIdx.x;
        if (t < n_tok) {
            cs_seq[t * n_head + h] = running + inclusive;
        }
        if (threadIdx.x == BLOCK_SIZE - 1) {
            step_total = inclusive;
        }
        __syncthreads();
        running += step_total;
    }
#else
    // Fallback: sequential prefix scan in shared memory, one slab at a time.
    __shared__ float sdata[BLOCK_SIZE];
    float running = 0.0f;
    for (int i = 0; i < items_per_thread; i++) {
        const int t = i * BLOCK_SIZE + threadIdx.x;
        sdata[threadIdx.x] = local_vals[i];
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int j = 1; j < BLOCK_SIZE; j++) {
                sdata[j] += sdata[j - 1];
            }
        }
        __syncthreads();
        if (t < n_tok) {
            cs_seq[t * n_head + h] = running + sdata[threadIdx.x];
        }
        running += sdata[BLOCK_SIZE - 1];
        __syncthreads();
    }
#endif
}

// Prepare SSD matmul inputs for one chunk: X_dt, B_weighted, C_scaled.
// T_matmul controls precision for X_dt, B_weighted (float or half).
// C_scaled is always float (pairs with float s_cur in step 3c).
// Computation is always FP32; only the final store converts to T_matmul.
// Also materializes the causal M matrix = exp(A*(cs_out - cs_in)) * CB (fused with prep to save a launch).
// Grid: (ceil(max(C*head_dim, d_state*C, chunk_len^2) / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE, typename T_matmul>
__global__ void ssm_ssd_pre_matmul_kernel(
        const float * __restrict__ cs,         // {n_tok, n_head} cumulative dt sums
        const float * __restrict__ dt_sp,      // {n_tok, n_head} softplus(dt)
        const float * __restrict__ A,          // {1, n_head}
        const float * __restrict__ x,          // {head_dim, n_head, n_tok, n_seqs}
        const float * __restrict__ B,          // {d_state, n_group, n_tok, n_seqs}
        const float * __restrict__ C_src,      // {d_state, n_group, n_tok, n_seqs}
        T_matmul * __restrict__ X_dt,          // {head_dim, C, n_head} x * dt, d-fastest
        T_matmul * __restrict__ B_weighted,    // {d_state, C, n_head} B * decay_from_end
        float * __restrict__ C_scaled,         // {d_state, C, n_head} C * decay_to_pos (always float)
        const float * __restrict__ CB,         // {chunk_len, chunk_len, n_group, n_seqs}
        half * __restrict__ M_out,             // {chunk_len, chunk_len, n_head, n_seqs}
        const int chunk_len, const int head_dim, const int n_head, const int n_group,
        const int d_state, const int A_stride,
        const int x_stride_tok, const int x_stride_seq,
        const int B_stride_tok, const int B_stride_seq,
        const int C_stride_tok, const int C_stride_seq,
        const int chunk_offset,
        const int n_tok_total) {

    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int g = h / (n_head / n_group);

    const float A_h = A[h * A_stride];
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const int cs_seq_off = s * n_tok_total * n_head;
    const float cs_base = (chunk_offset > 0) ? cs[cs_seq_off + (chunk_offset - 1) * n_head + h] : 0.0f;
    const float cs_last = cs[cs_seq_off + (chunk_offset + chunk_len - 1) * n_head + h] - cs_base;

    // Prepare X_dt = x * dt, stored d-fastest for coalesced reads and writes.
    const int n_xdt = chunk_len * head_dim;
    if (idx < n_xdt) {
        const int d = idx % head_dim;
        const int t = idx / head_dim;

        const float x_val = x[s * x_stride_seq + (chunk_offset + t) * x_stride_tok + d + h * head_dim];
        const float dt_val = dt_sp[cs_seq_off + (chunk_offset + t) * n_head + h];

        X_dt[d + t * head_dim + h * n_xdt + s * n_xdt * n_head] = (T_matmul)(x_val * dt_val);
    }

    // Prepare B_weighted and C_scaled together: both share the same index space (d_state * chunk_len)
    // and the same cs_t load, so merging halves the cs[] global memory traffic.
    const int n_bw = d_state * chunk_len;
    if (idx < n_bw) {
        const int n = idx % d_state;
        const int t = idx / d_state;

        const float cs_t = cs[cs_seq_off + (chunk_offset + t) * n_head + h] - cs_base;

        const float B_val = B[s * B_stride_seq + (chunk_offset + t) * B_stride_tok + g * d_state + n];
        B_weighted[n + t * d_state + h * n_bw + s * n_bw * n_head] = (T_matmul)(B_val * __expf(A_h * (cs_last - cs_t)));

        const float C_val = C_src[s * C_stride_seq + (chunk_offset + t) * C_stride_tok + g * d_state + n];
        C_scaled[n + t * d_state + h * n_bw + s * n_bw * n_head] = C_val * __expf(A_h * cs_t);
    }

    // Materialize M = exp(A*(cs_out - cs_in)) * CB with causal mask.
    const int n_M = chunk_len * chunk_len;
    if (idx < n_M) {
        const int t_out = idx % chunk_len;
        const int t_in  = idx / chunk_len;

        half val;
        if (t_in <= t_out) {
            const float cs_out = cs[cs_seq_off + (chunk_offset + t_out) * n_head + h] - cs_base;
            const float cs_in  = cs[cs_seq_off + (chunk_offset + t_in)  * n_head + h] - cs_base;
            const float decay  = __expf(A_h * (cs_out - cs_in));
            const float * CB_g = CB + (int64_t)s * chunk_len * chunk_len * n_group
                                   + (int64_t)g * chunk_len * chunk_len;
            const float cb_val = CB_g[t_out + t_in * chunk_len];
            val = __float2half(decay * cb_val);
        } else {
            val = __float2half(0.0f);
        }

        M_out[(int64_t)s * n_M * n_head + (int64_t)h * n_M + t_in * chunk_len + t_out] = val;
    }
}

// Scale running state in-place: s_cur *= decay_total(chunk).
// Called BEFORE cuBLAS state update (beta=1) to fuse inter-chunk decay.
// Eliminates the s_old buffer and D2D memcpy vs the old approach of:
//   memcpy(s_old, s_cur) -> cuBLAS(beta=0) -> s_cur += decay * s_old
// Grid: (ceil(d_state * head_dim / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE>
__global__ void ssm_ssd_scale_state_kernel(
        float * __restrict__ s_cur,            // {d_state, head_dim, n_head, n_seqs}
        const float * __restrict__ cs,         // {n_tok, n_head} cumulative dt sums
        const float * __restrict__ A,          // {1, n_head}
        const int d_state, const int head_dim, const int n_head,
        const int chunk_offset, const int chunk_len,
        const int n_tok_total, const int A_stride) {

    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int state_per_head = d_state * head_dim;
    if (idx >= state_per_head) return;

    const float A_h = A[h * A_stride];
    const int cs_seq_off = s * n_tok_total * n_head;
    const float cs_base = (chunk_offset > 0) ? cs[cs_seq_off + (chunk_offset - 1) * n_head + h] : 0.0f;
    const float cs_last = cs[cs_seq_off + (chunk_offset + chunk_len - 1) * n_head + h] - cs_base;
    const float decay_total = __expf(A_h * cs_last);

    const int off = s * state_per_head * n_head + h * state_per_head + idx;
    s_cur[off] *= decay_total;
}

// Copy initial state from src0[ids[s]] into s_cur for each sequence.
// src0 and s_cur can alias when the state is written straight into the cache.
// Grid: (ceil(d_state * head_dim * n_head / BLOCK), n_seqs)
template <int BLOCK_SIZE>
__global__ void ssm_ssd_init_state_kernel(
        const float * src0,                    // {d_state, head_dim, n_head, n_rs}
        const int32_t * __restrict__ ids,      // {n_seqs}
        float * s_cur,                         // {d_state, head_dim, n_head, n_seqs}
        const int state_size,                  // d_state * head_dim * n_head
        const int64_t s0_stride_seq) {         // elements between state rows
    const int s = blockIdx.y;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= state_size) return;

    const float * s_src = src0 + (int64_t)(ids ? ids[s] : s) * s0_stride_seq;
    s_cur[s * state_size + idx] = s_src[idx];
}

// SSD (State Space Duality) dispatch for Mamba-2 prefill.
// Chunked matmuls: CB, materialize M + cuBLAS Y, S@C, B@X_dt.
// All strides are in elements (floats), not bytes.
static void ssm_scan_ssd_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const float * src0_d, const float * src1_d, const float * src2_d, const float * src3_d,
        const float * src4_d, const float * src5_d, const int32_t * src6_d, float * dst_d,
        const int64_t s0_stride_seq,                                   // state (src0) stride between seqs
        const int x_stride_tok,  const int x_stride_seq,               // x (src1) strides
        const int dt_stride_tok, const int dt_stride_seq,              // dt (src2) strides
        const int A_stride,                                            // A (src3) stride between heads
        const int B_stride_tok,  const int B_stride_seq,               // B (src4) strides
        const int C_stride_tok,  const int C_stride_seq,               // C (src5) strides
        float * s_cur,                                                 // state: dst state tail, or the cache when fused
        const int64_t d_state, const int64_t head_dim,
        const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq) {

    cudaStream_t stream = ctx.stream();
    const int64_t d_inner = head_dim * n_head;

    const int64_t chunk_size = SSM_SSD_CHUNK_SIZE;
    const int64_t n_chunks = (n_tok + chunk_size - 1) / chunk_size;

    const int64_t state_per_head = d_state * head_dim;

    using matmul_t = half;
    static constexpr cudaDataType_t matmul_dtype = CUDA_R_16F;

    ggml_cuda_pool_alloc<float>    dt_sp_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    cs_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    CB_buf(ctx.pool(), chunk_size * chunk_size * n_group * n_seq);
    ggml_cuda_pool_alloc<matmul_t> X_dt_buf(ctx.pool(), chunk_size * head_dim * n_head * n_seq);
    ggml_cuda_pool_alloc<matmul_t> B_w_buf(ctx.pool(), d_state * chunk_size * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    C_s_buf(ctx.pool(), d_state * chunk_size * n_head * n_seq);
    float    * dt_sp      = dt_sp_buf.get();
    float    * cs         = cs_buf.get();
    float    * CB         = CB_buf.get();
    matmul_t * X_dt       = X_dt_buf.get();
    matmul_t * B_weighted = B_w_buf.get();
    float    * C_scaled   = C_s_buf.get();

    // Step 1: softplus(dt) and parallel prefix sum over full sequence
    {
        dim3 grid(n_head, n_seq);
        ssm_ssd_prepare_dt_kernel<SSM_SSD_DT_BLOCK, SSM_SSD_DT_MAX_ITEMS><<<grid, SSM_SSD_DT_BLOCK, 0, stream>>>(
            src2_d, dt_sp, cs, n_head, n_tok, dt_stride_tok, dt_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: initialize running state from src0[ids[s]]
    {
        constexpr int BLOCK = 256;
        const int64_t state_size = d_state * head_dim * n_head;
        dim3 grid((state_size + BLOCK - 1) / BLOCK, n_seq);
        ssm_ssd_init_state_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
            src0_d, src6_d, s_cur, state_size, s0_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: chunked SSD loop
    // Per chunk: pre_matmul (incl. M) + 4 cuBLAS (CB, Y, S@C, state update) + scale_state
    cublasHandle_t handle = ctx.cublas_handle();
    const float alpha_one  = 1.0f;
    const float beta_zero  = 0.0f;
    const float beta_one   = 1.0f;
    const int lda_C_src = C_stride_tok;  // leading dim for C in CB = C^T @ B
    const int ldb_B_src = B_stride_tok;  // leading dim for B in CB = C^T @ B

    // Scratch buffer for causal M matrix, reused across chunks (max size at chunk_size)
    const int64_t n_M_max = chunk_size * chunk_size;
    ggml_cuda_pool_alloc<half> M_buf(ctx.pool(), n_M_max * n_head * n_seq);
    half * M_mat = M_buf.get();

    for (int64_t k = 0; k < n_chunks; k++) {
        const int64_t chunk_offset = k * chunk_size;
        const int64_t chunk_len = (chunk_offset + chunk_size <= n_tok) ? chunk_size : (n_tok - chunk_offset);

        // 3a: CB = C^T @ B per group
        for (int64_t s = 0; s < n_seq; s++) {
            const float * C_s = src5_d + s * C_stride_seq + chunk_offset * C_stride_tok;
            const float * B_s = src4_d + s * B_stride_seq + chunk_offset * B_stride_tok;
            float *      CB_s = CB + s * chunk_len * chunk_len * n_group;

            if (n_group == 1) {
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    chunk_len, chunk_len, d_state,
                    &alpha_one, C_s, lda_C_src, B_s, ldb_B_src,
                    &beta_zero, CB_s, (int)chunk_len));
            } else {
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    chunk_len, chunk_len, d_state,
                    &alpha_one,
                    C_s, CUDA_R_32F, lda_C_src, d_state,
                    B_s, CUDA_R_32F, ldb_B_src, d_state,
                    &beta_zero,
                    CB_s, CUDA_R_32F, (int)chunk_len, (long long)(chunk_len * chunk_len),
                    n_group,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3b: prepare X_dt, B_weighted, C_scaled + materialize causal M matrix
        const int64_t n_M = chunk_len * chunk_len;
        {
            constexpr int BLOCK = 256;
            const int64_t n_xdt   = chunk_len * head_dim;
            const int64_t n_bw    = d_state * chunk_len;
            int64_t max_work = n_xdt;
            if (n_bw  > max_work) max_work = n_bw;
            if (n_M   > max_work) max_work = n_M;
            dim3 grid((max_work + BLOCK - 1) / BLOCK, n_head, n_seq);
            ssm_ssd_pre_matmul_kernel<BLOCK, matmul_t><<<grid, BLOCK, 0, stream>>>(
                cs, dt_sp, src3_d, src1_d, src4_d, src5_d,
                X_dt, B_weighted, C_scaled,
                CB, M_mat,
                chunk_len, head_dim, n_head, n_group, d_state, A_stride,
                x_stride_tok, x_stride_seq, B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq,
                chunk_offset, n_tok);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3c: dst = S_cur^T @ C_scaled (state contribution)
        {
            const int64_t stride_S  = state_per_head;
            const int64_t stride_Cs = d_state * chunk_len;

            for (int64_t s = 0; s < n_seq; s++) {
                float * dst_chunk = dst_d + s * d_inner * n_tok + chunk_offset * d_inner;

                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    head_dim, chunk_len, d_state,
                    &alpha_one,
                    s_cur    + s * stride_S  * n_head, CUDA_R_32F, d_state, stride_S,
                    C_scaled + s * stride_Cs * n_head, CUDA_R_32F, d_state, stride_Cs,
                    &beta_zero,
                    dst_chunk, CUDA_R_32F, d_inner, head_dim,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3d: dst += X_dt @ M^T (intra-chunk contribution, adds to 3c result)
        // M is stored as M[t_out, t_in] (lower-triangular), transpose needed for Y = X @ M^T.
        {
            const int64_t stride_M = n_M;
            const int64_t stride_X_h = (int64_t)chunk_len * head_dim;

            for (int64_t s = 0; s < n_seq; s++) {
                float * dst_chunk = dst_d + s * d_inner * n_tok + chunk_offset * d_inner;
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    head_dim, chunk_len, chunk_len,
                    &alpha_one,
                    X_dt       + s * stride_X_h * n_head, matmul_dtype, head_dim, stride_X_h,
                    M_mat      + s * stride_M   * n_head, matmul_dtype, chunk_len, stride_M,
                    &beta_one,
                    dst_chunk, CUDA_R_32F, d_inner, head_dim,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3e: s_cur = B_weighted @ X_dt^T + decay_total * s_cur_old (state update)
        {
            // Scale s_cur in-place by per-head decay_total BEFORE cuBLAS overwrites it
            constexpr int BLOCK = 256;
            dim3 grid((state_per_head + BLOCK - 1) / BLOCK, n_head, n_seq);
            ssm_ssd_scale_state_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
                s_cur, cs, src3_d,
                d_state, head_dim, n_head,
                chunk_offset, chunk_len, n_tok, A_stride);
            CUDA_CHECK(cudaGetLastError());

            // cuBLAS with beta=1: s_cur = B_weighted @ X_dt^T + 1.0 * s_cur (already scaled)
            const int64_t stride_Bw = d_state * chunk_len;
            const int64_t stride_X  = chunk_len * head_dim;
            const int64_t stride_S  = state_per_head;

            for (int64_t s = 0; s < n_seq; s++) {
                // X_dt is d-fastest {hd, C}, read as OP_T to get {C, hd}
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    d_state, head_dim, chunk_len,
                    &alpha_one,
                    B_weighted + s * stride_Bw * n_head, matmul_dtype, d_state, stride_Bw,
                    X_dt       + s * stride_X  * n_head, matmul_dtype, head_dim, stride_X,
                    &beta_one,
                    s_cur      + s * stride_S  * n_head, CUDA_R_32F, d_state, stride_S,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
    }
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_cuda_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                               const ggml_cuda_ssm_scan_fused_cache * cache,
                               const float * staged = nullptr, const int32_t * staged_ids = nullptr) {
    const struct ggml_tensor * src0 = dst->src[0];  // s
    const struct ggml_tensor * src1 = dst->src[1];  // x
    const struct ggml_tensor * src2 = dst->src[2];  // dt
    const struct ggml_tensor * src3 = dst->src[3];  // A
    const struct ggml_tensor * src4 = dst->src[4];  // B
    const struct ggml_tensor * src5 = dst->src[5];  // C
    const struct ggml_tensor * src6 = dst->src[6];  // ids

    const int64_t nc  = src4->ne[0];  // d_state
    const int64_t nr  = src1->ne[0];  // head_dim or 1
    const int64_t nh  = src1->ne[1];  // n_head
    const int64_t ng  = src4->ne[1];  // n_group
    const int64_t n_t = src1->ne[2];  // number of tokens per sequence
    const int64_t n_s = src1->ne[3];  // number of sequences in the batch
    const int32_t K_param = ggml_get_op_params_i32(dst, 0);
    const int64_t K = K_param > 0 ? K_param : 1;

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    const int64_t n_scan = ggml_nelements(src1) + K*nc*nr*nh*n_s;
    GGML_ASSERT(staged ? (K == 1 && n_scan <= ggml_nelements(dst)) : n_scan == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    const float * src0_d = staged ? staged : (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    const float * src3_d = (const float *) src3->data;
    const float * src4_d = (const float *) src4->data;
    const float * src5_d = (const float *) src5->data;
    const int32_t * src6_d = staged ? staged_ids : (const int32_t *) src6->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Byte strides are narrowed to int for both scan and SSD paths.
    const size_t state_nb2 = staged ? nc*nr*sizeof(float) : src0->nb[2];
    const size_t state_nb3 = staged ? state_nb2*nh : src0->nb[3];
    GGML_ASSERT(state_nb2 <= (size_t)INT_MAX);
    GGML_ASSERT(state_nb3 <= (size_t)INT_MAX);
    GGML_ASSERT(src1->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src1->nb[3] <= (size_t)INT_MAX);
    GGML_ASSERT(src2->nb[1] <= (size_t)INT_MAX);
    GGML_ASSERT(src2->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src3->nb[1] <= (size_t)INT_MAX);
    GGML_ASSERT(src4->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src4->nb[3] <= (size_t)INT_MAX);
    GGML_ASSERT(src5->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src5->nb[3] <= (size_t)INT_MAX);

#if !defined(GGML_USE_HIP)
    // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
    // Requires NVIDIA Turing+ otherwise fallback to scan.
    const bool is_mamba2 = (src3->nb[1] == sizeof(float));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_ssd = is_mamba2 && n_t > SSM_SSD_MIN_TOKENS
                      && K == 1
                      && n_t <= SSM_SSD_MAX_TOKENS
                      && GGML_CUDA_CC_IS_NVIDIA(cc)
                      && cc >= GGML_CUDA_CC_TURING
                      && nr % 8 == 0;  // cuBLAS requires 8-element (16-byte) alignment

    if (use_ssd) {
        // ssm_ssd_init_state_kernel uses flat linear indexing within each sequence,
        // so src0 must be fully contiguous across all inner dimensions.
        // The scan path handles non-contiguous nb[2] via src0_nb2 but does not handle nb[1].
        GGML_ASSERT(staged || src0->nb[1] == nc * sizeof(float));
        GGML_ASSERT(state_nb2 == nc * nr * sizeof(float));

        ssm_scan_ssd_f32_cuda(ctx,
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            (int64_t)(state_nb3 / sizeof(float)),
            (int)(src1->nb[2] / sizeof(float)), (int)(src1->nb[3] / sizeof(float)),
            (int)(src2->nb[1] / sizeof(float)), (int)(src2->nb[2] / sizeof(float)),
            (int)(src3->nb[1] / sizeof(float)),
            (int)(src4->nb[2] / sizeof(float)), (int)(src4->nb[3] / sizeof(float)),
            (int)(src5->nb[2] / sizeof(float)), (int)(src5->nb[3] / sizeof(float)),
            cache ? cache->data : (float *) ((char *) dst_d + s_off), nc, nr, nh, ng, n_t, n_s);
        return;
    }
#endif
    ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                      state_nb2, state_nb3, src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
                      src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                      s_off, nc, nr, nh, ng, n_t, n_s, K, cache, stream);
}

struct ssm_replay_params {
    int64_t d_state, head_dim, n_head, n_group, window, slots;
    int64_t x_seq, dt_seq, B_seq, C_seq; // the first two dimensions of x, B and C are contiguous
    int64_t history_off[4]; // x, dt, B column offsets and total width per token
};

struct ssm_replay_history {
    const float * x;
    const float * dt;
    const float * B;
};

static __device__ int ssm_replay_count(const int32_t * ids, int seq, const ssm_replay_params & p) {
    const int count = ids[gridDim.y + seq];
    if (ids[seq] < 0 || ids[seq] >= p.slots || count < 0 || count >= p.window) {
        __trap();
    }
    return count;
}

static __global__ void ssm_replay_precompute(
        const float * B_history, const float * B, const float * C, const int32_t * ids,
        float * bc, const ssm_replay_params p) {
    const int seq = blockIdx.y;
    const int group = blockIdx.x;
    const int count = ssm_replay_count(ids, seq, p);
    if (count + 1 == p.window) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t first = ids[seq] * p.window;
    const float * q = C + seq * p.C_seq + group * p.d_state;
    const int j = blockIdx.z * (blockDim.x / WARP_SIZE) + warp;
    if (j <= count) {
        const float * key = j == count ? B + seq * p.B_seq + group * p.d_state :
            B_history + ((first + j) * p.n_group + group) * p.d_state;
        float dot = 0.0f;
        for (int i = lane; i < p.d_state; i += WARP_SIZE) {
            dot += key[i] * q[i];
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            bc[(seq * p.n_group + group) * p.window + j] = dot;
        }
    }
}

template <int d_state>
static __global__ void ssm_replay_scan(
        const float * s, const float * x, const float * dt, const float * A,
        const float * B, const float * C, const int32_t * ids, const float * bc,
        float * y, float * out, const float * cache, int32_t * identity, const ssm_replay_history history,
        const ssm_replay_params p, const uint3 tiles_fd, const uint3 heads_per_group_fd, const bool materialize) {
    const int seq  = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const uint2 tile = fast_div_modulo(blockIdx.x, tiles_fd);
    const int head = tile.x;
    const int64_t channel = (int64_t) tile.y * (blockDim.x / WARP_SIZE) + warp;
    const int group = fastdiv(head, heads_per_group_fd);
    const int64_t inner = p.head_dim * p.n_head;
    const int64_t D = p.d_state * inner;
    const int count = ssm_replay_count(ids, seq, p);
    const int64_t first = ids[seq] * p.window;
    const float * src = s + ids[seq] * D;
    float * dst = out + seq * D;
    if (materialize && blockIdx.x == 0 && threadIdx.x == 0) {
        identity[seq] = seq;
    }
    const int n_steps = materialize ? count : count + 1;
    const bool flush = materialize || n_steps == p.window;
    const float * q = C + seq * p.C_seq + group * p.d_state;
    const float * k = B + seq * p.B_seq + group * p.d_state;
    const float * v = x + seq * p.x_seq + head * p.head_dim;

    __shared__ float scale[WARP_SIZE];
    __shared__ float total_decay;
    if (warp == 0) {
        // lane l holds step n_steps - 1 - l: the sum of the lanes below it is the dt after that step (total - prefix loses it when |A| is large)
        const int j = n_steps - 1 - lane;
        float step = 0.0f;
        if (j >= 0 && j == count) {
            const float dt_cur = dt[seq * p.dt_seq + head];
            step = dt_cur <= 20.0f ? log1pf(expf(dt_cur)) : dt_cur;
        } else if (j >= 0) {
            step = history.dt[(first + j) * p.n_head + head];
        }
        const float prefix = warp_prefix_inclusive_sum(step);
        const float below  = ggml_cuda_shfl_up(prefix, 1, WARP_SIZE);
        const float a = A[head];
        if (j >= 0) {
            scale[j] = step * expf(a * (lane == 0 ? 0.0f : below));
        }
        if (lane == WARP_SIZE - 1) {
            total_decay = expf(a * prefix);
        }
    }
    __syncthreads();

    if (channel < p.head_dim) {
        const int64_t col = head * p.head_dim + channel;
        float state[d_state / WARP_SIZE];
#pragma unroll
        for (int i = 0; i < d_state / WARP_SIZE; ++i) {
            const int64_t offset = col * d_state + i * WARP_SIZE + lane;
            state[i] = src[offset];
            if (!flush && src != (cache ? cache + seq * D : dst)) {
                dst[offset] = state[i];
            }
        }
        if (flush && n_steps > 0) {
            float delta[d_state / WARP_SIZE] = {};
            for (int j = 0; j < n_steps; ++j) {
                const float * key = j == count ? k : history.B + ((first + j) * p.n_group + group) * d_state;
                const float val = j == count ? v[channel] : history.x[(first + j) * inner + col];
#pragma unroll
                for (int i = 0; i < d_state / WARP_SIZE; ++i) {
                    const float scaled_B = key[i * WARP_SIZE + lane] * scale[j];
                    delta[i] = fmaf(val, scaled_B, delta[i]);
                }
            }
#pragma unroll
            for (int i = 0; i < d_state / WARP_SIZE; ++i) {
                state[i] = fmaf(state[i], total_decay, delta[i]);
            }
        }
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < d_state / WARP_SIZE; ++i) {
            if (!materialize) {
                dot += state[i] * q[i * WARP_SIZE + lane];
            }
            if (flush) {
                dst[col * d_state + i * WARP_SIZE + lane] = state[i];
            }
        }
        if (!materialize) {
            dot = warp_reduce_sum(dot);
            if (lane == 0) {
                if (!flush) {
                    float update = 0.0f;
                    for (int j = 0; j < n_steps; ++j) {
                        const float val = j == count ? v[channel] : history.x[(first + j) * inner + col];
                        update += val * (scale[j] * bc[(seq * p.n_group + group) * p.window + j]);
                    }
                    dot = fmaf(dot, total_decay, update);
                }
                y[seq * inner + col] = dot;
            }
        }
    }
}

// Keep kernel arguments in parameter memory: a dynamic index can copy the whole struct to each thread's stack.
template <typename T>
static __device__ __forceinline__ T ssm_replay_field(int field, T x, T dt, T B) {
    return field == 0 ? x : field == 1 ? dt : B;
}

static __global__ void ssm_replay_store(
        const float * x, const float * dt, const float * B, const int32_t * ids,
        const ssm_replay_history history, float * out, ggml_cuda_ssm_scan_fused_cache cache, const ssm_replay_params p) {
    const int seq = blockIdx.y;
    const int count = ssm_replay_count(ids, seq, p);
    const bool flush = count + 1 == p.window;
    const int field = blockIdx.z;
    const int64_t offset = ssm_replay_field(field, p.history_off[0], p.history_off[1], p.history_off[2]);
    const int64_t width = ssm_replay_field(field, p.history_off[1], p.history_off[2], p.history_off[3]) - offset;
    const int64_t row = width * p.window;
    const float * src = ssm_replay_field(field, history.x, history.dt, history.B) + ids[seq] * row;
    float * target = ssm_replay_field(field, cache.history[0], cache.history[1], cache.history[2]);
    target = target ? target + seq * row : nullptr;
    float * dst = gridDim.y == 1 && target ? target : out + offset * p.window * gridDim.y + seq * row;
    const bool append = src == target && !flush;
    const int64_t begin = append ? count * width : 0;
    const int64_t end = append ? begin + width : row;
    for (int64_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        if (flush) {
            dst[i] = 0.0f;
        } else if (i < count * width || i >= (count + 1) * width) {
            dst[i] = src[i];
        } else {
            const int64_t j = i - count * width;
            if (field == 0) {
                dst[i] = x[seq * p.x_seq + j];
            } else if (field == 1) {
                const float val = dt[seq * p.dt_seq + j];
                dst[i] = val <= 20.0f ? log1pf(expf(val)) : val;
            } else {
                dst[i] = B[seq * p.B_seq + j];
            }
        }
    }
    if (field == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        out[p.history_off[3] * p.window * gridDim.y + seq] = flush ? 0.0f : (float) (count + 1);
    }
}

// Commit batched writes after all sources have been read. Stationary rows only append one history record.
static __global__ void ssm_replay_commit(
        const float * staged, const float * s, const int32_t * ids, ggml_cuda_ssm_scan_fused_cache cache, const ssm_replay_params p) {
    const int seq = blockIdx.y;
    const int count = ids[gridDim.y + seq];
    const int64_t D = p.d_state * p.head_dim * p.n_head;
    const bool append = count + 1 != p.window && s + ids[seq] * D == cache.data + seq * D;
    const float * src = staged + seq * D;
    float * dst = cache.data + seq * D;
    int64_t begin = 0, end = D;
    if (blockIdx.z == 0) {
        if (append) {
            return;
        }
        // Copy a whole checkpoint on flush or slot movement. All supported state widths are divisible by four.
        if (((uintptr_t) src | (uintptr_t) dst) % alignof(float4) == 0) {
            for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < D / 4; i += blockDim.x * gridDim.x) {
                ((float4 *) dst)[i] = ((const float4 *) src)[i];
            }
            return;
        }
    } else {
        const int field = blockIdx.z - 1;
        const int64_t offset = ssm_replay_field(field, p.history_off[0], p.history_off[1], p.history_off[2]);
        const int64_t width = ssm_replay_field(field, p.history_off[1], p.history_off[2], p.history_off[3]) - offset;
        src = staged + (D + offset * p.window) * gridDim.y + seq * width * p.window;
        dst = ssm_replay_field(field, cache.history[0], cache.history[1], cache.history[2]) + seq * width * p.window;
        begin = append ? count * width : 0;
        end = append ? begin + width : width * p.window;
    }
    for (int64_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

template<int d_state>
static __global__ void ssm_replay_checkpoint(float * s, const float * A, const int32_t * counts, ssm_replay_history hist,
            int64_t capacity, int64_t heads, int64_t width, int64_t groups, int window, int nt) {
    const int seq = blockIdx.z, head = blockIdx.y, h = counts[seq];
    if (h < 0 || (h > window && h > capacity-window)) { __trap(); }
    if (h == 0 || (nt <= capacity-window && h + (int64_t)nt <= window)) { return; }
    const int lane = threadIdx.x%WARP_SIZE, group = head/(heads/groups);
    const int warps = blockDim.x/WARP_SIZE;
    for (int64_t m = blockIdx.x*warps + threadIdx.x/WARP_SIZE; m < width; m += gridDim.x*warps) {
        const int64_t col = head*width + m;
        float * dst = s + (seq*heads*width+col)*d_state + lane;
        float state[d_state/WARP_SIZE];
#pragma unroll
        for (int n = 0; n < d_state/WARP_SIZE; ++n) { state[n] = dst[n*WARP_SIZE]; }
        for (int j = 0; j < h; ++j) {
            const int64_t t = seq*capacity + j;
            const float dt = hist.dt[t*heads+head], decay = expf(A[head]*dt), xdt = hist.x[t*heads*width+col]*dt;
            const float * b = hist.B + (t*groups+group)*d_state + lane;
#pragma unroll
            for (int n = 0; n < d_state/WARP_SIZE; ++n) { state[n] = state[n]*decay + b[n*WARP_SIZE]*xdt; }
        }
#pragma unroll
        for (int n = 0; n < d_state/WARP_SIZE; ++n) { dst[n*WARP_SIZE] = state[n]; }
    }
}

void ggml_cuda_op_ssm_replay_checkpoint(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * x = dst->src[3], * dt = dst->src[4], * B = dst->src[5];
    const int64_t heads = dst->src[1]->ne[1], capacity = dt->ne[0]/heads, inner = x->ne[0]/capacity;
    const int64_t state = dst->ne[0]/inner, groups = B->ne[0]/(capacity*state);
    const ssm_replay_history hist = { (const float *)x->data, (const float *)dt->data, (const float *)B->data };
    const auto kernel = state == 64 ? ssm_replay_checkpoint<64> : state == 96 ? ssm_replay_checkpoint<96> :
                        state == 128 ? ssm_replay_checkpoint<128> : ssm_replay_checkpoint<256>;
    constexpr int warps = 4;
    const int64_t nheads = heads*dst->ne[1];
    // Spread small batches over the SMs and keep one block per head when the batch already fills the device.
    const int64_t blocks = 4*ggml_cuda_info().devices[ctx.device].nsm;
    const int tiles = std::min((inner/heads + warps-1)/warps, (blocks + nheads-1)/nheads);
    kernel<<<dim3(tiles,heads,dst->ne[1]), warps*WARP_SIZE, 0, ctx.stream()>>>((float *)dst->data,
            (const float *)dst->src[1]->data, (const int32_t *)dst->src[2]->data, hist, capacity, heads, inner/heads,
            groups, ggml_get_op_params_i32(dst,0), ggml_get_op_params_i32(dst,1));
}

struct ssm_verify {
    ssm_replay_params p;
    ssm_replay_history live, hist;
    const float * s, * A, * C;
    const int32_t * ids;
    float * y, * next, * cb, * scales;
    int64_t ns, nt, capacity, span, x_token, dt_token, b_token, c_token;

    template<int field>
    __device__ const float * feature(int seq, int h, int64_t j) const {
        if (j < h) {
            return ssm_replay_field(field, hist.x, hist.dt, hist.B) +
                (seq*capacity + j)*(p.history_off[field + 1] - p.history_off[field]);
        }
        return ssm_replay_field(field, live.x, live.dt, live.B) +
            seq*ssm_replay_field(field, p.x_seq, p.dt_seq, p.B_seq) +
            (j-h)*ssm_replay_field(field, x_token, dt_token, b_token);
    }
    __device__ float step(int seq, int h, int64_t j, int head) const {
        const float d = feature<1>(seq, h, j)[head];
        return j < h || d > 20.0f ? d : log1pf(expf(d));
    }

    template<int field>
    __device__ float * append(int seq, int t) const {
        return next + ns*nt*p.history_off[field] + (seq*nt+t)*(p.history_off[field+1]-p.history_off[field]);
    }
};

static __global__ void ssm_verify_coefficients(ssm_verify v) {
    const int seq = blockIdx.z, head = blockIdx.y, t = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    const int group = head/(v.p.n_head/v.p.n_group);
    const int h = v.ids[seq];
    const int first = h + v.nt > v.p.window ? h : 0;
    const int last = h + t;
    for (int m = threadIdx.x; m < v.p.head_dim; m += blockDim.x) {
        const int col = head*v.p.head_dim+m;
        v.append<0>(seq,t)[col] = v.feature<0>(seq,h,last)[col];
    }
    if (warp == 0) {
        float * scales = v.scales + ((seq*v.nt+t)*v.p.n_head+head)*(v.span+1);
        float suffix = 0;
        for (int end = last; end >= first; end -= WARP_SIZE) {
            const int j = end - lane;
            const float d = j >= first ? v.step(seq, h, j, head) : 0.0f;
            if (j == last) { v.append<1>(seq,t)[head] = d; }
            const float sum = warp_prefix_inclusive_sum(d);
            const float below = ggml_cuda_shfl_up(sum, 1, WARP_SIZE);
            if (j >= first) {
                scales[j-first] = d*expf(v.A[head]*(suffix + (lane == 0 ? 0.0f : below)));
            }
            suffix += ggml_cuda_shfl(sum, WARP_SIZE-1, WARP_SIZE);
        }
        if (lane == 0) { scales[v.span] = expf(v.A[head]*suffix); }
    }
    if (head % (v.p.n_head/v.p.n_group) == 0) {
        for (int n = threadIdx.x; n < v.p.d_state; n += blockDim.x) {
            const int k = group*v.p.d_state+n;
            v.append<2>(seq,t)[k] = v.feature<2>(seq,h,last)[k];
        }
        const float * c = v.C + seq*v.p.C_seq + t*v.c_token + group*v.p.d_state;
        float * cb = v.cb + ((seq*v.nt+t)*v.p.n_group+group)*v.span;
        for (int j = first+warp; j <= last; j += blockDim.x/WARP_SIZE) {
            const float * b = v.feature<2>(seq, h, j) + group*v.p.d_state;
            float dot = 0;
            for (int n = lane; n < v.p.d_state; n += WARP_SIZE) { dot += c[n]*b[n]; }
            dot = warp_reduce_sum(dot);
            if (lane == 0) { cb[j-first] = dot; }
        }
    }
}

template<int d_state>
static __global__ void ssm_verify_scan(ssm_verify v) {
    const int seq = blockIdx.y, lane = threadIdx.x % WARP_SIZE;
    const int col = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int64_t inner = v.p.head_dim*v.p.n_head;
    if (col >= inner) { return; }
    const int head = col/v.p.head_dim, group = head/(v.p.n_head/v.p.n_group);
    const int h = v.ids[seq];
    float state[d_state/WARP_SIZE];
#pragma unroll
    for (int n = 0; n < d_state/WARP_SIZE; ++n) {
        state[n] = v.s[(seq*inner+col)*d_state + n*WARP_SIZE+lane];
    }
    const int first = h + v.nt > v.p.window ? h : 0;
    for (int64_t t = 0; t < v.nt; ++t) {
        const float * c = v.C + seq*v.p.C_seq + t*v.c_token + group*d_state;
        const float * scales = v.scales + ((seq*v.nt+t)*v.p.n_head+head)*(v.span+1);
        const float * cb = v.cb + ((seq*v.nt+t)*v.p.n_group+group)*v.span;
        float anchor = 0, delta = 0;
#pragma unroll
        for (int n = 0; n < d_state/WARP_SIZE; ++n) { anchor += state[n]*c[n*WARP_SIZE+lane]; }
        for (int j = first+lane; j <= h+t; j += WARP_SIZE) {
            delta += v.feature<0>(seq, h, j)[col]*(scales[j-first]*cb[j-first]);
        }
        anchor = warp_reduce_sum(anchor);
        delta = warp_reduce_sum(delta);
        if (lane == 0) { v.y[(seq*v.nt+t)*inner+col] = fmaf(anchor, scales[v.span], delta); }
    }
}

static void ggml_cuda_op_ssm_scan_replay(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_ssm_scan_fused_cache * cache) {
    const ggml_tensor * s   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * dt  = dst->src[2];
    const ggml_tensor * A   = dst->src[3];
    const ggml_tensor * B   = dst->src[4];
    const ggml_tensor * C   = dst->src[5];
    const ggml_tensor * ids = dst->src[6];
    const ssm_replay_history history = {
        (const float *) dst->src[7]->data, (const float *) dst->src[8]->data, (const float *) dst->src[9]->data,
    };
    const int64_t n_seq = x->ne[3];
    const int64_t n_tok = x->ne[2];
    const int64_t inner = x->ne[0] * x->ne[1];
    const int64_t D = B->ne[0] * inner;
    const int64_t n_y = ggml_nelements(x);
    const ssm_replay_params p = {
        B->ne[0], x->ne[0], x->ne[1], B->ne[1], ggml_get_op_params_i32(dst, 1), s->ne[1],
        (int64_t) (x->nb[3] / sizeof(float)), (int64_t) (dt->nb[2] / sizeof(float)),
        (int64_t) (B->nb[3] / sizeof(float)), (int64_t) (C->nb[3] / sizeof(float)),
        { 0, inner, inner + x->ne[1], inner + x->ne[1] + B->ne[0] * B->ne[1] },
    };
    cudaStream_t stream = ctx.stream();
    if (ids->ne[1] == 1) {
        GGML_ASSERT(!cache);
        const int64_t capacity = dst->src[7]->ne[0]/inner, span = std::max(p.window, n_tok);
        if (n_tok > capacity - p.window) {
            ggml_cuda_ssm_scan(ctx, dst, nullptr, (const float *)s->data, nullptr);
            return;
        }
        const size_t n_cb = n_seq*n_tok*p.n_group*span;
        ggml_cuda_pool_alloc<float> coeff(ctx.pool(), n_cb + n_seq*n_tok*p.n_head*(span+1));
        float * y = (float *)dst->data;
        const ssm_verify v = { p, { (const float *)x->data, (const float *)dt->data, (const float *)B->data },
            history, (const float *)s->data, (const float *)A->data, (const float *)C->data, (const int32_t *)ids->data,
            y, y+n_y, coeff.get(), coeff.get()+n_cb, n_seq, n_tok, capacity, span,
            (int64_t)(x->nb[2]/sizeof(float)), (int64_t)(dt->nb[1]/sizeof(float)),
            (int64_t)(B->nb[2]/sizeof(float)), (int64_t)(C->nb[2]/sizeof(float)) };
        constexpr int warps = 4;
        ssm_verify_coefficients<<<dim3(n_tok, p.n_head, n_seq), warps*WARP_SIZE, 0, stream>>>(v);
        CUDA_CHECK(cudaGetLastError());
        const auto kernel = p.d_state == 64 ? ssm_verify_scan<64> : p.d_state == 96 ? ssm_verify_scan<96> :
                            p.d_state == 128 ? ssm_verify_scan<128> : ssm_verify_scan<256>;
        kernel<<<dim3((inner+warps-1)/warps, n_seq), warps*WARP_SIZE, 0, stream>>>(v);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    float * out = cache ? cache->data : (float *) dst->data + n_y;
    float * out_history = (float *) dst->data + n_y + D * n_seq;
    // Prefill needs the current state after applying all pending updates.
    const bool materialize = n_tok > 1;
    ggml_cuda_pool_alloc<int32_t> identity_ids(ctx.pool());
    ggml_cuda_pool_alloc<float> bc_buf(ctx.pool());
    ggml_cuda_pool_alloc<float> state_buf(ctx.pool());
    const int threads = 4 * WARP_SIZE;
    const int warps = threads / WARP_SIZE;
    float * staged = cache && n_seq > 1 ? (float *) dst->data + n_y : out;
    if (materialize) {
        identity_ids.alloc(n_seq);
        staged = state_buf.alloc(D * n_seq);
    } else {
        bc_buf.alloc(n_seq * p.n_group * p.window);
        ssm_replay_precompute<<<dim3(p.n_group, n_seq, (p.window + warps - 1) / warps), threads, 0, stream>>>(
            history.B, (const float *) B->data, (const float *) C->data,
            (const int32_t *) ids->data, bc_buf.get(), p);
        CUDA_CHECK(cudaGetLastError());
    }
    const int64_t n_tiles = (p.head_dim + warps - 1) / warps;
    GGML_ASSERT(p.n_head * n_tiles <= INT32_MAX); // CUDA grid.x and its divisors fit in 31 bits
    const dim3 blocks(p.n_head * n_tiles, n_seq);
    GGML_ASSERT(p.d_state == 64 || p.d_state == 96 || p.d_state == 128 || p.d_state == 256);
    const auto kernel = p.d_state == 64  ? ssm_replay_scan<64>  :
                        p.d_state == 96  ? ssm_replay_scan<96>  :
                        p.d_state == 128 ? ssm_replay_scan<128> : ssm_replay_scan<256>;
    kernel<<<blocks, threads, 0, stream>>>(
        (const float *) s->data, (const float *) x->data, (const float *) dt->data, (const float *) A->data,
        (const float *) B->data, (const float *) C->data, (const int32_t *) ids->data, bc_buf.get(),
        (float *) dst->data, staged, cache ? cache->data : nullptr, identity_ids.get(), history, p,
        init_fastdiv_values(n_tiles), init_fastdiv_values(p.n_head / p.n_group), materialize);
    CUDA_CHECK(cudaGetLastError());
    if (n_tok == 1) {
        // Size the grid for a one-token append; flushes and moved rows use the same grid-stride loops.
        const int64_t record = std::max(inner, p.n_group * p.d_state);
        const int blocks_x = (int) std::min<int64_t>(128, (record + 255) / 256);
        ssm_replay_store<<<dim3(blocks_x, n_seq, 3), 256, 0, stream>>>(
            (const float *) x->data, (const float *) dt->data, (const float *) B->data,
            (const int32_t *) ids->data, history, out_history, cache ? *cache : ggml_cuda_ssm_scan_fused_cache{}, p);
        CUDA_CHECK(cudaGetLastError());
        if (staged != out) {
            ssm_replay_commit<<<dim3(blocks_x, n_seq, cache->history[0] ? 4 : 1), 256, 0, stream>>>(
                staged, (const float *) s->data, (const int32_t *) ids->data, *cache, p);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    ggml_cuda_ssm_scan(ctx, dst, cache, state_buf.get(), identity_ids.get());
    CUDA_CHECK(cudaMemsetAsync(out_history, 0, (ggml_nelements(dst) - n_y - D*n_seq) * sizeof(float), stream));
}

static void ggml_cuda_op_ssm_scan_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                       const ggml_cuda_ssm_scan_fused_cache * cache) {
    if (ggml_get_op_params_i32(dst, 1) != 0) {
        ggml_cuda_op_ssm_scan_replay(ctx, dst, cache);
    } else {
        ggml_cuda_ssm_scan(ctx, dst, cache);
    }
}

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_ssm_scan_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_ssm_scan_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                       ggml_cuda_ssm_scan_fused_cache cache) {
    ggml_cuda_op_ssm_scan_impl(ctx, dst, &cache);
}
