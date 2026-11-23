#include "common.cuh"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always the state row size, set in-kernel)
struct ggml_cuda_ssm_scan_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots
};

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                           const ggml_cuda_ssm_scan_fused_cache * cache = nullptr);
