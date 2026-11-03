#include "moe-weighted-reduction.cuh"

#include <cstring>

static const bool baiinf_parity_reduction = []() {
    const char * value = getenv("BAIINF_PARITY_REDUCTION");
#if defined(__linux__) && defined(GGML_USE_HIP)
    if (!value) { return true; }
#endif
    if (!value || strcmp(value, "0") == 0) { return false; }
    if (strcmp(value, "1") == 0) { return true; }
    GGML_ABORT("BAIINF_PARITY_REDUCTION must be 0 or 1, got '%s'", value);
}();

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  const bool    columns_first) {
    const int64_t token = columns_first ? blockIdx.y : blockIdx.x;
    const int64_t col   = (int64_t) (columns_first ? blockIdx.x : blockIdx.y) * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          bool          columns_first,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const int64_t n_col_blocks = (n_embd + threads - 1) / threads;
    const dim3 blocks(columns_first ? n_col_blocks : n_tokens, columns_first ? n_tokens : n_col_blocks, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used, columns_first);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    bool columns_first = false;
#if defined(GGML_USE_HIP)
    columns_first = baiinf_parity_reduction &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 &&
        experts->ne[0] == 2048 && experts->ne[1] == 8 && experts->ne[2] >= 33 &&
        experts->ne[2] <= 1024 && experts->ne[3] == 1 &&
        weights->ne[0] == 1 && weights->ne[1] == 8 && weights->ne[2] == experts->ne[2] && weights->ne[3] == 1 &&
        (!expert_scale || ggml_are_same_shape(expert_scale, weights)) &&
        dst->ne[0] == 2048 && dst->ne[1] == experts->ne[2] && dst->ne[2] == 1 && dst->ne[3] == 1;
#else
    GGML_UNUSED(baiinf_parity_reduction);
#endif

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, columns_first, stream);
    CUDA_CHECK(cudaGetLastError());
}
