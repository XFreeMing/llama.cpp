#include "common.cuh"
#include "mmid.cuh"

#include <cstring>

static const bool baiinf_parity_ids = []() {
    const char * value = getenv("BAIINF_PARITY_IDS");
#if defined(__linux__) && defined(GGML_USE_HIP)
    if (!value) { return true; }
#endif
    if (!value || strcmp(value, "0") == 0) { return false; }
    if (strcmp(value, "1") == 0) { return true; }
    GGML_ABORT("BAIINF_PARITY_IDS must be 0 or 1, got '%s'", value);
}();

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

static __device__ void write_tile_spec(ggml_cuda_mm_id_tile_spec * tile_spec, int expert,
        int nex_prev, int it_compact, int slots, uint3 tile_width) {
    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), it_compact);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + it_compact);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(slots + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

#if defined(GGML_USE_HIP)
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper_4warp(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        ggml_cuda_mm_id_tile_spec * spec, uint3 width, const int n_tokens, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_expert_used = 8;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int expert = blockIdx.x;
    const int n_tokens_warp = (n_tokens + 3) / 4;
    const int token_begin = warp*n_tokens_warp;
    const int token_end = min(token_begin + n_tokens_warp, n_tokens);

    extern __shared__ char data_mm_ids_helper_4warp[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper_4warp;
    __shared__ int counts[4];
    __shared__ int lower[4];

    int nex_prev = 0;
    int it_compact = 0;
    for (int it0 = token_begin; it0 < token_end; it0 += warp_size/n_expert_used) {
        const int it = it0 + lane/n_expert_used;
        const int iex = lane % n_expert_used;
        const int expert_used = it < token_end ? ids[it*si1 + iex] : INT_MAX;
        const int iex_used = expert_used == expert ? iex : -1;
        nex_prev += expert_used < expert;

        const int add_self = warp_reduce_any<n_expert_used>(iex_used != -1);
        int add_lower = 0;
#pragma unroll
        for (int offset = n_expert_used; offset < warp_size; offset += n_expert_used) {
            const int tmp = __shfl_up_sync(0xFFFFFFFFULL, add_self, offset, warp_size);
            if (lane >= offset) {
                add_lower += tmp;
            }
        }
        if (iex_used != -1) {
            store[token_begin + it_compact + add_lower] = mm_ids_helper_store(it, iex_used);
        }
        it_compact += __shfl_sync(0xFFFFFFFFULL, add_lower + add_self, warp_size - 1, warp_size);
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    if (lane == 0) {
        counts[warp] = it_compact;
        lower[warp] = nex_prev;
    }
    __syncthreads();

    int compact_begin = 0;
    int compact_total = 0;
    nex_prev = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        nex_prev += lower[i];
        compact_total += counts[i];
        if (i < warp) {
            compact_begin += counts[i];
        }
    }
    for (int itc = lane; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[token_begin + itc];
        const int it = store_it.it();
        const int iex_used = store_it.iex_used();
        const int row = nex_prev + compact_begin + itc;
        ids_dst[row] = it*n_expert_used + iex_used;
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = row;
        } else {
            ids_src1[row] = it*sis1 + iex_used % nchannels_y;
        }
    }
    if (threadIdx.x == 0) {
        write_tile_spec(spec, expert, nex_prev, compact_total, n_tokens*8, width);
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + compact_total;
        }
    }
}
#endif

#if defined(GGML_USE_HIP) && defined(__linux__)
namespace {
// A tile contains 32 tokens (256 slots). One bit per lane records a slot's
// position, so simultaneous inserts preserve the original slot order.
__global__ void top8_tile_histogram(const int * ids, int * ranks, int * hist, int tokens, int si1) {
    __shared__ unsigned masks[8][256];
    const int t = threadIdx.x;
    const int wave = t / 32;
    const int lane = t % 32;
    const int slot = blockIdx.x*256 + t;
#pragma unroll
    for (int w = 0; w < 8; ++w) masks[w][t] = 0;
    __syncthreads();
    const int expert = slot < tokens*8 ? ids[(slot/8)*si1 + slot%8] : 0;
    if (slot < tokens*8) atomicOr(&masks[wave][expert], 1u << lane);
    __syncthreads();
    unsigned rank = __popc(masks[wave][expert] & ((1u << lane) - 1));
    unsigned count = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        if (w < wave) rank += __popc(masks[w][expert]);
        count += __popc(masks[w][t]);
    }
    if (slot < tokens*8) ranks[slot] = rank;
    hist[blockIdx.x*256 + t] = count;
}

// A single block scans the 256 expert totals and converts tile counts to offsets.
__global__ void top8_offsets(int * hist, int * bounds, int tiles, int slots, ggml_cuda_mm_id_tile_spec * spec, uint3 width) {
    __shared__ int totals[256];
    const int e = threadIdx.x;
    int total = 0;
    for (int tile = 0; tile < tiles; ++tile) total += hist[tile*256 + e];
    totals[e] = total;
    __syncthreads();
    for (int step = 1; step < 256; step *= 2) {
        const int add = e >= step ? totals[e-step] : 0;
        __syncthreads();
        totals[e] += add;
        __syncthreads();
    }
    int base = totals[e] - total;
    write_tile_spec(spec, e, base, total, slots, width);
    bounds[e] = base;
    if (e == 255) bounds[256] = slots;
    for (int tile = 0; tile < tiles; ++tile) {
        const int count = hist[tile*256 + e];
        hist[tile*256 + e] = base;
        base += count;
    }
}

__global__ void top8_scatter(const int * ids, const int * ranks, const int * hist, int * src, int * dst,
                             int tokens, int channels, int si1, int sis1, bool inverse) {
    const int slot = blockIdx.x*256 + threadIdx.x;
    if (slot >= tokens*8) return;
    const int token = slot/8, selected = slot%8;
    const int row = hist[blockIdx.x*256 + ids[token*si1 + selected]] + ranks[slot];
    dst[row] = slot;
    if (inverse) src[slot] = row;
    else src[row] = token*sis1 + selected%channels;
}
}
#endif

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = ggml_cuda_shfl_up(it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += ggml_cuda_shfl(it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    write_tile_spec(tile_spec, expert, nex_prev, it_compact, n_tokens*n_expert_used, tile_width);
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
    }
}

void ggml_cuda_group_mmq_ids(ggml_backend_cuda_context & ctx, const int * ids, int * src, int * dst,
                             ggml_cuda_mm_id_tile_spec * spec, int * bounds, int width, int experts,
                             int tokens, int top, int channels, int si1, int sis1, bool inverse, cudaStream_t stream) {
#if defined(GGML_USE_HIP) && defined(__linux__)
    if (bounds && tokens <= 1024) {
        const int wave = ggml_cuda_info().devices[ctx.device].warp_size;
        mm_ids_helper_4warp<<<experts, 4*wave, tokens*sizeof(mm_ids_helper_store), stream>>>(
            ids, src, dst, bounds, spec, init_fastdiv_values(width), tokens, channels, si1, sis1, inverse);
        return;
    }
    if (bounds) {
        const int tiles = (tokens + 31) / 32;
        ggml_cuda_pool_alloc<int> scratch(ctx.pool(), tokens * 8 + tiles * 256);
        int * ranks = scratch.get();
        int * hist = ranks + tokens * 8;
        top8_tile_histogram<<<tiles, 256, 0, stream>>>(ids, ranks, hist, tokens, si1);
        top8_offsets<<<1, 256, 0, stream>>>(hist, bounds, tiles, tokens * 8, spec, init_fastdiv_values(width));
        top8_scatter<<<tiles, 256, 0, stream>>>(ids, ranks, hist, src, dst, tokens, channels, si1, sis1, inverse);
        return;
    }
#else
    GGML_UNUSED_VARS(ctx, bounds);
#endif
    ggml_cuda_launch_mm_ids_helper(ids, src, dst, spec, experts, tokens, top, channels, width, si1, sis1, inverse, stream);
}

bool ggml_cuda_parity_ids_enabled() { return baiinf_parity_ids; }
