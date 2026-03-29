#include "turboquant.cuh"
#include <cstdint>
#include <cmath>

// =============================================================================
// TurboQuant CUDA Implementation
//
// Two-stage KV cache compression:
//   Stage 1: PolarQuant - random rotation + polar coordinate quantization
//   Stage 2: QJL - sign-bit Johnson-Lindenstrauss residual correction
//
// Reference: "TurboQuant: Online Vector Quantization with Near-optimal
// Distortion Rate" (Google Research, ICLR 2026)
//
// Bitstream layout per vector (all wht_size = next-pow2(head_dim)):
//   [header: 4B] [angles: (wht_size-1)*num_bits] [sign: 1] [jl: QJL_PROJ_DIM]
// =============================================================================

// ---------- Pseudo-random rotation via hash (deterministic, no state) --------

// Fast hash for generating rotation matrix elements from seed + indices
__device__ __forceinline__ uint32_t tq_hash(uint64_t seed, uint32_t idx) {
    uint64_t h = seed ^ ((uint64_t)idx * 0x9E3779B97F4A7C15ULL);
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    return (uint32_t)(h ^ (h >> 31));
}

// Generate a pseudo-random normal value using Box-Muller from hash
__device__ __forceinline__ float tq_randn(uint64_t seed, uint32_t idx) {
    uint32_t h1 = tq_hash(seed, idx * 2);
    uint32_t h2 = tq_hash(seed, idx * 2 + 1);
    // Use __uint_as_float for proper bit reinterpretation: set exponent to 127
    // (biased) so the result is in [1.0, 2.0), then subtract 1 to get [0.0, 1.0)
    float u1 = __uint_as_float((h1 & 0x7FFFFFu) | 0x3F800000u) - 1.0f;
    u1 = fmaxf(u1, 1e-7f);  // avoid log(0)
    float u2 = __uint_as_float((h2 & 0x7FFFFFu) | 0x3F800000u) - 1.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

// ---------- Shared-memory parallel reduction helper --------------------------

// Parallel sum reduction using warp shuffles + cross-warp shared memory.
// Returns the total sum (broadcast to all threads via shared memory).
__device__ float tq_parallel_sum(float local_val) {
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_val += __shfl_down_sync(0xFFFFFFFF, local_val, offset);
    }

    __shared__ float warp_sums_reduce[8];  // up to 256 threads (8 warps)
    __shared__ float reduce_result;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) {
        warp_sums_reduce[warp_id] = local_val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) {
            total += warp_sums_reduce[w];
        }
        reduce_result = total;
    }
    __syncthreads();
    return reduce_result;
}

// ---------- Stage 1: PolarQuant Encode/Decode --------------------------------

// Encode kernel: apply random rotation, convert to polar, quantize angles
//
// Shared memory layout: [original: wht_size floats] [residual: wht_size floats]
// The two-array design lets us compute the last component's residual exactly
// by keeping the original WHT output intact while building dequantized values.
__launch_bounds__(TQ_BLOCK_SIZE, 1)
static __global__ void turboquant_encode_kernel(
    const half * __restrict__ src,
    void * __restrict__ dst,
    const int head_dim,
    const int num_kv_heads,
    const int batch_size,
    const int num_bits,
    const uint64_t rotation_seed
) {
    // Each block handles one vector (one head of one token)
    const int vec_idx = blockIdx.x;
    const int total_vecs = num_kv_heads * batch_size;
    if (vec_idx >= total_vecs) return;

    const int head_idx = vec_idx / batch_size;
    const int token_idx = vec_idx % batch_size;
    const int src_offset = token_idx * (head_dim * num_kv_heads) + head_idx * head_dim;

    // Compute padded size = next power of 2 >= head_dim
    int wht_size = 1;
    while (wht_size < head_dim) wht_size *= 2;

    // Two shared memory arrays: original WHT output + residuals
    extern __shared__ float shared[];
    float * original = shared;              // [wht_size]
    float * residual = shared + wht_size;   // [wht_size]

    // Randomized Hadamard Transform (RHT):
    //   1. Apply random sign-flips: x'[i] = x[i] * s[i], s[i] in {-1, +1}
    //   2. Zero-pad to next power of 2 so all elements participate in WHT
    //   3. Apply in-place Walsh-Hadamard butterfly (orthogonal, O(n log n))
    //   4. Normalize by 1/sqrt(wht_size)

    // Load + random sign flips, zero-pad tail
    for (int i = threadIdx.x; i < wht_size; i += blockDim.x) {
        if (i < head_dim) {
            uint32_t sign_bits = tq_hash(rotation_seed ^ (uint64_t)head_idx, (uint32_t)(i / 32));
            float sign = ((sign_bits >> (i & 31)) & 1) ? -1.0f : 1.0f;
            original[i] = __half2float(src[src_offset + i]) * sign;
        } else {
            original[i] = 0.0f;  // zero-pad for WHT orthogonality
        }
    }
    __syncthreads();

    // In-place Walsh-Hadamard butterfly over padded size
    for (int half_step = 1; half_step < wht_size; half_step *= 2) {
        for (int idx = threadIdx.x; idx < wht_size / 2; idx += blockDim.x) {
            int block_start = (idx / half_step) * (half_step * 2);
            int offset = idx % half_step;
            int i0 = block_start + offset;
            int i1 = i0 + half_step;
            float a = original[i0];
            float b = original[i1];
            original[i0] = a + b;
            original[i1] = a - b;
        }
        __syncthreads();
    }

    // Normalize ALL wht_size elements by 1/sqrt(wht_size) to make the
    // transform orthogonal.  Non-power-of-2 head_dim produces non-zero
    // tail coefficients [head_dim..wht_size) that must be preserved.
    float inv_sqrt_n = rsqrtf((float)wht_size);
    for (int i = threadIdx.x; i < wht_size; i += blockDim.x) {
        original[i] *= inv_sqrt_n;
    }
    __syncthreads();

    // Compute L2 norm over all wht_size coefficients
    // (orthogonal WHT preserves L2 norm, so this equals ||input||)
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < wht_size; i += blockDim.x) {
        local_sum_sq += original[i] * original[i];
    }
    float magnitude = sqrtf(tq_parallel_sum(local_sum_sq) + 1e-12f);

    // Quantize all wht_size-1 angular components
    int max_quant = (1 << num_bits) - 1;

    // Calculate output layout
    size_t angle_bits_per_vec = (size_t)(wht_size - 1) * num_bits;
    size_t sign_bit_offset = angle_bits_per_vec;  // 1 bit for last-component sign
    size_t jl_bits_per_vec = TQ_QJL_PROJ_DIM;
    size_t total_bits_per_vec = angle_bits_per_vec + 1 + jl_bits_per_vec;
    size_t packed_bytes = (total_bits_per_vec + 7) / 8;
    packed_bytes = (packed_bytes + 3) & ~(size_t)3;  // 4-byte align for atomicOr
    size_t bytes_per_vec = sizeof(turboquant_header) + packed_bytes;

    uint8_t * out_base = (uint8_t *)dst + vec_idx * bytes_per_vec;
    turboquant_header * hdr = (turboquant_header *)out_base;
    uint8_t * packed_data = out_base + sizeof(turboquant_header);

    // Write header
    if (threadIdx.x == 0) {
        hdr->magnitude = __float2half(magnitude);
        hdr->reserved = 0;
    }

    // Quantize all wht_size-1 angles, pack bits, and store dequantized values
    // in residual[] (temporarily — will be converted to actual residuals below).
    for (int i = threadIdx.x; i < wht_size - 1; i += blockDim.x) {
        float val = original[i];
        float normalized = (magnitude > 1e-12f) ? (val / magnitude) : 0.0f;
        normalized = fminf(fmaxf(normalized, -1.0f), 1.0f);
        int quantized = __float2int_rn((normalized + 1.0f) * 0.5f * (float)max_quant);
        quantized = min(max(quantized, 0), max_quant);

        // Store dequantized value for later residual computation
        float dequant = ((float)quantized / (float)max_quant) * 2.0f - 1.0f;
        dequant *= magnitude;
        residual[i] = dequant;

        // Pack quantized bits
        int bit_offset = i * num_bits;
        for (int b = 0; b < num_bits; b++) {
            int global_bit = bit_offset + b;
            int byte_idx = global_bit / 8;
            int bit_idx = global_bit % 8;
            if ((quantized >> b) & 1) {
                atomicOr((unsigned int *)(packed_data + (byte_idx & ~3)),
                         1u << (bit_idx + (byte_idx & 3) * 8));
            }
        }
    }
    __syncthreads();

    // Store sign bit of last component (index wht_size-1).
    // The decoder reconstructs magnitude via sqrt; this bit preserves the sign.
    if (threadIdx.x == 0) {
        if (original[wht_size - 1] < 0.0f) {
            int global_bit = (int)sign_bit_offset;
            int byte_idx = global_bit / 8;
            int bit_idx = global_bit % 8;
            atomicOr((unsigned int *)(packed_data + (byte_idx & ~3)),
                     1u << (bit_idx + (byte_idx & 3) * 8));
        }
    }

    // Reconstruct last component from unit-vector constraint + sign, then
    // compute residuals for ALL wht_size components.
    // sum_sq = Σ (dequant[i] / magnitude)² for i in [0, wht_size-2]
    float local_dq_sq = 0.0f;
    for (int i = threadIdx.x; i < wht_size - 1; i += blockDim.x) {
        float norm_dq = (magnitude > 1e-12f) ? (residual[i] / magnitude) : 0.0f;
        local_dq_sq += norm_dq * norm_dq;
    }
    float total_dq_sq = tq_parallel_sum(local_dq_sq);

    // Compute dequantized last component and convert all to residuals
    float last_sign = (original[wht_size - 1] < 0.0f) ? -1.0f : 1.0f;
    float last_dequant = last_sign * sqrtf(fmaxf(1.0f - total_dq_sq, 0.0f)) * magnitude;

    for (int i = threadIdx.x; i < wht_size - 1; i += blockDim.x) {
        residual[i] = original[i] - residual[i];  // angle residual
    }
    if (threadIdx.x == 0) {
        residual[wht_size - 1] = original[wht_size - 1] - last_dequant;
    }
    __syncthreads();

    // Stage 2: QJL residual encoding
    // Project all wht_size residuals through random Gaussian matrix and store signs.
    int jl_bit_base = (int)(sign_bit_offset + 1);  // after angles + sign bit
    for (int p = threadIdx.x; p < TQ_QJL_PROJ_DIM; p += blockDim.x) {
        float projection = 0.0f;
        for (int j = 0; j < wht_size; j++) {
            float r = tq_randn(rotation_seed + 0xDEAD, (uint32_t)(p * wht_size + j));
            projection += residual[j] * r;
        }

        // Store sign bit
        int sign_val = (projection >= 0.0f) ? 1 : 0;
        int global_bit = jl_bit_base + p;
        int byte_idx = global_bit / 8;
        int bit_idx = global_bit % 8;
        if (sign_val) {
            atomicOr((unsigned int *)(packed_data + (byte_idx & ~3)),
                     1u << (bit_idx + (byte_idx & 3) * 8));
        }
    }
}

// Decode kernel: dequantize angles, apply QJL correction, inverse rotation
__launch_bounds__(TQ_BLOCK_SIZE, 1)
static __global__ void turboquant_decode_kernel(
    const void * __restrict__ src,
    half * __restrict__ dst,
    const int head_dim,
    const int num_kv_heads,
    const int count,
    const int num_bits,
    const uint64_t rotation_seed
) {
    const int vec_idx = blockIdx.x;
    const int total_vecs = num_kv_heads * count;
    if (vec_idx >= total_vecs) return;

    const int head_idx = vec_idx / count;
    const int token_idx = vec_idx % count;
    const int dst_offset = token_idx * (head_dim * num_kv_heads) + head_idx * head_dim;

    int wht_size = 1;
    while (wht_size < head_dim) wht_size *= 2;

    int max_quant = (1 << num_bits) - 1;

    // Layout must match encode kernel
    size_t angle_bits_per_vec = (size_t)(wht_size - 1) * num_bits;
    size_t sign_bit_offset = angle_bits_per_vec;
    size_t jl_bits_per_vec = TQ_QJL_PROJ_DIM;
    size_t total_bits_per_vec = angle_bits_per_vec + 1 + jl_bits_per_vec;
    size_t packed_bytes = (total_bits_per_vec + 7) / 8;
    packed_bytes = (packed_bytes + 3) & ~(size_t)3;
    size_t bytes_per_vec = sizeof(turboquant_header) + packed_bytes;

    const uint8_t * in_base = (const uint8_t *)src + vec_idx * bytes_per_vec;
    const turboquant_header * hdr = (const turboquant_header *)in_base;
    const uint8_t * packed_data = in_base + sizeof(turboquant_header);

    float magnitude = __half2float(hdr->magnitude);

    extern __shared__ float shared[];
    float * decoded = shared;  // [wht_size]

    // Dequantize all wht_size-1 angular coordinates
    for (int i = threadIdx.x; i < wht_size - 1; i += blockDim.x) {
        int bit_offset = i * num_bits;
        int quant_val = 0;
        for (int b = 0; b < num_bits; b++) {
            int global_bit = bit_offset + b;
            int byte_idx = global_bit / 8;
            int bit_idx = global_bit % 8;
            if ((packed_data[byte_idx] >> bit_idx) & 1) {
                quant_val |= (1 << b);
            }
        }

        float normalized = ((float)quant_val / (float)max_quant) * 2.0f - 1.0f;
        decoded[i] = normalized * magnitude;
    }
    __syncthreads();

    // Reconstruct last component from unit-vector constraint + stored sign bit
    float local_dq_sq = 0.0f;
    for (int i = threadIdx.x; i < wht_size - 1; i += blockDim.x) {
        float norm_i = (magnitude > 1e-12f) ? decoded[i] / magnitude : 0.0f;
        local_dq_sq += norm_i * norm_i;
    }
    float total_dq_sq = tq_parallel_sum(local_dq_sq);

    if (threadIdx.x == 0) {
        // Read sign bit
        int global_bit = (int)sign_bit_offset;
        int byte_idx = global_bit / 8;
        int bit_idx = global_bit % 8;
        float last_sign = ((packed_data[byte_idx] >> bit_idx) & 1) ? -1.0f : 1.0f;
        decoded[wht_size - 1] = last_sign * sqrtf(fmaxf(1.0f - total_dq_sq, 0.0f)) * magnitude;
    }
    __syncthreads();

    // Apply QJL correction (approximate residual recovery)
    int jl_bit_base = (int)(sign_bit_offset + 1);
    for (int i = threadIdx.x; i < wht_size; i += blockDim.x) {
        float correction = 0.0f;
        for (int p = 0; p < TQ_QJL_PROJ_DIM; p++) {
            int global_bit = jl_bit_base + p;
            int byte_idx = global_bit / 8;
            int bit_idx = global_bit % 8;
            float sign = ((packed_data[byte_idx] >> bit_idx) & 1) ? 1.0f : -1.0f;

            float r = tq_randn(rotation_seed + 0xDEAD, (uint32_t)(p * wht_size + i));
            correction += sign * r;
        }
        // Scale correction: expected quantization error ~ magnitude * 2^(-num_bits)
        // divided by sqrt(proj_dim) for JL concentration.
        float qjl_scale = magnitude / (float)(1 << num_bits) / sqrtf((float)TQ_QJL_PROJ_DIM);
        correction *= qjl_scale;
        decoded[i] += correction;
    }
    __syncthreads();

    // Inverse Randomized Hadamard Transform:
    //   Same padded wht_size as encode. Normalize, butterfly, undo sign-flips.
    //   Only output the first head_dim elements.
    float inv_sqrt_d = rsqrtf((float)wht_size);
    for (int i = threadIdx.x; i < wht_size; i += blockDim.x) {
        decoded[i] *= inv_sqrt_d;
    }
    __syncthreads();

    for (int half_step = 1; half_step < wht_size; half_step *= 2) {
        for (int idx = threadIdx.x; idx < wht_size / 2; idx += blockDim.x) {
            int block_start = (idx / half_step) * (half_step * 2);
            int offset = idx % half_step;
            int i0 = block_start + offset;
            int i1 = i0 + half_step;
            float a = decoded[i0];
            float b = decoded[i1];
            decoded[i0] = a + b;
            decoded[i1] = a - b;
        }
        __syncthreads();
    }

    // Undo random sign-flips and write first head_dim elements to output
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        uint32_t sign_bits = tq_hash(rotation_seed ^ (uint64_t)head_idx, (uint32_t)(i / 32));
        float sign = ((sign_bits >> (i & 31)) & 1) ? -1.0f : 1.0f;
        dst[dst_offset + i] = __float2half(decoded[i] * sign);
    }
}

// ---------- Host-side launcher functions -------------------------------------

void turboquant_encode_cuda(
    const half * src,
    void * dst,
    int head_dim,
    int num_kv_heads,
    int batch_size,
    int num_bits,
    uint64_t rotation_seed,
    cudaStream_t stream
) {
    const int total_vecs = num_kv_heads * batch_size;
    if (total_vecs == 0) return;

    // Zero the output buffer first
    size_t buf_size = turboquant_buffer_size(head_dim, num_kv_heads, batch_size, num_bits);
    cudaMemsetAsync(dst, 0, buf_size, stream);

    int wht_size = tq_next_pow2(head_dim);

    int block_size = min(TQ_BLOCK_SIZE, wht_size);
    // Round up to next power of 2 for warp efficiency
    if (block_size < 32) block_size = 32;
    else if (block_size < 64) block_size = 64;
    else if (block_size < 128) block_size = 128;
    else block_size = 256;

    // Encode needs two arrays: original[wht_size] + residual[wht_size]
    size_t shared_mem = 2 * wht_size * sizeof(float);

    turboquant_encode_kernel<<<total_vecs, block_size, shared_mem, stream>>>(
        src, dst, head_dim, num_kv_heads, batch_size, num_bits, rotation_seed
    );
}

void turboquant_decode_cuda(
    const void * src,
    half * dst,
    int head_dim,
    int num_kv_heads,
    int count,
    int num_bits,
    uint64_t rotation_seed,
    cudaStream_t stream
) {
    const int total_vecs = num_kv_heads * count;
    if (total_vecs == 0) return;

    int wht_size = tq_next_pow2(head_dim);

    int block_size = min(TQ_BLOCK_SIZE, wht_size);
    if (block_size < 32) block_size = 32;
    else if (block_size < 64) block_size = 64;
    else if (block_size < 128) block_size = 128;
    else block_size = 256;

    // Decode needs one array: decoded[wht_size]
    size_t shared_mem = wht_size * sizeof(float);

    turboquant_decode_kernel<<<total_vecs, block_size, shared_mem, stream>>>(
        src, dst, head_dim, num_kv_heads, count, num_bits, rotation_seed
    );
}
