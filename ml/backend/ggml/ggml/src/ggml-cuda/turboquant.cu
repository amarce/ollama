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

// ---------- Stage 1: PolarQuant Encode/Decode --------------------------------

// Encode kernel: apply random rotation, convert to polar, quantize angles
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

    // Step 1: Load vector into shared memory and apply random rotation
    extern __shared__ float shared[];
    float * rotated = shared;  // [head_dim]

    // Each thread handles multiple elements
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = 0.0f;
        // Apply random rotation: y_i = sum_j R_ij * x_j
        // R is constructed from hash-based random normals, then orthogonalized
        // via Gram-Schmidt-like projection. For efficiency we use a fast
        // randomized Hadamard-like rotation instead of full random matrix.
        //
        // Simplified: multiply by random sign flips + fixed permutation
        // This gives a random rotation that concentrates angular distribution
        uint32_t sign_bits = tq_hash(rotation_seed ^ (uint64_t)head_idx, (uint32_t)i);

        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            float xj = __half2float(src[src_offset + j]);
            // Random sign flip for each dimension pair
            float sign = ((sign_bits >> (j & 31)) & 1) ? -1.0f : 1.0f;
            if (i == j) {
                val += xj * sign;
            }
        }

        // For diagonal-only fast rotation, just apply sign flips
        float xval = __half2float(src[src_offset + i]);
        float sign = ((sign_bits >> (i & 31)) & 1) ? -1.0f : 1.0f;
        rotated[i] = xval * sign;
    }
    __syncthreads();

    // Step 2: Compute L2 norm (magnitude in polar coordinates)
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        local_sum_sq += rotated[i] * rotated[i];
    }

    // Warp reduction for sum of squares
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
    }

    // Cross-warp reduction: each warp leader writes its partial sum, then
    // thread 0 accumulates all warp contributions to compute the full norm.
    __shared__ float warp_sums[8]; // supports up to 256 threads (8 warps)
    __shared__ float shared_norm;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) {
            total += warp_sums[w];
        }
        shared_norm = sqrtf(total + 1e-12f);
    }
    __syncthreads();
    float magnitude = shared_norm;

    // Step 3: Quantize angular coordinates
    // After rotation, coordinates follow a concentrated distribution
    // Normalize to unit vector, then quantize each angular component to num_bits
    int max_quant = (1 << num_bits) - 1;

    // Calculate output layout
    size_t angle_bits_per_vec = (size_t)(head_dim - 1) * num_bits;
    size_t jl_bits_per_vec = TQ_QJL_PROJ_DIM;
    size_t total_bits_per_vec = angle_bits_per_vec + jl_bits_per_vec;
    size_t bytes_per_vec = sizeof(turboquant_header) + (total_bits_per_vec + 7) / 8;

    uint8_t * out_base = (uint8_t *)dst + vec_idx * bytes_per_vec;
    turboquant_header * hdr = (turboquant_header *)out_base;
    uint8_t * packed_data = out_base + sizeof(turboquant_header);

    // Write header
    if (threadIdx.x == 0) {
        hdr->magnitude = __float2half(magnitude);
        hdr->reserved = 0;
    }

    // Quantize angles: map normalized component from [-1,1] to [0, max_quant]
    for (int i = threadIdx.x; i < head_dim - 1; i += blockDim.x) {
        float normalized = (magnitude > 1e-12f) ? (rotated[i] / magnitude) : 0.0f;
        // Clamp to [-1, 1] and map to [0, max_quant]
        normalized = fminf(fmaxf(normalized, -1.0f), 1.0f);
        int quantized = __float2int_rn((normalized + 1.0f) * 0.5f * (float)max_quant);
        quantized = min(max(quantized, 0), max_quant);

        // Pack bits
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

    // Stage 2: QJL residual encoding
    // Project quantization residual through random Gaussian matrix and store signs
    //
    // IMPORTANT: __syncthreads() + __threadfence_block() here ensures all
    // atomicOr writes to packed_data from Stage 1 are visible before any
    // thread reads packed_data bytes for residual reconstruction.
    __syncthreads();
    __threadfence_block();

    int jl_bit_base = (int)angle_bits_per_vec;
    for (int p = threadIdx.x; p < TQ_QJL_PROJ_DIM; p += blockDim.x) {
        float projection = 0.0f;
        for (int j = 0; j < head_dim - 1; j++) {
            float orig = rotated[j];
            // Reconstruct quantized value for residual
            int bit_offset = j * num_bits;
            int quant_val = 0;
            for (int b = 0; b < num_bits; b++) {
                int global_bit = bit_offset + b;
                int byte_idx = global_bit / 8;
                int bit_idx = global_bit % 8;
                // Use volatile read to ensure we see the atomicOr writes
                if ((((volatile uint8_t *)packed_data)[byte_idx] >> bit_idx) & 1) {
                    quant_val |= (1 << b);
                }
            }
            float dequant = ((float)quant_val / (float)max_quant) * 2.0f - 1.0f;
            dequant *= magnitude;
            float residual = orig - dequant;
            // Random projection
            float r = tq_randn(rotation_seed + 0xDEAD, (uint32_t)(p * head_dim + j));
            projection += residual * r;
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

    int max_quant = (1 << num_bits) - 1;

    size_t angle_bits_per_vec = (size_t)(head_dim - 1) * num_bits;
    size_t jl_bits_per_vec = TQ_QJL_PROJ_DIM;
    size_t total_bits_per_vec = angle_bits_per_vec + jl_bits_per_vec;
    size_t bytes_per_vec = sizeof(turboquant_header) + (total_bits_per_vec + 7) / 8;

    const uint8_t * in_base = (const uint8_t *)src + vec_idx * bytes_per_vec;
    const turboquant_header * hdr = (const turboquant_header *)in_base;
    const uint8_t * packed_data = in_base + sizeof(turboquant_header);

    float magnitude = __half2float(hdr->magnitude);

    extern __shared__ float shared[];
    float * decoded = shared;  // [head_dim]

    // Dequantize angular coordinates
    for (int i = threadIdx.x; i < head_dim - 1; i += blockDim.x) {
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

    // Reconstruct last dimension from unit-vector constraint
    // Parallelized: each thread sums its slice, then warp+cross-warp reduce
    __syncthreads();  // ensure all decoded[] writes are visible
    {
        float partial_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim - 1; i += blockDim.x) {
            float norm_i = (magnitude > 1e-12f) ? decoded[i] / magnitude : 0.0f;
            partial_sq += norm_i * norm_i;
        }

        // Warp reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            partial_sq += __shfl_down_sync(0xFFFFFFFF, partial_sq, offset);
        }

        // Cross-warp reduction (reuse warp_sums from shared memory)
        __shared__ float warp_sums_decode[8];
        __shared__ float shared_last_norm;
        int warp_id = threadIdx.x / warpSize;
        int lane_id = threadIdx.x % warpSize;
        if (lane_id == 0) {
            warp_sums_decode[warp_id] = partial_sq;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float total_sq = 0.0f;
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            for (int w = 0; w < num_warps; w++) {
                total_sq += warp_sums_decode[w];
            }
            shared_last_norm = sqrtf(fmaxf(1.0f - total_sq, 0.0f));
            decoded[head_dim - 1] = shared_last_norm * magnitude;
        }
    }
    __syncthreads();

    // Apply QJL correction (approximate residual recovery)
    // The QJL sign bits allow an unbiased estimate of the residual direction
    int jl_bit_base = (int)angle_bits_per_vec;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float correction = 0.0f;
        for (int p = 0; p < TQ_QJL_PROJ_DIM; p++) {
            int global_bit = jl_bit_base + p;
            int byte_idx = global_bit / 8;
            int bit_idx = global_bit % 8;
            float sign = ((packed_data[byte_idx] >> bit_idx) & 1) ? 1.0f : -1.0f;

            float r = tq_randn(rotation_seed + 0xDEAD, (uint32_t)(p * head_dim + i));
            correction += sign * r;
        }
        // Scale correction by expected residual magnitude / sqrt(proj_dim)
        correction *= magnitude * 0.05f / sqrtf((float)TQ_QJL_PROJ_DIM);
        decoded[i] += correction;
    }
    __syncthreads();

    // Inverse rotation (sign-flip is self-inverse)
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        uint32_t sign_bits = tq_hash(rotation_seed ^ (uint64_t)head_idx, (uint32_t)i);
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

    int block_size = min(TQ_BLOCK_SIZE, head_dim);
    // Round up to next power of 2 for warp efficiency
    if (block_size < 32) block_size = 32;
    else if (block_size < 64) block_size = 64;
    else if (block_size < 128) block_size = 128;
    else block_size = 256;

    size_t shared_mem = head_dim * sizeof(float) + sizeof(float);

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

    int block_size = min(TQ_BLOCK_SIZE, head_dim);
    if (block_size < 32) block_size = 32;
    else if (block_size < 64) block_size = 64;
    else if (block_size < 128) block_size = 128;
    else block_size = 256;

    size_t shared_mem = head_dim * sizeof(float);

    turboquant_decode_kernel<<<total_vecs, block_size, shared_mem, stream>>>(
        src, dst, head_dim, num_kv_heads, count, num_bits, rotation_seed
    );
}
