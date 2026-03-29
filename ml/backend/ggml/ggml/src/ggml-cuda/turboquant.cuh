#pragma once

#include "common.cuh"
#include <cstdint>

// TurboQuant: Two-stage KV cache compression (PolarQuant + QJL)
// Based on Google Research paper "TurboQuant: Online Vector Quantization
// with Near-optimal Distortion Rate" (ICLR 2026, arXiv 2504.19874)
//
// Stage 1 - PolarQuant: Random rotation + polar coordinate quantization
// Stage 2 - QJL: Johnson-Lindenstrauss sign-bit residual encoding
//
// Achieves ~5.3x compression at 3-bit with 99.5% attention fidelity.

#define TQ_BLOCK_SIZE 256
#define TQ_BITS_DEFAULT 3
#define TQ_QJL_PROJ_DIM 32  // JL projection dimension per head dim

// Compressed KV cache entry header (per-vector metadata)
struct turboquant_header {
    half magnitude;       // L2 norm (polar radius)
    uint16_t reserved;    // alignment padding
};

// Encode fp16 KV vectors into TurboQuant compressed format
// src: [head_dim, num_kv_heads, batch_size] in fp16
// dst: compressed output buffer
// rotation_seed: per-layer seed for random rotation matrix
void turboquant_encode_cuda(
    const half * src,
    void * dst,
    int head_dim,
    int num_kv_heads,
    int batch_size,
    int num_bits,
    uint64_t rotation_seed,
    cudaStream_t stream);

// Decode TurboQuant compressed format back to fp16
// src: compressed input buffer
// dst: [head_dim, num_kv_heads, count] in fp16
void turboquant_decode_cuda(
    const void * src,
    half * dst,
    int head_dim,
    int num_kv_heads,
    int count,
    int num_bits,
    uint64_t rotation_seed,
    cudaStream_t stream);

// Compute next power of 2 >= n (used for WHT padding)
static inline int tq_next_pow2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

// Returns the number of bytes per element for TurboQuant at given bit width
// This accounts for PolarQuant angles + sign bit + QJL sign bits + metadata
static inline float turboquant_bytes_per_element(int num_bits) {
    // Approximation for head_dim=128 (power-of-2, wht_size == head_dim):
    //   Angles: (128-1) * num_bits bits
    //   Last-component sign: 1 bit
    //   QJL signs: TQ_QJL_PROJ_DIM bits
    //   Metadata: 4 bytes (magnitude + padding) per vector
    //
    // Per element (averaged over head_dim=128):
    //   3-bit: (127*3 + 1 + 32 + 32) / (128*8) = 447/1024 ≈ 0.437 bytes/element
    //   4-bit: (127*4 + 1 + 32 + 32) / (128*8) = 573/1024 ≈ 0.560 bytes/element
    //
    // Non-power-of-2 head dims (e.g. 80) pad to wht_size (128) and store
    // wht_size-1 angles, using proportionally more bits per input element.
    float angle_bits = (float)num_bits;  // per angular coordinate
    float sign_bit_per_elem = 1.0f / 128.0f;
    float jl_bits_per_elem = (float)TQ_QJL_PROJ_DIM / 128.0f;
    float meta_bits_per_elem = 32.0f / 128.0f;  // 4-byte header spread over head_dim
    float total_bits = angle_bits + sign_bit_per_elem + jl_bits_per_elem + meta_bits_per_elem;
    return total_bits / 8.0f;
}

// Returns the compressed buffer size for a given KV cache shape
static inline size_t turboquant_buffer_size(int head_dim, int num_kv_heads, int num_tokens, int num_bits) {
    // WHT pads to next power of 2; we quantize all wht_size-1 angles + 1 sign bit
    int wht_size = tq_next_pow2(head_dim);
    size_t angle_bits_per_vec = (size_t)(wht_size - 1) * num_bits;
    size_t sign_bits = 1;  // sign of last WHT component
    size_t jl_bits_per_vec = TQ_QJL_PROJ_DIM;
    size_t total_bits_per_vec = angle_bits_per_vec + sign_bits + jl_bits_per_vec;
    // Round packed payload up to 4 bytes so 32-bit atomicOr in encode kernel
    // never writes past the vector boundary into the next vector's header.
    size_t packed_bytes = (total_bits_per_vec + 7) / 8;
    packed_bytes = (packed_bytes + 3) & ~(size_t)3;  // align to 4 bytes
    size_t bytes_per_vec = sizeof(turboquant_header) + packed_bytes;

    return bytes_per_vec * num_kv_heads * num_tokens;
}
