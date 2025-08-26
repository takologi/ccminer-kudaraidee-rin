#include "sha3-256.cuh"


// 64-bit rotate left using funnel shifts when beneficial
static __device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
/*
    // Implement via two 32-bit funnel shifts to map to SHF.R.WRAP.B32
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    n &= 63u;
    if (!n) return x;
    if (n < 32u) {
        uint32_t n32 = (uint32_t)n;
        uint32_t new_lo = __funnelshift_l(lo, hi, n32);
        uint32_t new_hi = __funnelshift_l(hi, lo, n32);
        return ((uint64_t)new_hi << 32) | new_lo;
    } else {
        uint32_t n2 = (uint32_t)(n - 32u);
        uint32_t new_lo = __funnelshift_l(hi, lo, n2);
        uint32_t new_hi = __funnelshift_l(lo, hi, n2);
        return ((uint64_t)new_hi << 32) | new_lo;
    }
*/
   return (x << n) | (x >> (64 - n));
}

__constant__ __align__(8) int kRho[24] = {
     1,  3,  6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44
};

__constant__ __align__(4) int kPi[24] = {
    10,  7, 11, 17, 18, 3,
     5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2,
    20, 14, 22,  9, 6,  1
};

__constant__ __align__(8) uint64_t kRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

// Keccak-f[1600]
static __device__ __forceinline__ void keccakf(uint64_t st[25]) {
    uint64_t bc[5];
    uint64_t t;
#pragma unroll
    for (int round = 0; round < 24; round++) {
        // Theta
#pragma unroll
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // Rho and Pi
        t = st[1];
#pragma unroll
        for (int i = 0; i < 24; i++) {
            int j = kPi[i];
            bc[0] = st[j];
            st[j] = rotl64(t, kRho[i]);
            t = bc[0];
        }

        // Chi
        uint64_t b0;
        uint64_t b1;
        uint64_t b2;
        uint64_t b3;
        uint64_t b4;
#pragma unroll
        for (int j = 0; j < 25; j += 5) {
/*
#pragma unroll
            for (int i = 0; i < 5; i++)
                bc[i] = st[j + i];
#pragma unroll
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];        }
*/
            b0 = st[j + 0];
            b1 = st[j + 1];
            b2 = st[j + 2];
            b3 = st[j + 3];
            b4 = st[j + 4];
            st[j + 0] ^= (~b1) & b2;
            st[j + 1] ^= (~b2) & b3;
            st[j + 2] ^= (~b3) & b4;
            st[j + 3] ^= (~b4) & b0;
            st[j + 4] ^= (~b0) & b1;
        }
        // Iota
        st[0] ^= kRC[round];
    }
}

// little-endian load/store for 64-bit
static __device__ __forceinline__ uint64_t load64_le(const uint8_t *src) {
    // Assume unaligned ok on CUDA; keep simple and let compiler optimize
    uint64_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        x |= ((uint64_t)src[i]) << (8 * i);
    }
    return x;
}

static __device__ __forceinline__ void store64_le(uint8_t *dst, uint64_t x) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)(x >> (8 * i));
    }
}

// Phase 1 optimized SHA3-256 device function (intrinsics, constants, inlining)
__device__ void sha3_256_device(const uint8_t * __restrict__ input, size_t inlen, uint8_t * __restrict__ hash_out) {
    const size_t rate = 136; // bytes
    uint64_t st[25];
#pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0ull;

    // Absorb full blocks
    while (inlen >= rate) {
#pragma unroll
        for (int i = 0; i < (int)(rate / 8); i++) {
            st[i] ^= load64_le(input + i * 8);
        }
        keccakf(st);
        input += rate;
        inlen -= rate;
    }

    // Absorb remaining bytes (common path: 32 bytes)
#pragma unroll
    for (int i = 0; i < 4; i++) {
        st[i] ^= load64_le(input + i * 8);
    }
    ((uint8_t*)st)[32] ^= 0x06;      // domain separation and padding
    ((uint8_t*)st)[rate - 1] ^= 0x80;
    keccakf(st);

    // Squeeze 32 bytes
#pragma unroll
    for (int i = 0; i < 4; i++) {
        store64_le(hash_out + i * 8, st[i]);
    }
}
