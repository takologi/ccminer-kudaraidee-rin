#ifndef RINHASH_SHA3_256_CUH
#define RINHASH_SHA3_256_CUH

#include <stdint.h>
#include <stddef.h>

#define KECCAKF_ROUNDS 24

__device__ void sha3_256_device(const uint8_t * __restrict__ input, size_t inlen, uint8_t * __restrict__ hash_out);

#endif // RINHASH_SHA3_256_CUH