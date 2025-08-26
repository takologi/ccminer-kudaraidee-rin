#ifndef ARGON2D_DEVICE_CUH
#define ARGON2D_DEVICE_CUH

#include <stdint.h>

//=== Argon2 定数 ===//
#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_OWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 16)
#define ARGON2_HWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 32)
#define ARGON2_SYNC_POINTS 4
#define ARGON2_PREHASH_DIGEST_LENGTH 64
#define ARGON2_PREHASH_SEED_LENGTH 72
#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13
#define ARGON2_ADDRESSES_IN_BLOCK 128

//=== Blake2b 定数 ===//
#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES 64
#define BLAKE2B_KEYBYTES 64
#define BLAKE2B_SALTBYTES 16
#define BLAKE2B_PERSONALBYTES 16
#define BLAKE2B_ROUNDS 12

//=== 構造体定義 ===//
typedef struct __align__(64) block_ {
    uint64_t v[ARGON2_QWORDS_IN_BLOCK];
} block;

//static __constant__ uint8_t RinSalt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };

// device_argon2d_hash run on device
__device__ void device_argon2d_hash(
    uint8_t* output,
    const uint8_t* input, size_t input_len,
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes,
    block* memory,
    const uint8_t* salt, size_t salt_len
);


#endif // ARGON2D_DEVICE_CUH