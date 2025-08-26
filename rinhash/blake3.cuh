#ifndef RINHASH_BLAKE3_DEVICE_CUH
#define RINHASH_BLAKE3_DEVICE_CUH 

#include <stdint.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <thrust/host_vector.h>

using namespace std;

using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;
 
const u32 OUT_LEN = 32;
const u32 KEY_LEN = 32;
const u32 BLOCK_LEN = 64;
const u32 CHUNK_LEN = 1024;
// Multiple chunks make a snicker bar :)
const u32 SNICKER = 1U << 10;
// Factory height and snicker size have an inversly propotional relationship
// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
const u32 FACTORY_HT = 5;

const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
const u32 KEYED_HASH = 1 << 4;
const u32 DERIVE_KEY_CONTEXT = 1 << 5;
const u32 DERIVE_KEY_MATERIAL = 1 << 6;

const int usize = sizeof(u32) * 8;
// Number of threads per thread block
__constant__ const int NUM_THREADS = 16;

// redefine functions, but for the GPU
// all of them are the same but with g_ prefixed
__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

__device__ __forceinline__ u32 g_rotr(u32 value, int shift); 

__device__ void g_g(u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my);

__device__ __forceinline__ void g_round(u32 state[16], u32 m[16]);

__device__ __forceinline__ void g_permute(u32 m[16]);

// custom memcpy, apparently cuda's memcpy is slow 
// when called within a kernel
__device__ __forceinline__ void g_memcpy(u32 * __restrict__ lhs, const u32 * __restrict__ rhs, int size);

// custom memset
template<typename T, typename ptr_t>

__device__ __forceinline__ void g_memset(ptr_t dest, T val, int count);

__device__ void g_compress(
    u32 *chaining_value,
    u32 *block_words,
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *state
);
__device__ void g_words_from_little_endian_bytes(
    u8 *bytes, u32 *words, u32 bytes_len
);
struct Chunk {
    // use only when it is a leaf node
    // leaf data may have less than 1024 bytes
    u8 leaf_data[1024];
    u32 leaf_len;
    // use in all other cases
    // data will always have 64 bytes
    u32 data[16];
    u32 flags;
    u32 raw_hash[16];
    u32 key[8];
    // only useful for leaf nodes
    u64 counter;
    // Constructor for leaf nodes
    __device__ __host__ Chunk(char *input, int size, u32 _flags, u32 *_key, u64 ctr){
        counter = ctr;
        flags = _flags;
        memcpy(key, _key, 8*sizeof(*key));
        memset(leaf_data, 0, 1024);
        memcpy(leaf_data, input, size);
        leaf_len = size;
    }
    __device__ __host__ Chunk(u32 _flags, u32 *_key) {
        counter = 0;
        flags = _flags;
        memcpy(key, _key, 8*sizeof(*key));
        leaf_len = 0;
    }
    __device__ __host__ Chunk() {}
    // Chunk() : leaf_len(0) {}
    // process data in sizes of message blocks and store cv in hash
    void compress_chunk(u32=0);
    __device__ void g_compress_chunk(u32=0);
};


// Remove recursion from device kernel
__global__ void compute(Chunk *data, int l, int r);

/*
using thrust_vector = thrust::host_vector<Chunk>;
thrust::host_vector<int> h_vec(100);
*/

// CPU version of light_hash (unchanged)
void light_hash(Chunk *data, int N, Chunk *result, Chunk *memory_bar);

// Device-callable version of light_hash
__device__ void light_hash_device(const uint8_t* input, size_t input_len, uint8_t* output);

// Alias for compatibility with other device code
__device__ void blake3_hash_device(const uint8_t* input, size_t input_len, uint8_t* output);

#endif // RINHASH_BLAKE3_DEVICE_CUH
