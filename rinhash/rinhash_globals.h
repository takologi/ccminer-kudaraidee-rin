#ifndef RINHASH_GLOBALS_H
#define RINHASH_GLOBALS_H

// Performance knobs
#ifndef CUDA_MAX_REPETITIONS_PER_THREAD
#define CUDA_MAX_REPETITIONS_PER_THREAD 32
#endif

#ifndef CUDA_THREADS_PER_BLOCK
#define CUDA_THREADS_PER_BLOCK 256
#endif

#ifndef CUDA_BLOCKS
#define CUDA_BLOCKS 12
#endif

// Argon2d parameter - memory size in blocks (= size in kb as 1 block = 1024 bytes)
#ifndef ARGON2_SIZE
#define ARGON2_SIZE 64
#endif

#ifndef RINHASH_HEADER_LENGTH
#define RINHASH_HEADER_LENGTH 80
#endif

#ifndef RINHASH_HASH_32B_LENGTH
#define RINHASH_HASH_32B_LENGTH 8
#endif

#ifndef RINHASH_HASH_8B_LENGTH
#define RINHASH_HASH_8B_LENGTH RINHASH_HASH_32B_LENGTH * 4
#endif


#ifndef RINHASH_NONCE_OFFSET
#define RINHASH_NONCE_OFFSET 19
#endif

#endif