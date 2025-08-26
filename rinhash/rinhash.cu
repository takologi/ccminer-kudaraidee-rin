#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

// include global definitions
#include "rinhash_globals.h"

// Include shared device functions (chỉ include .cuh hoặc .h)
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3-256.cu"
#include "blake3_device.cuh"

//#include <chrono>
#include <thread>

#include <miner.h>

// 🚀 GTX 1060 3GB OPTIMIZED: Balance memory usage vs performance
#define MAX_BATCH_BLOCKS 32768

// Kernel: each thread tests one nonce (start_nonce + thread_id), up to num_nonces.
struct RinStatus {
    int found_flag = 0; 
    int stop_flag = 0;
    uint32_t nonce = 0x00000000; 
    uint8_t hash[32] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
    }; 
};

// Device-side constant memory buffers (read-only for device kernels)
__constant__ static uint8_t c_d_work[RINHASH_HEADER_LENGTH]; // 80B work data (pinned, per ping-pong index)
__constant__ static uint32_t  c_d_target[RINHASH_HASH_32B_LENGTH]; // work target hash (8x uint32_t) for comparison

// Device-side global values - the nonce counter used for threads communication and the maximum nonce for this workload
__device__ uint64_t g_next_nonce = 0;  // global nonce counter (is set to starting nonce each round) - uint64_t takes care of overflows
__device__ uint32_t g_max_nonce = 0;   // maximum value of nonce to test



/* ******************************************************************************************************** */
/* ***                                  compose_header_with_nonce                                       *** */     
/* ******************************************************************************************************** */

// Composes per-thread header - copies the header into a local buffer and inserts a nonce into it.
static __device__ __forceinline__ void compose_header_with_nonce(
    uint8_t dest_header[RINHASH_HEADER_LENGTH],
    const uint8_t* __restrict__ orig_header,
    const uint32_t nonce
) {
    memcpy(dest_header, orig_header, RINHASH_HEADER_LENGTH);
    dest_header[RINHASH_NONCE_OFFSET*4 + 0] = (uint8_t)(nonce & 0xFF);
    dest_header[RINHASH_NONCE_OFFSET*4 + 1] = (uint8_t)((nonce >> 8) & 0xFF);
    dest_header[RINHASH_NONCE_OFFSET*4 + 2] = (uint8_t)((nonce >> 16) & 0xFF);
    dest_header[RINHASH_NONCE_OFFSET*4 + 3] = (uint8_t)((nonce >> 24) & 0xFF);
}


/* ******************************************************************************************************** */
/* ***                                  rinhash_cuda_kernel                                             *** */     
/* ******************************************************************************************************** */

// Kernel đơn: mỗi lần chỉ chạy 1 thread
extern "C" __global__ void rinhash_cuda_kernel(
    const uint8_t* input, 
    size_t input_len, 
    uint8_t* output,
    block* memory,      // bộ nhớ argon2 đã cấp phát trên host, truyền vào
    uint32_t m_cost
) {
    // Chỉ 1 thread xử lý
    if (threadIdx.x == 0) {
        uint8_t blake3_out[32];
        light_hash_device(input, input_len, blake3_out);

        const uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
        uint8_t argon2_out[32];
        device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

        uint8_t sha3_out[32];
        sha3_256_device(argon2_out, 32, sha3_out);

        // Copy kết quả ra output
        for (int i = 0; i < 32; i++) output[i] = sha3_out[i];
    }
}


/* ******************************************************************************************************** */
/* ***                                  hash32_to_hex_cu                                                *** */     
/* ******************************************************************************************************** */

// for logging - to format nonce
static __host__ __device__ __forceinline__ inline char* hash32_to_hex_cu(char* hex, const uint32_t* hash32) {
    // Same logic as host: MSB..LSB
    int pos = 0;
    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = hash32[wi];
        uint8_t b3 = (uint8_t)((w >> 24) & 0xFF);
        uint8_t b2 = (uint8_t)((w >> 16) & 0xFF);
        uint8_t b1 = (uint8_t)((w >> 8) & 0xFF);
        uint8_t b0 = (uint8_t)(w & 0xFF);
        // write 8 hex chars
        const char* hexmap = "0123456789abcdef";
        hex[pos++] = hexmap[(b3 >> 4) & 0xF]; hex[pos++] = hexmap[b3 & 0xF];
        hex[pos++] = hexmap[(b2 >> 4) & 0xF]; hex[pos++] = hexmap[b2 & 0xF];
        hex[pos++] = hexmap[(b1 >> 4) & 0xF]; hex[pos++] = hexmap[b1 & 0xF];
        hex[pos++] = hexmap[(b0 >> 4) & 0xF]; hex[pos++] = hexmap[b0 & 0xF];
        hex[pos++] = ' ';
    }
    hex[pos] = '\0';
    return hex;
}

/* ******************************************************************************************************** */
/* ***                                  rinhash_cuda_kernel_optimized                                   *** */     
/* ******************************************************************************************************** */

// 🚀 NEW: Target-aware kernel with atomic solution detection
extern "C" __global__ void rinhash_cuda_kernel_optimized(
    block* memories,
    RinStatus* status,           // status that is shared between the threads and the host
    bool opt_debug
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31; // index vlákna ve warpu

    __align__(4) uint8_t blake3_out[RINHASH_HASH_8B_LENGTH];
    __align__(4) uint8_t argon2_out[RINHASH_HASH_8B_LENGTH];
    __align__(4) uint8_t sha3_out[RINHASH_HASH_8B_LENGTH];

    __align__(4) uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };

    block* memory = memories + tid * ARGON2_SIZE;

    __align__(4) uint8_t local_work[RINHASH_HEADER_LENGTH];                     // local work buffer (80B) to compose header with nonce

    __align__(4) uint64_t nonce_base64;
    uint32_t nonce;


    while (!status->stop_flag) {

        // select a new warp leader
        unsigned mask   = __activemask();
        int      leader = __ffs(mask) - 1;

        if ((int)(threadIdx.x & 31) == leader) {
            nonce_base64 = atomicAdd(&g_next_nonce, __popc(mask));
        }
        // broadcast base_nonce do všech vláken warpu
        nonce_base64 = __shfl_sync(mask, nonce_base64, leader);

        // test the nonce max limit overflow
        if (nonce_base64 > (uint64_t)g_max_nonce) {
            // work's over in this thread, return from the round
            if (opt_debug) {
                printf("thread %3d over, exiting (nonce=%d)\n", 
                    tid, (uint32_t)nonce_base64);
            }
            return;
        }
         // přidělení unikátní nonce lane-u mezi aktivními:
        int lane  = threadIdx.x & 31;
        int rank  = __popc(mask & ((1u << lane) - 1)); // pořadí lane mezi aktivními

        nonce = (uint32_t)nonce_base64 + rank;   // nonce to be tested in this thread


        compose_header_with_nonce(local_work, (const uint8_t *) c_d_work, (const uint32_t) nonce);

        // rinhash computing
        light_hash_device(local_work, RINHASH_HEADER_LENGTH, blake3_out);
        device_argon2d_hash(argon2_out, blake3_out, RINHASH_HASH_8B_LENGTH, 2, ARGON2_SIZE, 1, memory, salt, sizeof(salt));
        sha3_256_device(argon2_out, RINHASH_HASH_8B_LENGTH, sha3_out);
        
        // Check if hash meets target (little-endian comparison from back)
        bool meets_target = true;
#pragma unroll
        for (int i = RINHASH_HASH_32B_LENGTH-1; i >= 0; i--) {
            uint32_t swapped_hash = 
                            ((int32_t)sha3_out[i*4+0] )
                            | ((int32_t)sha3_out[i*4+1] << 8) 
                            | ((int32_t)sha3_out[i*4+2] << 16)
                            | ((int32_t)sha3_out[i*4+3] << 24);
            if (opt_debug) {
                char hex[100];
                printf("thread %3d computing nonce=%10d, TARGET HIGH = %08x, HASH HIGH = %08x (i=%d)\n", 
                    tid, nonce, c_d_target[7-i], swapped_hash, i);
            }
            if (swapped_hash < c_d_target[7-i]) {
                break; // This hash is better, skip checking and set solution
            } else if (swapped_hash > c_d_target[7-i]) {
                meets_target = false;
                break; // This hash is worse, skip checking and work on next nonce
            }
        }
        
        if (meets_target) {
            if (opt_debug) {
                printf("thread %3d found nonce=%d \n", 
                    tid, nonce);
            }
            status->stop_flag = 1;          // set stop flag so other threads stop searching
            status->nonce = nonce;          // set the nonce found
#pragma unroll
            for (int i = 0; i < RINHASH_HASH_8B_LENGTH; i++) {  // set the hash found
                status->hash[i] = sha3_out[i];
            }
            status->found_flag = 1;       // tell the host that nonce was found
        }

        __syncwarp(mask);

    }
}


/* ******************************************************************************************************** */
/* ***                                  check_cuda                                                      *** */     
/* ******************************************************************************************************** */
// Helper: kiểm tra lỗi CUDA
inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error("CUDA error");
    }
}

/* ******************************************************************************************************** */
/* ***                                  rinhash_cuda_cleanup_persistent                                 *** */     
/* ******************************************************************************************************** */
// Cleanup persistent GPU memory (required by rinhash_scanhash.cpp)
extern "C" void rinhash_cuda_cleanup_persistent() {
    // Reset CUDA device to clean up any persistent memory
    cudaDeviceReset();
}

/* ******************************************************************************************************** */
/* ***                                  rinhash_cuda                                                    *** */     
/* ******************************************************************************************************** */
// RinHash CUDA implementation (single)
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint8_t *d_input = nullptr;
    uint8_t *d_output = nullptr;
    block* d_memory = nullptr;
    uint32_t m_cost = 64;

    cudaError_t err;

    // Alloc device memory
    err = cudaMalloc(&d_input, input_len);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc input fail\n"); return; }

    err = cudaMalloc(&d_output, 32);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc output fail\n"); cudaFree(d_input); return; }

    err = cudaMalloc(&d_memory, m_cost * sizeof(block));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memory fail\n"); cudaFree(d_input); cudaFree(d_output); return; }

    // Copy input
    err = cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy input fail\n"); cudaFree(d_input); cudaFree(d_output); cudaFree(d_memory); return; }

    // Launch kernel
    rinhash_cuda_kernel<<<512, 4096>>>(d_input, input_len, d_output, d_memory, m_cost);
    cudaDeviceSynchronize();
    check_cuda("rinhash_cuda_kernel");

    // Copy result
    err = cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy output fail\n"); }

    // Free
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_memory);
}

/* ******************************************************************************************************** */
/* ***                                  rinhash_cuda_batch_optimized                                    *** */     
/* ******************************************************************************************************** */

// 🚀 OPTIMIZED: Target-aware batch processing for faster mining
extern "C" void rinhash_cuda_batch_optimized(
    uint8_t* hash_found,
    uint32_t start_nonce,
    uint32_t num_blocks,
    uint32_t* target,           // Target for early termination
    uint32_t* solution_found,   // Output: 1 if solution found
    uint32_t* solution_nonce,    // Output: winning nonce
    int thr_id
) {
    // if (num_blocks > MAX_BATCH_BLOCKS) {
    //     fprintf(stderr, "Batch too large (max %u)\n", MAX_BATCH_BLOCKS);
    //     return;
    // }

    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    block* d_memories = nullptr;
    
    size_t memories_size = CUDA_THREADS_PER_BLOCK * CUDA_BLOCKS * ARGON2_SIZE * sizeof(block);


    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&d_memories, memories_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memories fail\n"); cudaFree(d_headers); cudaFree(d_outputs); return; }

    // Initialize data
    // create nonblocking CUDA stream and event
    cudaStream_t kernel_stream;
    cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking);
    cudaEvent_t evt;
    cudaEventCreate(&evt);

    // Mapped pinned status struct for zero-copy signaling
    RinStatus* h_status = nullptr;
    RinStatus* d_status = nullptr;
    cudaHostAlloc((void**)&h_status, sizeof(RinStatus), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&d_status, (void*)h_status, 0);

    // set init values for status
    h_status->found_flag = 0;
    h_status->stop_flag = 0;
    h_status->nonce = 0x00000000; 
    memset(h_status->hash, 0xFF, RINHASH_HASH_8B_LENGTH); 


    uint64_t start_nonce64 = (uint64_t)start_nonce; //TO-DO get start_nonce
    uint32_t max_nonce = (uint32_t) start_nonce + num_blocks - 1;


    // set start and maximum nonces for kernel threads to know where to start and when to finish
    cudaMemcpyToSymbol(g_next_nonce, &start_nonce64, sizeof(uint64_t));
    cudaMemcpyToSymbol(g_max_nonce, &max_nonce, sizeof(uint32_t));

printf("cuda started, start nonce = %d\n", start_nonce);


/* ****************************** KERNEL CALL ******************************************/
    rinhash_cuda_kernel_optimized<<<CUDA_BLOCKS, CUDA_THREADS_PER_BLOCK, 0, kernel_stream>>>(
        d_memories, d_status, opt_debug
    );
/* ****************************** KERNEL CALL ******************************************/

    // add the event 
    cudaEventRecord(evt, kernel_stream);

    // main loop, waiting either to finding a nonce or to kernel finish
    while (true) {
        // Poll mapped host memory without any CUDA memcpy
        if (h_status->found_flag) {
            *solution_nonce = h_status->nonce;
            memcpy(hash_found, h_status->hash, RINHASH_HASH_8B_LENGTH);
            *solution_found = 1;
            printf("RinHash_mine: HASH FOUND (mapped) ! nonce=%u\n", *solution_nonce);
            goto cleanup;
        }

        cudaError_t status = cudaEventQuery(evt);
        if (status == cudaSuccess) {
            // kernel finished
            if (h_status->found_flag) {
                *solution_nonce = h_status->nonce;
                memcpy(hash_found, h_status->hash, RINHASH_HASH_8B_LENGTH);
                *solution_found = 1;
                printf("RinHash_mine: HASH FOUND AFTER KERNEL FINISH (mapped) !!! nonce=%u\n", *solution_nonce);
            } else {
                *solution_nonce = max_nonce;
                *solution_found = 0;
            }
printf("cuda finished, end nonce = %d\n", max_nonce);
            *solution_nonce = max_nonce;
            *solution_found = 0;
            goto cleanup;
        } else if (status != cudaErrorNotReady) {
            // some error happened
            printf("CUDA event error: %d\n", status);
            h_status->stop_flag = 1;
            solution_found = 0;
printf("cuda error\n");
            goto cleanup;
        } else if (work_restart[thr_id].restart){
            // work has to be restarted = stop the kernels
            h_status->stop_flag = 1;
            solution_found = 0;
printf("work_restart\n");
            goto cleanup;
        } else {
            // kernel is working, yield shortly so that CPU can do other things
            std::this_thread::yield();
        }
    }

cleanup:
    // wait for kernel to finish, then free mapped memory
    if (h_status) cudaFreeHost(h_status);
    cudaFree(d_memories);
}



/* ******************************************************************************************************** */
/* ***                                  blockheader_to_bytes                                            *** */     
/* ******************************************************************************************************** */

// Helper function to convert a block header to bytes
extern "C" void blockheader_to_bytes(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output,
    size_t* output_len
) {
    size_t offset = 0;
    memcpy(output + offset, version, 4); offset += 4;
    memcpy(output + offset, prev_block, 32); offset += 32;
    memcpy(output + offset, merkle_root, 32); offset += 32;
    memcpy(output + offset, timestamp, 4); offset += 4;
    memcpy(output + offset, bits, 4); offset += 4;
    memcpy(output + offset, nonce, 4); offset += 4;
    *output_len = offset;
}


/* ******************************************************************************************************** */
/* ***                                  RinHash                                                         *** */     
/* ******************************************************************************************************** */

// Main RinHash function that would be called from outside
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
) {
    uint8_t block_header[80]; // Standard block header size
    size_t block_header_len;
    blockheader_to_bytes(
        version,
        prev_block,
        merkle_root,
        timestamp,
        bits,
        nonce,
        block_header,
        &block_header_len
    );
    rinhash_cuda(block_header, block_header_len, output);
}


/* ******************************************************************************************************** */
/* ***                                  is_better                                                       *** */     
/* ******************************************************************************************************** */

bool is_better(uint8_t* hash1, uint8_t* hash2) {
    for (int i = RINHASH_HASH_8B_LENGTH-1; i >= 0; i--) {
        if (hash1[i] < hash2[i]) return true;
        if (hash1[i] > hash2[i]) return false;
    }
    return false; // equal
}


/* ******************************************************************************************************** */
/* ***                                  RinHash_mine_optimized                                          *** */     
/* ******************************************************************************************************** */

// 🚀 OPTIMIZED: Enhanced mining function with target-aware early termination
extern "C" void RinHash_mine_optimized(
    const uint32_t* work_data,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* target,           // 8 x uint32_t target  
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    uint32_t* solution_found,    // 1 if target was met
    int thr_id
) {
    // if (num_nonces > MAX_BATCH_BLOCKS) {
    //     fprintf(stderr, "Mining batch too large (max %u)\n", MAX_BATCH_BLOCKS);
    //     return;
    // }
    
    uint8_t hash_found[RINHASH_HASH_8B_LENGTH];

    uint32_t solution_nonce = 0;

    char hex[100];
    applog(LOG_INFO, "Target: %s", 
        hash32_to_hex_cu(hex, (uint32_t *)target_hash));

    // copy header to device memory
    cudaMemcpyToSymbol(c_d_work, work_data, RINHASH_HEADER_LENGTH);
    cudaMemcpyToSymbol(c_d_target, target, RINHASH_HASH_8B_LENGTH);

    
    // Use optimized kernel with target checking
    rinhash_cuda_batch_optimized(
        hash_found, start_nonce, num_nonces,
        target, solution_found, &solution_nonce, thr_id
    );

    if (*solution_found) {
        // Solution found! Extract the winning hash
        *found_nonce = solution_nonce;
        memcpy(best_hash, hash_found, RINHASH_HASH_8B_LENGTH);
        applog(LOG_INFO, "Found:  %s", 
            hash32_to_hex_cu(hex, (uint32_t *)best_hash));
    } else {
        // No solution found, set hash to 0xFFFF...FFFF and nonce to the last nonce tested
        *found_nonce = start_nonce + num_nonces - 1;
        memset(best_hash, 0xFF, RINHASH_HASH_8B_LENGTH);
    }
}

