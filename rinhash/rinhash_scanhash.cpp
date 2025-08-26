#include <stdio.h>
#include <cstdint>
#include <sys/time.h>

#include <memory.h>

#include <miner.h>
#include "cuda_helper.h"

// include global definitions
#include "rinhash_globals.h"


using namespace std;

// External reference to RinHash CUDA functions
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
);

// 🚀 NEW: Optimized mining function with target-aware early termination
extern "C" void RinHash_mine_optimized(
    const uint32_t* work_data,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* target,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    uint32_t* solution_found,
    int thr_id
);

// Thread-local variables
thread_local uint32_t *d_hash = NULL;
thread_local uint8_t *d_rinhash_out = NULL;

// Initialization function for RinHash algorithm
extern "C" void rinhash_init(int thr_id)
{
    
}

// External reference to persistent memory cleanup
extern "C" void rinhash_cuda_cleanup_persistent();

// Cleanup function for RinHash algorithm
extern "C" void rinhash_free(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    // Free allocated memory
    cudaFree(d_hash);
    cudaFree(d_rinhash_out);

    d_hash = NULL;
    d_rinhash_out = NULL;
    
    // Clean up persistent GPU memory
    rinhash_cuda_cleanup_persistent();
}

// Main scanning function that tries different nonces to find a valid hash
int scanhash_rinhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[RINHASH_NONCE_OFFSET];
    uint32_t nonce = first_nonce;
    if (opt_benchmark)
        ptarget[7] = 0x0000ff;

    // 🚀 GTX 1060 3GB OPTIMIZED: Auto-detect GPU memory and set optimal batch size
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    uint32_t batch_size;
    if (total_mem <= 3ULL * 1024 * 1024 * 1024) { // <= 3GB VRAM
        batch_size = 8192;  // 8K nonces for 3GB cards (GTX 1060 3GB)
    } else if (total_mem <= 6ULL * 1024 * 1024 * 1024) { // <= 6GB VRAM  
        batch_size = 16384; // 16K nonces for 6GB cards (GTX 1060 6GB)
    } else { // 8GB+ VRAM
        batch_size = 32768; // 32K nonces for high-end cards
    }
    
    if (opt_debug) {
        applog(LOG_INFO, "GPU Memory: %.1f GB, Using batch_size: %u", 
               total_mem / (1024.0 * 1024.0 * 1024.0), batch_size);
    }
    uint32_t found_nonce = 0;
    uint8_t best_hash[RINHASH_HASH_8B_LENGTH];
    uint8_t target_hash[RINHASH_HASH_8B_LENGTH];

    // Convert target to bytes
    for (int i = 0; i < RINHASH_HASH_32B_LENGTH; i++) {
        uint32_t tmp = ptarget[i];
        target_hash[i*4+0] = tmp & 0xff;
        target_hash[i*4+1] = (tmp >> 8) & 0xff;
        target_hash[i*4+2] = (tmp >> 16) & 0xff;
        target_hash[i*4+3] = (tmp >> 24) & 0xff;
    }

    work->valid_nonces = 0;
    cudaSetDevice(device_map[thr_id]);
    
    // Track start time for hashrate calculation
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    do {
        uint32_t current_batch_size = min(batch_size, max_nonce - nonce);
        
        if (current_batch_size <= 0) {
            *hashes_done = nonce - first_nonce;
            return 0;
        }

        // 🚀 OPTIMIZED: Use target-aware mining for early termination
        uint32_t solution_found = 0;
        uint32_t target_words[RINHASH_HASH_32B_LENGTH];
        
        // Convert target to uint32_t array for GPU kernel
        for (int i = 0; i < RINHASH_HASH_32B_LENGTH; i++) {
            target_words[i] = ptarget[RINHASH_HASH_32B_LENGTH-1-i]; // reverse order for proper comparison
        }
        
        RinHash_mine_optimized(
            pdata,
            nonce,
//            current_batch_size,
            max_nonce - nonce + 1,
            target_words,
            &found_nonce,
            target_hash,
            best_hash,
            &solution_found,
            thr_id

        );

        *hashes_done = nonce - first_nonce + current_batch_size;
        
        // Update thread hashrate every batch
        gettimeofday(&tv_end, NULL);
        double elapsed = (tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;
        if (elapsed > 0.0) {
                extern double thr_hashrates[];
                extern pthread_mutex_t stats_lock;
                
                pthread_mutex_lock(&stats_lock);
                thr_hashrates[thr_id] = (*hashes_done) / elapsed;
                pthread_mutex_unlock(&stats_lock);
        }

        // 🚀 OPTIMIZED: Check if solution was found or improved hash
        if (solution_found || found_nonce != nonce) {
            // Convert best_hash to vhash for verification
            uint32_t _ALIGN(64) vhash[RINHASH_HASH_32B_LENGTH];
            for (int i = 0; i < RINHASH_HASH_32B_LENGTH; i++) {
                vhash[i] = 0;
                vhash[i] |= (int32_t)best_hash[i*4+0] ;
                vhash[i] |= (int32_t)best_hash[i*4+1] << 8;
                vhash[i] |= (int32_t)best_hash[i*4+2] << 16;
                vhash[i] |= (int32_t)best_hash[i*4+3] << 24;
            }
            printf("SOLUTION FOUND? nonce=%10d, TARGET HIGH = %08x, HASH HIGH = %08x \n", 
                    nonce, ptarget[7], vhash[7]);
            if (fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                rinhash_set_target_ratio(work, vhash);  // Use RinHash-specific function
                work->nonces[0] = found_nonce;
                pdata[19] = found_nonce + 1;
                
                if (solution_found && opt_debug) {
                    applog(LOG_INFO, "🚀 GPU Optimized Early Solution Found! Nonce: %08x", found_nonce);
                }
                
                return 1;
            } else {
                gpu_increment_reject(thr_id);
                if (opt_debug && solution_found) {
                    applog(LOG_WARNING, "GPU found potential solution but failed verification");
                }
            }
        }

        nonce += current_batch_size;

    } while (nonce < max_nonce && !work_restart[thr_id].restart);

    pdata[19] = nonce;
    *hashes_done = nonce - first_nonce;
    return 0;
}

// Empty function to detect algorithm - needed by ccminer
extern "C" void rinhash_hash(void *output, const void *input)
{
    // Just pass through to the CUDA implementation
    RinHash(
        (const uint32_t*)input,
        (const uint32_t*)((const uint8_t*)input + 4),
        (const uint32_t*)((const uint8_t*)input + 36),
        (const uint32_t*)((const uint8_t*)input + 68),
        (const uint32_t*)((const uint8_t*)input + 72),
        (const uint32_t*)((const uint8_t*)input + 76),
        (uint8_t*)output
    );
}
