#pragma once

#include <deep_gemm/common/types.hpp>

#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

/**
 * @brief Configuration for TMA (Tensor Memory Accelerator) multicast operations
 * 
 * TMA multicast allows a single CTA to broadcast data to multiple CTAs, reducing
 * global memory bandwidth requirements. This is crucial for performance optimization
 * when multiple thread blocks need the same data.
 */
struct MulticastConfig {
    int num_multicast;        // Number of CTAs participating in multicast (1 or 2)
    bool is_multicast_on_a;   // Whether to multicast matrix A (vs matrix B)

    MulticastConfig(const int& num_multicast, const bool& is_multicast_on_a):
        num_multicast(num_multicast), is_multicast_on_a(is_multicast_on_a) {
        // Only support 1x or 2x multicast patterns
        DG_HOST_ASSERT(1 <= num_multicast and num_multicast <= 2);
    }
};

/**
 * @brief Shared memory configuration for optimal memory access patterns
 * 
 * Shared memory swizzling is critical for avoiding bank conflicts in CUDA shared memory.
 * Each bank can service one 4-byte request per cycle, so proper swizzling ensures
 * coalesced memory access patterns across the 32 memory banks.
 */
struct SharedMemoryConfig {
    int smem_size;          // Total shared memory usage in bytes
    int swizzle_a_mode;     // Swizzling pattern for matrix A (16B, 32B, 64B, or 128B)
    int swizzle_b_mode;     // Swizzling pattern for matrix B 
    int swizzle_cd_mode;    // Swizzling pattern for output matrix C/D
};

/**
 * @brief Thread block configuration for different GPU architectures
 * 
 * SM90 (Hopper) uses a clear separation between TMA threads (memory operations)
 * and math threads (computation). SM100 (Blackwell) uses a different organization
 * with non-epilogue threads handling main computation and epilogue threads for output.
 */
struct ThreadConfig {
    int num_threads;        // Total threads per thread block

    // SM90 Hopper Architecture
    int num_tma_threads;    // Threads handling TMA operations (typically 128)
    int num_math_threads;   // Threads performing WGMMA operations (128 or 256)

    // SM100 Blackwell Architecture  
    int num_non_epilogue_threads;  // Threads for main GEMM computation
    int num_epilogue_threads;      // Threads for epilogue operations

    /**
     * @brief Create SM90-specific thread configuration
     * @param num_tma_threads Usually 128 threads for TMA operations
     * @param num_math_threads 128 for 64x* blocks, 256 for larger blocks
     */
    static ThreadConfig sm90(const int& num_tma_threads,
                             const int& num_math_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_tma_threads + num_math_threads;
        config.num_tma_threads = num_tma_threads;
        config.num_math_threads = num_math_threads;
        return config;
    }

    /**
     * @brief Create SM100-specific thread configuration
     * Different pipeline organization compared to SM90
     */
    static ThreadConfig sm100(const int& num_non_epilogue_threads,
                              const int& num_epilogue_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_non_epilogue_threads + num_epilogue_threads;
        config.num_non_epilogue_threads = num_non_epilogue_threads;
        config.num_epilogue_threads = num_epilogue_threads;
        return config;
    }
};

/**
 * @brief Comprehensive GEMM kernel configuration
 * 
 * This structure encapsulates all parameters needed to generate optimal GEMM kernels.
 * The configuration is determined by sophisticated heuristics that consider hardware
 * constraints, problem size, and performance characteristics.
 */
struct GemmConfig {
    // Kernel type and data configuration
    GemmType gemm_type;           // Normal, MGroupedContiguous, MGroupedMasked, etc.
    KernelType kernel_type;       // Kernel1D1D, Kernel1D2D, KernelNoSF
    at::ScalarType ab_dtype, cd_dtype;  // Input (A,B) and output (C,D) data types
    cute::UMMA::Major major_a;    // Matrix A layout: K-major or M-major
    cute::UMMA::Major major_b;    // Matrix B layout: K-major or N-major
    bool with_accumulation;       // Whether to accumulate with existing C matrix

    // Block tiling configuration
    int block_m, block_n, block_k;  // Thread block tile sizes
    int num_stages, num_last_stages; // Software pipelining stages

    // Hardware resource configuration
    int num_sms;                  // Number of SMs to utilize (for L2 optimization)
    int tc_util;                  // Tensor core utilization percentage (SM100 only)

    // Detailed sub-configurations
    MulticastConfig multicast_config;      // TMA multicast settings
    SharedMemoryConfig smem_config;        // Shared memory layout
    ThreadConfig thread_config;            // Thread block organization
};

/**
 * @brief Check if TMA multicast configuration is legal for given problem size
 * 
 * TMA multicast requires specific divisibility constraints to work correctly:
 * - The number of blocks along the multicast dimension must be divisible by num_multicast
 * - The total number of SMs must be divisible by num_multicast for load balancing
 * 
 * @param shape_dim Problem size dimension (M or N)
 * @param block_dim Block size dimension (block_m or block_n)
 * @param num_multicast Number of multicast participants (1 or 2)
 * @param num_sms Total number of SMs available
 * @param require_divisible Whether strict divisibility is required
 */
static bool is_multicast_legal(const int& shape_dim, const int& block_dim,
                               const int& num_multicast, const int& num_sms,
                               const bool& require_divisible) {
    // Check if number of blocks is compatible with multicast pattern
    const bool& divisible = ceil_div(shape_dim, block_dim) % num_multicast == 0 or not require_divisible;
    // Ensure SM count is compatible with multicast pattern
    return divisible and num_sms % num_multicast == 0;
}

/**
 * @brief Determine optimal shared memory swizzling mode to avoid bank conflicts
 * 
 * CUDA shared memory has 32 banks, each 4 bytes wide. When multiple threads in a warp
 * access the same bank simultaneously, bank conflicts occur, reducing throughput.
 * Swizzling permutes memory addresses to distribute accesses across banks.
 * 
 * @param block_size Size of data block being accessed
 * @param elem_size Size of each element in bytes
 * @return Swizzling mode (128B, 64B, 32B, or 16B)
 */
template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving is enabled
    // 16B actually means non-swizzling but with interleaving
    // Try larger swizzling modes first for better performance
    for (const int& mode: {128, 64, 32, 16}) {
        // Check if total memory access size is compatible with swizzling mode
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}

/**
 * @brief Calculate comprehensive shared memory configuration
 * 
 * Shared memory is partitioned into several regions:
 * 1. Tensor maps (for TMA operations)
 * 2. Input matrices A and B (multi-stage buffering)
 * 3. Scaling factors for FP8 operations
 * 4. Output accumulation buffer
 * 5. Synchronization barriers
 * 6. Tensor memory pointers
 * 
 * @param gemm_type Type of GEMM operation
 * @param kernel_type Kernel variant being used
 * @param m,n,k Problem dimensions
 * @param block_m,block_n,block_k Block dimensions
 * @param major_a,major_b Matrix layouts
 * @param ab_dtype,cd_dtype Data types
 * @param num_stages Number of pipeline stages
 * @param multicast_config TMA multicast configuration
 * @return Complete shared memory configuration
 */
template <typename ArchSpec>
static SharedMemoryConfig get_smem_config(const GemmType& gemm_type, const KernelType& kernel_type,
                                          const int& m, const int& n, const int& k,
                                          const int& block_m, const int& block_n, const int& block_k,
                                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                          const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                          const int& num_stages, const MulticastConfig& multicast_config) {
    // Calculate element sizes for memory usage computation
    const int& ab_elem_size = static_cast<int>(c10::elementSize(ab_dtype));
    const int& cd_elem_size = static_cast<int>(c10::elementSize(cd_dtype));

    // Determine actual loading block sizes (may differ with multicast)
    const int& load_block_m = ArchSpec::get_ab_load_block_m(multicast_config, block_m);
    const int& load_block_n = ArchSpec::get_ab_load_block_n(multicast_config, block_n);
    
    // Calculate swizzling modes based on memory access patterns
    // For K-major layout, swizzle based on K dimension; otherwise use M/N dimension
    const int& swizzle_a_mode = get_swizzle_mode(major_a == cute::UMMA::Major::K ? block_k : load_block_m, ab_elem_size);
    const int& swizzle_b_mode = get_swizzle_mode(major_b == cute::UMMA::Major::K ? block_k : load_block_n, ab_elem_size);
    const int& swizzle_cd_mode = ArchSpec::enable_cd_swizzle(cd_dtype) ? get_swizzle_mode(block_n, cd_elem_size) : 0;

    // Different architectures have different epilogue pipeline requirements
    const int& smem_cd = ArchSpec::get_smem_cd_size(kernel_type, block_m, block_n, swizzle_cd_mode, cd_dtype);

    // Input matrices A and B shared memory (per pipeline stage)
    const int& smem_a_per_stage = load_block_m * block_k * ab_elem_size;
    const int& smem_b_per_stage = load_block_n * block_k * ab_elem_size;

    // Scaling factors (SF) shared memory for FP8 operations
    // FP8 requires scaling factors to convert to/from higher precision
    const auto& [smem_sfa_per_stage, smem_sfb_per_stage] =
        ArchSpec::get_sf_smem_size_per_stage(kernel_type, block_m, block_n, block_k, ab_dtype, cd_dtype);
    const int& smem_extra_sfb = ArchSpec::get_extra_sfb_smem_size(m, n, k, block_m, block_n, block_k);

    // Synchronization primitives shared memory
    const int& smem_barrier = ArchSpec::get_barrier_smem_size(num_stages);      // mbarrier objects
    const int& smem_tmem_ptr = ArchSpec::get_tmem_ptr_smem_size();              // Tensor memory pointers
    const int& smem_tensor_map = ArchSpec::get_tensormap_smem_size(gemm_type);  // TMA tensor map descriptors

    // Calculate total shared memory usage
    int smem_size = 0;
    smem_size += smem_tensor_map;                           // TMA tensor maps
    smem_size += smem_cd;                                   // Output accumulation buffer
    smem_size += num_stages * smem_a_per_stage;             // Multi-stage matrix A buffers
    smem_size += num_stages * smem_b_per_stage;             // Multi-stage matrix B buffers  
    smem_size += num_stages * smem_sfa_per_stage;           // Multi-stage scaling factor A
    smem_size += num_stages * smem_sfb_per_stage;           // Multi-stage scaling factor B
    smem_size += smem_extra_sfb;                            // Extra scaling factor storage
    smem_size += smem_barrier;                              // Synchronization barriers
    smem_size += smem_tmem_ptr;                             // Tensor memory pointers

    return SharedMemoryConfig {
        .smem_size = smem_size,
        .swizzle_a_mode = swizzle_a_mode,
        .swizzle_b_mode = swizzle_b_mode,
        .swizzle_cd_mode = swizzle_cd_mode,
    };
}

/**
 * @brief Main heuristic function to determine optimal GEMM configuration
 * 
 * This is the heart of DeepGEMM's performance optimization. The algorithm considers:
 * 1. Wave utilization: Minimize the number of waves to reduce latency
 * 2. Last wave efficiency: Maximize SM utilization in the final wave  
 * 3. Memory constraints: Ensure shared memory usage stays within limits
 * 4. Hardware capabilities: Use architecture-specific optimizations
 * 5. TMA multicast: Enable when beneficial for memory bandwidth
 * 
 * The heuristic uses a sophisticated multi-criteria optimization approach to balance
 * compute efficiency, memory bandwidth, and hardware resource utilization.
 * 
 * @param gemm_type Type of GEMM operation (normal, grouped, etc.)
 * @param kernel_type Kernel variant to use
 * @param m,n,k Problem dimensions  
 * @param num_groups Number of expert groups (for MoE)
 * @param major_a,major_b Matrix memory layouts
 * @param ab_dtype,cd_dtype Input and output data types
 * @param with_accumulation Whether to accumulate with existing output
 * @param num_sms Number of available SMs
 * @return Optimal GEMM configuration
 */
template <typename ArchSpec>
static GemmConfig get_best_config(const GemmType& gemm_type, const KernelType& kernel_type,
                                  const int& m, const int& n, const int& k, const int& num_groups,
                                  const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                  const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                  const bool& with_accumulation, const int& num_sms) {
    // Validate supported data types
    DG_HOST_ASSERT(ab_dtype == torch::kFloat8_e4m3fn or ab_dtype == torch::kBFloat16);
    DG_HOST_ASSERT(cd_dtype == torch::kBFloat16 or cd_dtype == torch::kFloat);

    // ===== STEP 1: Determine candidate block sizes =====
    
    // M block size candidates - different for different GEMM types
    auto block_ms = std::vector{64, 128, 256};
    if (gemm_type == GemmType::MGroupedContiguous)
        // Contiguous layout requires specific alignment for TMA operations
        block_ms = std::vector{get_mk_alignment_for_contiguous_layout()};
    if (gemm_type == GemmType::MGroupedMasked)  
        // Exclude 256 for performance reasons in masked operations
        block_ms = std::vector{64, 128};
    
    // N block size candidates - architecture-specific
    const auto block_ns = ArchSpec::get_block_n_candidates(cd_dtype);

    // K block size - determined by memory bandwidth optimization
    // 128 bytes total per element type ensures optimal TMA transfer sizes
    const auto& block_k = 128 / static_cast<int>(c10::elementSize(ab_dtype));

    // ===== STEP 2: Wave analysis utility functions =====
    
    /**
     * Calculate total number of thread blocks needed
     * This determines the total amount of work to be done
     */
    const auto& get_num_blocks = [=](const int& block_m, const int& block_n) {
        return ceil_div(m, block_m) * ceil_div(n, block_n) * num_groups;
    };
    
    /**
     * Calculate number of waves (iterations) needed to complete all work
     * Fewer waves generally means better performance due to reduced overhead
     */
    const auto& get_num_waves = [=](const int& block_m, const int& block_n) {
        return ceil_div(get_num_blocks(block_m, block_n), num_sms);
    };
    
    /**
     * Calculate SM utilization in the last (potentially incomplete) wave
     * Higher last wave utilization means better overall hardware utilization
     */
    const auto& get_last_wave_util = [=](const int& block_m, const int& block_n) {
        const auto& num_last_blocks = get_num_blocks(block_m, block_n) % num_sms;
        return num_last_blocks == 0 ? num_sms : num_last_blocks;
    };

    // ===== STEP 3: Block size selection using wave optimization =====
    
    int best_block_m = 0, best_block_n = 0;
    int best_num_waves = 0, best_last_util = 0;
    
    // Exhaustive search over all valid block size combinations
    for (const auto& block_m: block_ms) {
        for (const auto& block_n: block_ns) {
            const int& num_waves = get_num_waves(block_m, block_n);
            const auto& last_util = get_last_wave_util(block_m, block_n);
            
            // Check if this block size combination is architecturally valid
            if (not ArchSpec::is_block_size_legal(kernel_type, major_a, major_b, ab_dtype, cd_dtype, block_m, block_n, block_k))
                continue;

            bool success = false;
            
            // Primary criterion: minimize number of waves
            if (best_block_m == 0 or best_block_n == 0 or num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                // Secondary criterion: maximize last wave utilization
                success = last_util > best_last_util;
                
                if (last_util == best_last_util) {
                    // Tertiary criteria for tie-breaking:
                    
                    // Case 1: Same M block size, prefer smaller N (less wasted computation)
                    success |= block_m == best_block_m and block_n < best_block_n;
                    
                    // Case 2: Same N block size, prefer smaller M (less wasted computation)  
                    success |= block_n == best_block_n and block_m < best_block_m;
                    
                    // Case 3: Different M and N, prefer larger N (better memory coalescing)
                    // But don't exceed actual problem dimensions
                    success |= block_m != best_block_m and block_n > best_block_n 
                               and block_n <= n and block_m <= m;
                }
            }

            // Update best configuration if this one is better
            if (success) {
                best_block_m = block_m, best_block_n = block_n;
                best_num_waves = num_waves, best_last_util = last_util;
            }
        }
    }
    DG_HOST_ASSERT(best_block_m > 0 and best_block_n > 0);

    // ===== STEP 4: TMA multicast configuration selection =====
    
    MulticastConfig best_multicast_config = {1, false};  // Default: no multicast
    
    // Check multicast legality for both A and B matrices
    const auto& [is_legal_on_a, is_legal_on_b] = ArchSpec::get_multicast_legality(
        gemm_type, num_groups, m, n, best_block_m, best_block_n, num_sms);
    
    const bool is_legal[2] = {is_legal_on_b, is_legal_on_a};
    bool order[2] = {false, true};  // Default order: try B first, then A
    
    // If M block is larger than N block, try A multicast first
    if (best_block_m > best_block_n)
        std::swap(order[0], order[1]);
    
    // Try to enable 2x multicast if problem size is large enough
    for (const bool& is_multicast_on_a: order) {
        if (m >= 512 and is_legal[static_cast<int>(is_multicast_on_a)]) {
            best_multicast_config = {2, is_multicast_on_a};
            break;
        }
    }

    // ===== STEP 5: Pipeline stage selection =====
    
    // Always pick the largest number of stages that fits in shared memory
    // More stages = better latency hiding through deeper pipelining
    constexpr int smem_capacity = ArchSpec::smem_capacity;
    int best_num_stages = 0;
    SharedMemoryConfig best_smem_config;
    
    // Try from maximum stages (12) down to 1
    for (int num_stages = 12; num_stages > 0; -- num_stages) {
        // Check if this number of stages is architecturally supported
        if (not ArchSpec::is_num_stages_legal(ab_dtype, cd_dtype, num_stages, best_block_m, best_block_n, block_k))
            continue;

        // Calculate shared memory usage for this configuration
        best_smem_config = get_smem_config<ArchSpec>(gemm_type, kernel_type,
                                                     m, n, k,
                                                     best_block_m, best_block_n, block_k,
                                                     major_a, major_b,
                                                     ab_dtype, cd_dtype,
                                                     num_stages, best_multicast_config);
        
        // If shared memory usage is within limits, use this number of stages
        if (best_smem_config.smem_size <= smem_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }
    DG_HOST_ASSERT(best_num_stages != 0);

    // ===== STEP 6: SM count optimization for L2 cache efficiency =====
    
    // Recompute minimal number of SMs required for optimal performance
    // Using fewer SMs can improve L2 cache hit rates and reduce GPU frequency throttling
    int num_min_sms = num_sms;
    if (ArchSpec::should_minimize_num_sms()) {
        // Calculate minimum SMs needed to maintain the optimal wave count
        num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, best_num_waves);
        
        // Ensure SM count is compatible with multicast requirements
        num_min_sms = align(num_min_sms, best_multicast_config.num_multicast);
        DG_HOST_ASSERT(num_min_sms <= num_sms);
    }

    // ===== STEP 7: Final configuration assembly =====
    
    const auto& config = GemmConfig {
        .gemm_type = gemm_type,
        .kernel_type = kernel_type,
        .ab_dtype = ab_dtype,
        .cd_dtype = cd_dtype,
        .major_a = major_a,
        .major_b = major_b,
        .with_accumulation = with_accumulation,
        .block_m = best_block_m,
        .block_n = best_block_n,
        .block_k = block_k,
        .num_stages = best_num_stages,
        // Handle case where K dimension doesn't divide evenly into stages
        .num_last_stages = ceil_div(k, block_k) % best_num_stages,
        .num_sms = num_min_sms,
        .tc_util = device_runtime->get_tc_util(),
        .multicast_config = best_multicast_config,
        .smem_config = best_smem_config,
        .thread_config = ArchSpec::get_thread_config(kernel_type, best_block_m, best_block_n)
    };

    // Validate tensor core utilization control (SM100 BF16 only)
    if (config.tc_util < 100)
        DG_HOST_ASSERT(device_runtime->get_arch_major() == 10 and ab_dtype == torch::kBFloat16);

    // ===== STEP 8: Debug output =====
    
    // Print configuration details for debugging (first occurrence only)
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        auto key = std::make_tuple(gemm_type, kernel_type, m, n, k, num_groups, major_a, major_b,
                                   ab_dtype, cd_dtype, with_accumulation, num_sms);
        static std::set<decltype(key)> printed;
        if (printed.count(key) == 0) {
            printf("GEMM type: %d, kernel type: %d, M: %d, N: %d, K: %d, groups: %d, "
                   "A major: %d, B major: %d, AB dtype: %s, CD dtype: %s, accumulation: %d, "
                   "SM limit: %d -> block M: %d, block N: %d, block K: %d, stages: %d, last stages: %d, "
                   "SMs: %d, multicast: %d, multicast on A: %d, shared memory: %d bytes, swizzle A: %d, "
                   "swizzle B: %d, swizzle CD: %d, SMs: %d, threads: %d, TC util: %d%%\n",
                   static_cast<int>(gemm_type), static_cast<int>(kernel_type), m, n, k, num_groups,
                   static_cast<int>(major_a), static_cast<int>(major_b), c10::toString(ab_dtype), c10::toString(cd_dtype),
                   static_cast<int>(with_accumulation), num_sms, best_block_m, best_block_n, block_k,
                   best_num_stages, config.num_last_stages, num_min_sms, best_multicast_config.num_multicast,
                   static_cast<int>(best_multicast_config.is_multicast_on_a),
                   best_smem_config.smem_size, best_smem_config.swizzle_a_mode, best_smem_config.swizzle_b_mode,
                   best_smem_config.swizzle_cd_mode, config.num_sms, config.thread_config.num_threads, config.tc_util);
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm