#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.hpp>

#include "common.hpp"

namespace deep_gemm {

/**
 * @brief Architecture-specific optimizations for SM90 (Hopper) GPUs
 * 
 * The SM90ArchSpec class encapsulates all Hopper-specific constraints, optimizations,
 * and hardware characteristics. This includes shared memory limits, block size restrictions,
 * multicast capabilities, and architecture-specific performance tuning.
 * 
 * Key SM90 Features:
 * - 232KB shared memory per SM (vs 164KB in SM80)
 * - TMA (Tensor Memory Accelerator) for direct global-to-shared memory transfers
 * - WGMMA (Warp Group Matrix Multiply-Accumulate) for tensor core operations
 * - Advanced thread block clusters and distributed shared memory
 */
struct SM90ArchSpec {
    /**
     * @brief Shared memory capacity per SM in bytes
     * 
     * SM90 has 232KB of shared memory per SM, significantly more than previous generations.
     * This allows for deeper pipelining (more stages) and larger block sizes, but requires
     * careful management to avoid exceeding limits.
     */
    static constexpr int smem_capacity = 232448;

    /**
     * @brief Generate N block size candidates based on output data type
     * 
     * Block size selection must consider bank conflict avoidance and memory coalescing.
     * Different data types have different alignment requirements:
     * - FP32: Start at 8 to avoid bank conflicts (32 banks × 4 bytes = 128B alignment)
     * - BF16/FP16: Start at 16 for optimal memory coalescing
     * 
     * @param cd_dtype Output matrix data type (BF16 or FP32)
     * @return Vector of valid N block sizes in increments of 16
     */
    static std::vector<int> get_block_n_candidates(const at::ScalarType& cd_dtype) {
        // FP32 output requires careful bank conflict avoidance
        // Start at 8 for FP32 (bank conflict considerations), 16 for others
        const auto& start = cd_dtype == torch::kFloat ? 8 : 16;
        std::vector<int> candidates;
        
        // Generate candidates in multiples of 16 for optimal memory coalescing
        // 16-element alignment ensures warp-level coalesced access patterns
        for (int i = start; i <= 256; i += 16)
            candidates.push_back(i);
        return candidates;
    }

    /**
     * @brief Calculate effective M block size for matrix A loading
     * 
     * In SM90, multicast doesn't affect the M dimension loading pattern for matrix A.
     * Each CTA loads its own M block portion regardless of multicast configuration.
     * 
     * @param multicast_config TMA multicast settings (unused for M dimension)
     * @param block_m Original M block size
     * @return Effective M block size for loading
     */
    static int get_ab_load_block_m(const MulticastConfig& multicast_config, const int& block_m) {
        return block_m;  // No change for SM90
    }

    /**
     * @brief Calculate effective N block size for matrix B loading
     * 
     * Similar to M dimension, N dimension loading is not affected by multicast in SM90.
     * Each CTA handles its own N block portion.
     * 
     * @param multicast_config TMA multicast settings (unused for N dimension)
     * @param block_n Original N block size  
     * @return Effective N block size for loading
     */
    static int get_ab_load_block_n(const MulticastConfig& multicast_config, const int& block_n) {
        return block_n;  // No change for SM90
    }

    /**
     * @brief Calculate effective M block size for output matrix C/D storage
     * 
     * SM90 uses WGMMA (Warp Group Matrix Multiply-Accumulate) which operates on 64×64 tiles.
     * When single warpgroup synchronization is enabled, we limit to WGMMA tile size.
     * 
     * @param block_m Original M block size
     * @param single_warpgroup_sync Whether to use single warpgroup optimization
     * @return Effective M block size for storage operations
     */
    static int get_cd_store_block_m(const int& block_m, const bool& single_warpgroup_sync = false) {
        constexpr int wgmma_m = 64;  // WGMMA M dimension size
        return single_warpgroup_sync ? wgmma_m : block_m;
    }

    /**
     * @brief Calculate effective N block size for output matrix C/D storage
     * 
     * N dimension storage is not affected by warpgroup optimizations in SM90.
     * 
     * @param block_n Original N block size
     * @return Effective N block size for storage
     */
    static int get_cd_store_block_n(const int& block_n) {
        return block_n;  // No change needed
    }

    /**
     * @brief Determine whether to enable swizzling for output matrix C/D
     * 
     * Swizzling helps avoid bank conflicts in shared memory. FP32 has specific
     * bank conflict patterns that are handled differently than FP16/BF16.
     * 
     * @param cd_dtype Output matrix data type
     * @return true if swizzling should be enabled
     */
    static bool enable_cd_swizzle(const at::ScalarType& cd_dtype) {
        // Disable swizzling for FP32 due to different bank conflict patterns
        return cd_dtype != torch::kFloat;
    }

    /**
     * @brief Validate block size combination for SM90 architecture constraints
     * 
     * SM90 has several constraints that limit valid block size combinations:
     * 1. Register pressure: Large blocks require too many registers
     * 2. Shared memory: Must fit within 232KB limit
     * 3. Bank conflicts: Certain patterns cause performance degradation
     * 4. Scaling factors: FP8 operations have additional memory requirements
     * 
     * @param kernel_type Type of kernel being used
     * @param major_a,major_b Matrix layouts
     * @param ab_dtype,cd_dtype Input and output data types
     * @param block_m,block_n,block_k Block dimensions
     * @return true if block size combination is valid
     */
    static bool is_block_size_legal(const KernelType& kernel_type,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                    const int& block_m, const int& block_n, const int& block_k) {
        
        // ===== FP32 Output Constraints =====
        
        // SM90 FP32 output does not support large M blocks due to register pressure
        if (cd_dtype == at::kFloat and block_m == 256)
            return false;

        // Large C/D shared memory usage with FP32 limits pipeline depth
        // Ensure minimum pipeline stages: 4 for 1D1D kernels, 3 for NoSF kernels
        if (block_n > 128 and cd_dtype == torch::kFloat) {
            if (kernel_type == KernelType::Kernel1D1D and block_n > 152)
                return false;
            if (kernel_type == KernelType::KernelNoSF and block_n > 200)
                return false;
        }

        // ===== Scaling Factor Constraints =====
        
        // Complex scaling factor patterns can cause register spills or excessive memory usage
        // Condition: block_n > block_k and gcd(block_n, block_k) != block_n - block_k
        // This indicates non-uniform scaling factor access patterns
        if (block_n > 128 and kernel_type == KernelType::Kernel1D2D and 
           (block_n != 144 and block_n != 160 and block_n != 192))
            return false;

        // ===== Bank Conflict Avoidance =====
        
        // FP32 output with N dimensions divisible by 16 causes bank conflicts
        // This is due to the specific access patterns in FP32 epilogue operations
        if (cd_dtype == torch::kFloat and block_n % 16 == 0)
            return false;

        // ===== Register Pressure Constraint =====
        
        // Large block sizes in both dimensions require too many registers
        // At least one dimension must be ≤ 128 to ensure sufficient registers
        return block_m <= 128 or block_n <= 128;
    }

    /**
     * @brief Validate pipeline stage count for given configuration
     * 
     * Certain combinations of problem parameters can cause excessive code generation
     * when both pipeline stages and inner loop iterations are unrolled.
     * 
     * @param ab_dtype,cd_dtype Input and output data types
     * @param num_stages Number of pipeline stages
     * @param block_m,block_n,block_k Block dimensions
     * @return true if stage count is valid
     */
    static bool is_num_stages_legal(const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                    const int& num_stages,
                                    const int& block_m, const int& block_n, const int& block_k) {
        
        // For FP8 operations with specific block patterns, limit stages to prevent code bloat
        // Condition checks for patterns where both stage unrolling and inner loop unrolling occur
        if (ab_dtype == torch::kFloat8_e4m3fn and 
            block_k % block_n != 0 and 
            block_k / std::gcd(block_n, block_k) <= 4)
            return num_stages <= 4;
            
        return true;  // No restrictions for other cases
    }

    /**
     * @brief Determine whether to minimize SM usage for L2 cache optimization
     * 
     * SM90 benefits from using fewer SMs when possible to improve L2 cache hit rates
     * and reduce GPU frequency throttling. This is particularly effective for smaller
     * problems where not all SMs are needed.
     * 
     * @return true to enable SM count minimization
     */
    static bool should_minimize_num_sms() {
        return true;  // Always enable for SM90
    }

    /**
     * @brief Determine TMA multicast legality for both matrix dimensions
     * 
     * TMA multicast allows multiple CTAs to share data, reducing memory bandwidth.
     * However, it requires specific divisibility and alignment constraints.
     * 
     * SM90 TMA Multicast Constraints:
     * - Number of blocks must be divisible by multicast factor
     * - SM count must be compatible with multicast pattern
     * - Certain GEMM types have additional restrictions
     * 
     * @param gemm_type Type of GEMM operation
     * @param num_groups Number of expert groups (for MoE)
     * @param m,n Problem dimensions
     * @param block_m,block_n Block dimensions
     * @param num_sms Number of available SMs
     * @return {multicast_legal_on_A, multicast_legal_on_B}
     */
    static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type, const int& num_groups,
                                                        const int& m, const int& n, const int& block_m, const int& block_n,
                                                        const int& num_sms) {
        
        // Disable multicast for K-grouped GEMM with many groups
        // Large group counts create complex synchronization patterns
        if (gemm_type == GemmType::KGroupedContiguous and num_groups > 4)
            return {false, false};

        return {
            // Multicast on matrix A (along M dimension)
            is_multicast_legal(n, block_n, 2, num_sms, gemm_type == GemmType::MGroupedMasked),
            
            // Multicast on matrix B (along N dimension)
            // Masked GEMM requires additional divisibility on N for even block distribution
            is_multicast_legal(m, block_m, 2, num_sms, false)
                and (gemm_type != GemmType::MGroupedMasked or 
                     is_multicast_legal(n, block_n, 2, num_sms, true))
        };
    }

    /**
     * @brief Generate optimal thread configuration for SM90 architecture
     * 
     * SM90 uses a two-tier thread organization:
     * - TMA threads (128): Handle Tensor Memory Accelerator operations
     * - Math threads (128/256): Perform WGMMA tensor core computations
     * 
     * Thread count scales with block size to maintain optimal register usage.
     * 
     * @param kernel_type Type of kernel being used
     * @param block_m,block_n Block dimensions
     * @return Optimal thread configuration
     */
    static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                          const int& block_m, const int& block_n) {
        
        // TMA threads: Fixed at 128 for optimal TMA operation scheduling
        // Math threads: Scale based on M block size
        //   - 64×N blocks: 128 math threads (1 warpgroup)
        //   - Larger blocks: 256 math threads (2 warpgroups)
        return ThreadConfig::sm90(128, (block_m == 64 ? 1 : 2) * 128);
    }

    /**
     * @brief Calculate shared memory requirements for output matrix C/D
     * 
     * Output accumulation requires shared memory for partial results before
     * writing to global memory. The size depends on block dimensions and
     * epilogue pipeline requirements.
     * 
     * @param kernel_type Type of kernel being used
     * @param block_m,block_n Block dimensions
     * @param swizzle_cd_mode Swizzling pattern for C/D
     * @param cd_dtype Output data type
     * @return Shared memory size in bytes
     */
    static int get_smem_cd_size(const KernelType& kernel_type,
                                const int& block_m, const int& block_n,
                                const int& swizzle_cd_mode, const at::ScalarType& cd_dtype) {
        
        // Simple calculation: block size × element size
        // SM90 doesn't require additional padding or complex layouts
        return block_m * block_n * static_cast<int>(c10::elementSize(cd_dtype));
    }

    /**
     * @brief Calculate shared memory for scaling factors per pipeline stage
     * 
     * FP8 operations require scaling factors to convert between FP8 and higher precision.
     * Different kernel types have different scaling factor requirements.
     * 
     * SM90 Scaling Factor Requirements:
     * - SF_A: Per-row scaling factors for matrix A (always FP32)
     * - SF_B: Per-column scaling factors for matrix B (layout-dependent)
     * - BF16 operations don't need scaling factors
     * 
     * @param kernel_type Type of kernel being used
     * @param block_m,block_n,block_k Block dimensions
     * @param ab_dtype,cd_dtype Input and output data types
     * @return {SF_A_size_per_stage, SF_B_size_per_stage}
     */
    static std::pair<int, int> get_sf_smem_size_per_stage(const KernelType& kernel_type,
                                                          const int& block_m, const int& block_n, const int& block_k,
                                                          const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype) {
        
        // BF16 operations don't require scaling factors
        if (ab_dtype == torch::kBFloat16)
            return {0, 0};

        // SF_A: One FP32 scaling factor per M dimension (per row)
        int smem_sfa_per_stage = block_m * static_cast<int>(sizeof(float));
        
        int smem_sfb_per_stage = 0;
        if (kernel_type == KernelType::Kernel1D1D) {
            // SF_B: Per-column scaling factors with TMA alignment
            // 128-byte alignment required for optimal TMA performance
            smem_sfb_per_stage = align(block_n * 4, 128);
        }
        
        return {smem_sfa_per_stage, smem_sfb_per_stage};
    }

    /**
     * @brief Calculate additional scaling factor B shared memory requirements
     * 
     * Some kernel configurations require extra SF_B storage beyond per-stage buffers.
     * This handles edge cases where scaling factor access patterns don't align
     * with the main pipeline stages.
     * 
     * @param m,n,k Problem dimensions
     * @param block_m,block_n,block_k Block dimensions
     * @return Additional SF_B shared memory size in bytes
     */
    static int get_extra_sfb_smem_size(const int& m, const int& n, const int& k,
                                       const int& block_m, const int& block_n, const int& block_k) {
        
        // Determine if uniform SF_B access pattern is possible
        // If block_k divides block_n evenly, we can use uniform access (1x storage)
        // Otherwise, we need doubled storage (2x) for non-uniform patterns
        const auto& use_uniform_sfb = block_k % block_n == 0 ? 1 : 2;
        
        // Calculate total SF_B elements needed across all K iterations
        return align<int>(ceil_div(k, block_k) * static_cast<int>(sizeof(float)) * use_uniform_sfb, 8);
    }

    /**
     * @brief Calculate shared memory for synchronization barriers
     * 
     * SM90 uses mbarrier objects for synchronization between pipeline stages.
     * Each barrier requires 8 bytes, and we need barriers for both phases
     * of the pipeline (producer and consumer).
     * 
     * @param num_stages Number of pipeline stages
     * @return Barrier shared memory size in bytes
     */
    static int get_barrier_smem_size(const int& num_stages) {
        // Each stage needs 2 barriers (8 bytes each): producer and consumer
        return num_stages * 8 * 2;
    }

    /**
     * @brief Calculate shared memory for tensor memory pointers
     * 
     * SM90 doesn't require additional tensor memory pointer storage
     * beyond what's included in other allocations.
     * 
     * @return Tensor memory pointer storage size (0 for SM90)
     */
    static int get_tmem_ptr_smem_size() {
        return 0;  // Not needed for SM90
    }

    /**
     * @brief Calculate shared memory for TMA tensor map descriptors
     * 
     * TMA operations require tensor map descriptors stored in shared memory.
     * K-grouped GEMM operations need multiple tensor maps for different
     * expert weight matrices.
     * 
     * @param gemm_type Type of GEMM operation
     * @return Tensor map storage size in bytes
     */
    static int get_tensormap_smem_size(const GemmType& gemm_type) {
        // K-grouped contiguous GEMM needs 4 tensor maps
        // Other GEMM types don't require tensor map storage
        return gemm_type == GemmType::KGroupedContiguous ? 
               4 * static_cast<int>(sizeof(CUtensorMap)) : 0;
    }
};

} // namespace deep_gemm