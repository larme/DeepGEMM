# DeepGEMM Study Guide

This document provides a comprehensive study guide for understanding the DeepGEMM codebase, a high-performance CUDA library for optimized matrix multiplications.

## 1. Architecture Overview

### 1.1 Core Design Philosophy
- **JIT Compilation**: Kernels are compiled at runtime using NVCC or NVRTC
- **Architecture-Specific**: Separate implementations for SM90 (Hopper) and SM100 (Blackwell)
- **Type-Specific**: Optimized for FP8 and BF16 operations
- **Layout Flexibility**: SM90 supports NT layout only, SM100 supports all layouts

### 1.2 Directory Structure
```
DeepGEMM/
├── csrc/                    # C++ source code
│   ├── apis/               # High-level API definitions
│   ├── jit/                # JIT compilation system
│   ├── jit_kernels/        # Kernel implementations
│   └── utils/              # Helper utilities
├── deep_gemm/              # Python package
│   ├── include/            # CUDA kernel headers
│   └── utils/              # Python utilities
└── tests/                  # Test implementations
```

## 2. Key Components Analysis

### 2.1 JIT Compilation System

#### Compiler Infrastructure (`csrc/jit/compiler.hpp`)
- **Base Compiler Class**: Abstract interface for compilation
- **NVCCCompiler**: Uses NVIDIA's nvcc compiler
  - Requires NVCC >= 12.3, recommends 12.9+
  - Supports architecture-specific optimizations
- **NVRTCCompiler**: Uses NVIDIA's runtime compilation
  - Faster compilation but may have lower performance
  - Supports PCH (precompiled headers) for NVRTC 12.8+

**Key Features:**
- Kernel caching in `~/.deep_gemm/cache/`
- Signature-based cache invalidation
- Atomic file operations for thread safety

#### Kernel Runtime (`csrc/jit/kernel_runtime.hpp`)
- **KernelRuntime**: Manages compiled kernel lifecycle
- **LaunchRuntime**: Template class for kernel launching
- Uses `cuobjdump` to extract kernel symbols
- Configures launch parameters (grid, blocks, shared memory)

### 2.2 API Layer

#### GEMM Operations (`csrc/apis/gemm.hpp`)
Core GEMM functions follow the naming pattern:
- `fp8_gemm_XY`: X=A transpose, Y=B transpose (nt, nn, tn, tt)
- `m_grouped_*`: Group operations on M dimension
- `k_grouped_*`: Group operations on K dimension

**Key APIs:**
1. **Normal GEMMs**: `fp8_gemm_nt`, `bf16_gemm_nt`, etc.
2. **Grouped GEMMs**: 
   - Contiguous: All groups packed together
   - Masked: Sparse groups with validity mask
3. **cuBLAS Integration**: Fallback for comparison

#### Layout Management (`csrc/apis/layout.hpp`)
- Handles tensor layout transformations
- Scaling factor management (FP32 for SM90, UE8M0 for SM100)
- TMA (Tensor Memory Accelerator) alignment requirements

### 2.3 Kernel Implementations

#### Architecture-Specific Kernels
Located in `csrc/jit_kernels/impls/`:
- **SM90 Kernels**: `sm90_fp8_gemm_1d1d.hpp`, `sm90_fp8_gemm_1d2d.hpp`
- **SM100 Kernels**: `sm100_fp8_gemm_1d1d.hpp`, `sm100_fp8_gemm_1d2d.hpp`

**Kernel Types:**
- **1D1D**: Single-dimensional scaling factors
- **1D2D**: Two-dimensional scaling factors
- **NoSF**: No scaling factors (BF16 operations)

#### Heuristics (`csrc/jit_kernels/heuristics/`)
- Architecture-specific configuration selection
- Optimizes for block sizes, stages, thread counts
- Considers multicast capabilities

### 2.4 Python Interface

#### Module Initialization (`deep_gemm/__init__.py`)
- Finds CUDA installation
- Initializes C++ runtime
- Exports all kernel functions
- Version management

#### Utilities (`deep_gemm/utils/`)
- **layout.py**: Python wrappers for layout transformations
- **math.py**: Mathematical utilities
- Testing and benchmarking tools

## 3. Key Concepts

### 3.1 Tensor Layouts
- **Major Types**: K-major vs MN-major
- **TMA Requirements**: Specific alignment for tensor memory access
- **Scaling Factors**: Different formats for SM90 (FP32) vs SM100 (UE8M0)

### 3.2 Grouped Operations
- **M-Grouped**: Variable tokens per expert (MoE forward)
- **K-Grouped**: Variable dimensions per expert (MoE backward)
- **Contiguous vs Masked**: Dense vs sparse group representation

### 3.3 Performance Optimization
- **Multicast**: Efficient data distribution across SMs
- **Swizzling**: Memory access pattern optimization
- **Cluster Scheduling**: SM90/SM100 cluster features

### 3.4 Two-Layer Implementation Architecture

DeepGEMM uses a sophisticated two-layer architecture that separates GPU kernel implementations from host-side runtime management. Understanding this separation is crucial for navigating the codebase.

#### Layer 1: GPU Kernel Templates (`deep_gemm/include/deep_gemm/impls/`)
**Purpose**: Contains the actual CUDA kernel implementations
- **File Type**: `.cuh` files (CUDA headers)
- **Content**: Hand-written, heavily templated GPU kernels
- **Role**: The "what" - actual matrix multiplication algorithms
- **Persistence**: Permanent files checked into the repository

**Example Structure**:
```cuda
// In deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh
template<int M, int N, int K, int BlockM, int BlockN, int BlockK, int Stages, ...>
__global__ __launch_bounds__(NumThreads, 1) void
sm90_fp8_gemm_1d1d_impl(/* kernel parameters */) {
    // Actual GPU computation using CuTe, tensor cores, etc.
    // This is the performance-critical code
}
```

#### Layer 2: Host Runtime Wrappers (`csrc/jit_kernels/impls/`)
**Purpose**: Orchestrates JIT compilation and kernel instantiation
- **File Type**: `.hpp` files (C++ headers)
- **Content**: Runtime management classes that inherit from LaunchRuntime
- **Role**: The "how" - when and with what parameters to instantiate kernels
- **Function**: Bridges Python API to GPU kernels

**Example Structure**:
```cpp
// In csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp
class SM90FP8Gemm1D1DRuntime final: public LaunchRuntime<SM90FP8Gemm1D1DRuntime> {
    static std::string generate_impl(const Args& args) {
        // Generates C++ code string that includes the .cuh file
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>
static void __instantiate_kernel() {{
    auto ptr = &sm90_fp8_gemm_1d1d_impl<{}, {}, {}, {}, {}, ...>;
}})", args.m, args.n, args.k, args.block_m, args.block_n, ...);
    }
    
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // Launches the compiled kernel with proper parameters
    }
};
```

#### The JIT Compilation Flow

1. **API Call**: User calls `deep_gemm.fp8_gemm_nt(a, b, d)`
2. **Parameter Analysis**: Runtime wrapper analyzes tensor shapes and selects optimal configuration
3. **Code Generation**: Runtime generates instantiation code:
   ```cpp
   #include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>
   auto ptr = &sm90_fp8_gemm_1d1d_impl<4096, 4096, 2048, 128, 128, 64, 3, ...>;
   ```
4. **JIT Compilation**: NVCC/NVRTC compiles this specific template instantiation
5. **Caching**: Compiled kernel is cached based on signature (dimensions + configuration)
6. **Execution**: Kernel is launched with runtime-computed parameters

#### Key Benefits of This Architecture

**Separation of Concerns**:
- GPU experts can optimize kernel algorithms without worrying about host integration
- Runtime engineers can improve parameter selection without touching GPU code
- Python interface developers work with clean APIs

**Performance Optimization**:
- Templates allow compile-time optimization for specific problem sizes
- JIT compilation eliminates runtime overhead of generic kernels
- Caching prevents recompilation of identical configurations

**Maintainability**:
- GPU kernels are reusable across different host interfaces
- Changes to runtime logic don't require rewriting GPU code
- Clear boundaries make debugging easier

**Flexibility**:
- Same GPU kernel can be instantiated with different parameters
- Easy to add new runtime strategies without changing GPU code
- Support for architecture-specific optimizations

#### Common Misunderstanding

❌ **Incorrect**: "csrc/jit_kernels/impls/ are templates that generate deep_gemm/include/deep_gemm/impls/"

✅ **Correct**: "deep_gemm/include/deep_gemm/impls/ contains templated GPU kernels, and csrc/jit_kernels/impls/ generates code strings that instantiate those templates with specific parameters"

The GPU kernels are permanent, hand-optimized code. The runtime wrappers dynamically generate the "glue code" that connects Python calls to specific GPU kernel instantiations.

## 4. Usage Examples

### 4.1 Basic FP8 GEMM
```python
import deep_gemm

# D = A @ B.T (with scaling factors)
deep_gemm.fp8_gemm_nt(
    a=(a_tensor, a_scale),
    b=(b_tensor, b_scale),
    d=output_tensor
)
```

### 4.2 Grouped GEMM
```python
# MoE forward pass
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    a=(tokens, token_scales),
    b=(expert_weights, weight_scales),
    d=output,
    m_indices=token_to_expert_mapping
)
```

### 4.3 Configuration Tuning
```python
# Set SM count for better occupancy
deep_gemm.set_num_sms(108)

# Set tensor core utilization
deep_gemm.set_tc_util(90)
```

## 5. Testing Strategy

### Test Structure (`tests/`)
- **test_fp8.py**: FP8 GEMM correctness and performance
- **test_bf16.py**: BF16 operations
- **test_attention.py**: MQA kernels
- **test_layout.py**: Layout transformations

### Key Test Patterns:
1. Generate random inputs with proper alignment
2. Compute reference with PyTorch/cuBLAS
3. Compare outputs within tolerance
4. Benchmark against baselines

## 6. Environment Variables

### JIT Configuration
- `DG_JIT_DEBUG`: Enable verbose JIT output
- `DG_JIT_CACHE_DIR`: Custom cache location
- `DG_JIT_USE_NVRTC`: Use NVRTC instead of NVCC
- `DG_JIT_NVCC_COMPILER`: Custom NVCC path

### Runtime Configuration
- `DG_PRINT_CONFIGS`: Print selected kernel configs
- `DG_JIT_PTXAS_VERBOSE`: Show PTXAS output

## 7. Advanced Topics

### 7.1 Kernel Generation
Kernels are generated from templates with compile-time constants:
- Problem dimensions (M, N, K)
- Block sizes and configurations
- Architecture-specific features

### 7.2 Cache Management
- Signature includes: kernel name, library version, compiler, flags, code
- Atomic file operations prevent race conditions
- Automatic invalidation on code changes

### 7.3 Error Handling
- Host assertions for API validation
- CUDA error checking macros
- Detailed error messages with context

## 8. Performance Considerations

### 8.1 Optimization Tips
- Use aligned dimensions (check `get_mk_alignment_for_contiguous_layout`)
- Prefer contiguous memory layouts
- Match architecture capabilities (SM90 vs SM100)

### 8.2 Benchmarking
- Use `deep_gemm.testing.bench_kineto` for accurate timing
- Compare against cuBLAS baselines
- Monitor memory bandwidth utilization

## 9. Future Development

### Planned Features (from README roadmap)
- Split/stream-k optimizations
- Ampere architecture support
- CUDA PDL integration
- Larger TMA multicast sizes

### Extension Points
- New data types in `common/types.hpp`
- Architecture variants in heuristics
- Custom kernel implementations

## Study Recommendations

1. **Start with**: Python API and test examples
2. **Deep dive into**: JIT compilation flow
3. **Understand**: Layout requirements and transformations
4. **Experiment with**: Different configurations and benchmarks
5. **Extend**: Add new kernel variants or optimizations

## 10. Suggested Code Reading Order

### Phase 1: High-Level Understanding (Python Interface)
Start with the Python layer to understand the API and usage patterns:

1. **`deep_gemm/__init__.py`** - Entry point, see exported functions and initialization
2. **`tests/test_fp8.py`** - Real usage examples, understand API patterns
3. **`tests/generators.py`** - Test data generation, understand tensor requirements
4. **`deep_gemm/utils/layout.py`** - Python-side layout utilities

### Phase 2: Core API Layer (C++ Interface)
Understand the bridge between Python and CUDA:

5. **`csrc/python_api.cpp`** - Python bindings, see how APIs are exposed
6. **`csrc/apis/gemm.hpp`** - Core GEMM operations and dispatch logic
7. **`csrc/apis/layout.hpp`** - Layout transformation APIs
8. **`csrc/apis/runtime.hpp`** - Runtime configuration APIs

### Phase 3: JIT Compilation System
Understand how kernels are compiled and managed:

9. **`csrc/jit/device_runtime.hpp`** - Device capability detection
10. **`csrc/jit/compiler.hpp`** - Compilation infrastructure (NVCC/NVRTC)
11. **`csrc/jit/kernel_runtime.hpp`** - Kernel loading and execution
12. **`csrc/jit/cache.hpp`** - Kernel caching mechanism

### Phase 4: Kernel Architecture
Understand kernel selection and configuration:

13. **`deep_gemm/include/deep_gemm/common/types.hpp`** - Core type definitions
14. **`csrc/jit_kernels/heuristics/common.hpp`** - Configuration structures
15. **`csrc/jit_kernels/heuristics/sm90.hpp`** - SM90-specific heuristics
16. **`csrc/jit_kernels/heuristics/sm100.hpp`** - SM100-specific heuristics

### Phase 5: Kernel Implementation
Study actual kernel implementations (start with one architecture):

17. **`csrc/jit_kernels/impls/runtime_utils.hpp`** - Common runtime utilities
18. **`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp`** - Simple FP8 kernel wrapper
19. **`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`** - Actual CUDA kernel
20. **`deep_gemm/include/deep_gemm/common/utils.cuh`** - CUDA utilities

### Phase 6: Advanced Features (Optional)
For deeper understanding of specific features:

21. **`csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp`** - Grouped GEMM implementations
22. **`csrc/apis/attention.hpp`** - MQA kernels for attention
23. **`csrc/apis/einsum.hpp`** - Einstein summation operations
24. **`csrc/utils/`** - Various utility headers (exception, hash, format, etc.)

### Phase 7: Architecture Comparison (Optional)
Compare implementations across architectures:

25. **SM90 vs SM100 implementations** - Compare corresponding files
26. **FP8 vs BF16 implementations** - Understand precision differences
27. **1D1D vs 1D2D kernels** - Scaling factor dimensionality

### Reading Tips:

1. **Keep test files open**: Reference `tests/test_*.py` while reading C++ code to understand usage
2. **Trace a single operation**: Follow one API call (e.g., `fp8_gemm_nt`) from Python to CUDA
3. **Use IDE features**: Jump to definitions to understand type relationships
4. **Run examples**: Modify and run tests to see behavior changes
5. **Check environment variables**: See how `DG_JIT_DEBUG=1` changes output

### Quick Navigation Guide:

- **Want to use the library?** Start with Phase 1
- **Want to understand architecture?** Focus on Phases 2-3
- **Want to optimize/extend?** Deep dive into Phases 4-6
- **Want to port to new hardware?** Study Phase 7

This progressive reading order moves from user-facing APIs to low-level implementation details, allowing you to build understanding layer by layer.

This codebase demonstrates modern CUDA programming patterns with JIT compilation, making it an excellent resource for learning GPU optimization techniques.