# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepGEMM is a high-performance CUDA library for General Matrix Multiplications (GEMMs), specializing in FP8 and BF16 operations for both normal and Mix-of-Experts (MoE) grouped scenarios. It uses Just-In-Time (JIT) compilation for optimal performance without kernel pre-compilation during installation.

## Development Commands

### Build and Setup
```bash
# Clone with submodules (required for CUTLASS and fmt dependencies)
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git

# Development build (links includes and builds CPP JIT module)
./develop.sh

# Full installation
./install.sh
```

### Running Tests
```bash
# Test all implementations
python tests/test_layout.py
python tests/test_attention.py
python tests/test_bf16.py
python tests/test_fp8.py
python tests/test_lazy_init.py

# Run specific test
python tests/test_fp8.py -k "test_fp8_gemm_nt"
```

## Architecture Overview

### Core Components

1. **C++ APIs** (`csrc/apis/`)
   - `gemm.hpp`: Main GEMM operations (FP8/BF16, grouped/non-grouped)
   - `attention.hpp`: MQA kernels for the indexer
   - `layout.hpp`: Tensor layout transformation utilities
   - `runtime.hpp`: Device runtime management

2. **JIT System** (`csrc/jit/`)
   - `compiler.hpp`: Runtime kernel compilation using NVCC
   - `kernel_runtime.hpp`: Kernel execution management
   - `cache.hpp`: Compiled kernel caching

3. **Kernel Implementations** (`csrc/jit_kernels/impls/`)
   - Architecture-specific kernels (SM90/SM100)
   - Different precisions (FP8 1D1D/1D2D, BF16)
   - Grouped and masked variants

4. **Python Interface** (`deep_gemm/`)
   - Python bindings via pybind11
   - High-level APIs with documentation

### Key Design Principles

- **JIT Compilation**: Kernels compiled at runtime for optimal performance
- **Architecture Support**: SM90 (Hopper) and SM100 (Blackwell) GPUs
- **Layout Requirements**: 
  - SM90: NT memory layout only (row-major A, col-major B)
  - SM100: All layouts supported (NT, TN, NN, TT)
  - Scaling factors require TMA-aligned transposed layout

### Environment Variables

- `DG_JIT_DEBUG`: Enable JIT debugging output
- `DG_JIT_CACHE_DIR`: Kernel cache directory (default: `$HOME/.deep_gemm`)
- `DG_JIT_USE_NVRTC`: Use NVRTC instead of NVCC
- `DG_JIT_NVCC_COMPILER`: Custom NVCC path
- `DG_PRINT_CONFIGS`: Print selected kernel configs

### GEMM Naming Convention

Operations follow the pattern `D = C + A @ B`:
- `fp8_gemm_nt`: Non-transposed A, transposed B
- `fp8_gemm_nn`: Non-transposed A, non-transposed B  
- `fp8_gemm_tn`: Transposed A, non-transposed B
- `fp8_gemm_tt`: Transposed A, transposed B

### Testing Strategy

Tests in `tests/` directory validate:
- Numerical correctness against reference implementations
- All GEMM variants and layouts
- Grouped operations (contiguous and masked)
- Edge cases and alignment requirements