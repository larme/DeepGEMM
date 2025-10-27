#pragma once

// Standard library headers for filesystem operations, I/O, and string manipulation
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>  // NVIDIA Runtime Compilation library for JIT compilation
#include <regex>
#include <string>

// Internal utility headers
#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"

namespace deep_gemm {

/**
 * Base Compiler class that provides the infrastructure for JIT (Just-In-Time) compilation
 * of CUDA kernels. This abstract base class defines the interface and common functionality
 * for different compilation backends (NVCC and NVRTC).
 * 
 * Key responsibilities:
 * 1. Manage compilation cache to avoid recompiling identical kernels
 * 2. Generate unique signatures for kernel caching
 * 3. Provide atomic file operations for thread-safe cache management
 * 4. Define the compilation interface for derived classes
 */
class Compiler {
public:
    // Static member variables that store global paths and version information
    // These are initialized once during library initialization via prepare_init()
    static std::filesystem::path library_root_path;      // Root directory of DeepGEMM library
    static std::filesystem::path library_include_path;   // Include directory for CUDA headers
    static std::filesystem::path cuda_home;              // CUDA installation directory
    static std::string library_version;                  // Hash of all library headers (for cache invalidation)

    /**
     * Computes a version hash of all header files in the library.
     * This hash is used to invalidate the kernel cache when the library code changes.
     * 
     * Process:
     * 1. Recursively collect all files in the include/deep_gemm directory
     * 2. Read all file contents into a single stream
     * 3. Compute a hex digest (hash) of the concatenated content
     * 
     * This ensures that any change to kernel headers triggers recompilation.
     */
    static std::string get_library_version() {
        std::stringstream ss;
        // collect_files() recursively finds all files in the directory
        for (const auto& f: collect_files(library_include_path / "deep_gemm")) {
            std::ifstream in(f, std::ios::binary);
            ss << in.rdbuf();  // Read entire file content
        }
        return get_hex_digest(ss.str());  // Hash the concatenated content
    }

    /**
     * Static initialization function called once when the library is loaded.
     * Must be called before any Compiler instances are created.
     * 
     * @param library_root_path: Path to DeepGEMM installation
     * @param cuda_home_path_by_python: CUDA installation path found by Python
     */
    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::library_version = get_library_version();
    }

    // Instance variables
    std::string signature;              // Compiler-specific signature (e.g., "NVCC12.9")
    std::string flags;                  // Compilation flags common to all kernels
    std::filesystem::path cache_dir_path;  // Root directory for kernel cache

    /**
     * Base constructor that initializes common compiler settings.
     * Sets up:
     * 1. Cache directory location (default: ~/.deep_gemm)
     * 2. Common compilation flags (C++ standard, diagnostics suppression, etc.)
     * 3. Debug and profiling options based on environment variables
     */
    Compiler() {
        // Verify that prepare_init() was called
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not cuda_home.empty());
        DG_HOST_ASSERT(not library_version.empty());

        // Set up cache directory
        // Default: $HOME/.deep_gemm, can be overridden by DG_JIT_CACHE_DIR
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto& env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // Initialize compiler signature and flags
        signature = "unknown-compiler";  // Will be overridden by derived classes
        
        // Base compilation flags:
        // - C++ standard (default C++20)
        // - Suppress various harmless warnings
        // - Set PTX register usage level for better occupancy
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_JIT_CPP_STANDARD", 20));
        
        // Add debug flags if requested
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            flags += " --ptxas-options=--verbose";  // Show PTX assembly details
        
        // Add line info for profiling/debugging
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
    }

    virtual ~Compiler() = default;

    /**
     * Creates a temporary directory for compilation artifacts.
     * Ensures the directory exists, creating it if necessary.
     */
    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    /**
     * Generates a unique temporary file path using UUID.
     * Used for atomic file operations to prevent race conditions.
     */
    std::filesystem::path get_tmp_file_path() const {
        return make_tmp_dir() / get_uuid();
    }

    /**
     * Atomically writes data to a file.
     * Process:
     * 1. Write to a temporary file with unique name
     * 2. Use filesystem::rename for atomic replacement
     * 
     * This prevents partial writes and race conditions in multi-process scenarios.
     */
    void put(const std::filesystem::path& path, const std::string& data) const {
        const auto tmp_file_path = get_tmp_file_path();

        // Write to temporary file
        std::ofstream out(tmp_file_path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // Atomic rename (POSIX guarantees atomicity)
        std::filesystem::rename(tmp_file_path, path);
    }

    /**
     * Main entry point for kernel compilation.
     * Implements caching logic to avoid redundant compilations.
     * 
     * @param name: Kernel name (e.g., "sm90_fp8_gemm_1d1d")
     * @param code: Complete CUDA source code
     * @return: Shared pointer to compiled kernel runtime
     * 
     * Cache key components:
     * 1. Kernel name
     * 2. Library version (invalidates on header changes)
     * 3. Compiler signature (NVCC version, etc.)
     * 4. Compilation flags
     * 5. Complete source code
     */
    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        // Create unique signature by concatenating all factors that affect compilation
        const auto kernel_signature = fmt::format("{}$${}$${}$${}$${}", 
            name,             // Kernel name
            library_version,  // Library headers hash
            signature,        // Compiler version
            flags,           // Compilation flags
            code);           // Source code
        
        // Generate cache directory path: cache/kernel.<name>.<hash>
        const auto dir_path = cache_dir_path / "cache" / 
            fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Check if kernel is already in runtime cache (fast path)
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Create cache directory if it doesn't exist
        make_dirs(dir_path);

        // Compile to temporary location first
        const auto tmp_cubin_path = get_tmp_file_path();
        compile(code, dir_path, tmp_cubin_path);  // Virtual function call

        // Atomically move compiled binary to cache
        make_dirs(dir_path);  // Ensure directory still exists
        std::filesystem::rename(tmp_cubin_path, dir_path / "kernel.cubin");

        // Load the compiled kernel into runtime cache
        const auto& runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    /**
     * Pure virtual function that derived classes must implement.
     * Performs the actual compilation using specific backend (NVCC/NVRTC).
     * 
     * @param code: CUDA source code to compile
     * @param dir_path: Cache directory for this kernel
     * @param cubin_path: Output path for compiled CUBIN file
     */
    virtual void compile(const std::string &code, 
                        const std::filesystem::path& dir_path, 
                        const std::filesystem::path &cubin_path) const = 0;
};

// Macro declarations for static member variables (defined in .cpp file)
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_version);

/**
 * NVCC-based compiler implementation.
 * Uses the traditional NVIDIA CUDA Compiler (nvcc) for kernel compilation.
 * 
 * Advantages:
 * - More mature and stable
 * - Better optimization in some cases
 * - Familiar compilation model
 * 
 * Disadvantages:
 * - Slower compilation times
 * - Requires nvcc binary in system
 */
class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;  // Path to nvcc executable

    /**
     * Extracts and validates NVCC version.
     * DeepGEMM requires NVCC >= 12.3, recommends >= 12.9 for best performance.
     * 
     * @return: Pair of (major, minor) version numbers
     */
    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Run "nvcc --version" command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // Parse version from output using regex
        // Example output: "release 12.3, V12.3.107"
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        
        // Version requirements
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    /**
     * NVCC compiler constructor.
     * Sets up NVCC-specific paths and compilation flags.
     */
    NVCCCompiler() {
        // Find nvcc executable
        nvcc_path = cuda_home / "bin" / "nvcc";
        // Allow override via environment variable
        if (const auto& env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        
        // Get version and create signature
        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // Build NVCC-specific flags
        // Note: Only NVCC >= 12.9 supports architecture family suffixes (e.g., sm_90a)
        const auto& arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags,                          // Base flags from parent
                            library_include_path.c_str(),   // Include path for headers
                            arch);                          // GPU architecture
        
        // Additional flag explanations:
        // -cubin: Generate CUBIN (CUDA binary) instead of object file
        // -O3: Maximum optimization level
        // --expt-relaxed-constexpr: Allow more constexpr usage
        // --expt-extended-lambda: Enable advanced lambda features
    }

    /**
     * NVCC compilation implementation.
     * Writes source to file and invokes nvcc as external process.
     */
    void compile(const std::string &code, 
                const std::filesystem::path& dir_path, 
                const std::filesystem::path &cubin_path) const override {
        // Write source code to cache directory for debugging
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Build and execute nvcc command
        const auto& command = fmt::format("{} {} -o {} {}", 
            nvcc_path.c_str(),   // nvcc executable
            code_path.c_str(),   // Input .cu file
            cubin_path.c_str(),  // Output .cubin file
            flags);              // Compilation flags
        
        // Print command if debugging
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        
        // Execute compilation
        const auto& [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Print PTX assembler output if requested
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

/**
 * NVRTC-based compiler implementation.
 * Uses NVIDIA Runtime Compilation (NVRTC) for in-process JIT compilation.
 * 
 * Advantages:
 * - Much faster compilation (no process spawn)
 * - No dependency on nvcc binary
 * - Better integration with runtime
 * 
 * Disadvantages:
 * - May produce slightly less optimized code in some cases
 * - Newer technology, less battle-tested
 */
class NVRTCCompiler final: public Compiler {
public:
    /**
     * NVRTC compiler constructor.
     * Sets up NVRTC-specific compilation flags and features.
     */
    NVRTCCompiler() {
        // Get NVRTC version
        int major, minor;
        DG_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

        // Build include directories
        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());

        // PCH (Precompiled Headers) support for NVRTC 12.8+
        // This significantly speeds up compilation by pre-parsing common headers
        std::string pch_flags;
        if (major > 12 or minor >= 8) {
            pch_flags = "--pch ";
            if (get_env<int>("DG_JIT_DEBUG", 0))
                pch_flags += "--pch-verbose=true ";
        }

        // Build NVRTC flags
        const auto& arch = device_runtime->get_arch(false, major > 12 or minor >= 9);
        flags = fmt::format("{} {}--gpu-architecture=sm_{} -default-device {}",
                            flags,         // Base flags
                            include_dirs,  // Include paths
                            arch,          // GPU architecture
                            pch_flags);    // PCH options
    }

    /**
     * NVRTC compilation implementation.
     * Compiles code in-process using NVRTC API.
     */
    void compile(const std::string &code, 
                const std::filesystem::path& dir_path, 
                const std::filesystem::path &cubin_path) const override {
        // Save source for debugging (optional)
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Parse flags string into individual options
        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option)
            options.push_back(option);

        // Convert to C-style array for NVRTC API
        std::vector<const char*> option_cstrs;
        for (const auto& opt: options)
            option_cstrs.push_back(opt.c_str());

        // Debug output
        if (get_env<int>("DG_JIT_DEBUG", 0) or get_env<int>("DG_JIT_PRINT_COMPILER_COMMAND", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt: options)
                printf("%s ", opt.c_str());
            printf("\n");
        }

        // Create NVRTC program
        nvrtcProgram program;
        DG_NVRTC_CHECK(nvrtcCreateProgram(
            &program,      // Output program handle
            code.c_str(),  // Source code
            "kernel.cu",   // Name (for error messages)
            0,             // Number of headers
            nullptr,       // Header contents
            nullptr));     // Header names
        
        // Compile the program
        const auto& compile_result = nvrtcCompileProgram(
            program, 
            static_cast<int>(option_cstrs.size()), 
            option_cstrs.data());

        // Handle compilation output/errors
        size_t log_size;
        DG_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);  // Should have error message
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        // Extract compiled CUBIN
        size_t cubin_size;
        DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write CUBIN to file
        put(cubin_path, cubin_data);

        // Clean up
        DG_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

/**
 * Global compiler instance using lazy initialization.
 * - Created on first use
 * - Selection between NVCC and NVRTC based on DG_JIT_USE_NVRTC environment variable
 * - Default: NVCC (more stable, potentially better optimization)
 * - NVRTC: Faster compilation, good for development
 */
static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCCCompiler>();
    }
});

} // namespace deep_gemm