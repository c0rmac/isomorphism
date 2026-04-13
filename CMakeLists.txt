cmake_minimum_required(VERSION 3.25)
project(isomorphism VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# 1. Global C++ Standards & Project Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend Selection Options
option(USE_MLX   "Build with Apple Silicon (Metal/MLX) backend" OFF)
option(USE_SYCL  "Build with PC/Intel/NVIDIA (SYCL/oneMKL) backend" OFF)
option(USE_EIGEN "Build with Eigen CPU backend" OFF)
option(USE_TORCH "Build with PyTorch (LibTorch) backend" OFF)
option(BUILD_TESTING "Build the Isomorphism test suite" ON)

set(_num_backends 0)
if (USE_MLX)
    math(EXPR _num_backends "${_num_backends} + 1")
endif ()
if (USE_SYCL)
    math(EXPR _num_backends "${_num_backends} + 1")
endif ()
if (USE_EIGEN)
    math(EXPR _num_backends "${_num_backends} + 1")
endif ()
if (USE_TORCH)
    math(EXPR _num_backends "${_num_backends} + 1")
endif ()

if (_num_backends GREATER 1)
    message(FATAL_ERROR "Isomorphism: Choose exactly one backend (USE_MLX, USE_SYCL, USE_EIGEN, or USE_TORCH).")
elseif (_num_backends EQUAL 0)
    message(FATAL_ERROR "Isomorphism: No backend selected. Use -DUSE_MLX=ON, -DUSE_SYCL=ON, -DUSE_EIGEN=ON, or -DUSE_TORCH=ON.")
endif ()

if (APPLE)
    # Provide a hint to CMake for Homebrew's libomp location
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP REQUIRED)

# ==============================================================================
# 2. Core Library Definition
# ==============================================================================
add_library(isomorphism)

# Setting up the File Set for modern header management
target_sources(isomorphism
        PUBLIC FILE_SET HEADERS
        BASE_DIRS include
        FILES
        include/isomorphism/math.hpp
        include/isomorphism/tensor.hpp
)

target_include_directories(isomorphism PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ==============================================================================
# 3. Debug & Optimization Configuration
# ==============================================================================
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Isomorphism: Debug mode enabled. Adding sanitizers.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -g")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        target_compile_options(isomorphism PRIVATE -g -fno-omit-frame-pointer -fstack-protector-strong)
    endif ()
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        target_compile_options(isomorphism PRIVATE -O3 -ffast-math -march=native)
    endif ()
endif ()

# ==============================================================================
# 4. Backend-Specific Implementation
# ==============================================================================

# Apple Silicon / MLX Backend
if (USE_MLX)
    message(STATUS "Isomorphism: Configuring MLX Backend (Apple Silicon)")
    target_compile_definitions(isomorphism PUBLIC USE_MLX)

    target_sources(isomorphism PRIVATE
            src/backends/mlx/tensor_mlx.cpp
            src/backends/mlx/math_mlx.cpp
    )

    find_package(MLX REQUIRED)
    target_link_libraries(isomorphism PRIVATE mlx)

# Eigen CPU Backend
elseif (USE_EIGEN)
    message(STATUS "Isomorphism: Configuring Eigen CPU Backend")
    target_compile_definitions(isomorphism PUBLIC USE_EIGEN)

    target_sources(isomorphism PRIVATE
            src/backends/eigen/tensor_eigen.cpp
            src/backends/eigen/math_eigen.cpp
    )

    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(isomorphism PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

# PyTorch (LibTorch) Backend
# Requires LibTorch to be discoverable via CMAKE_PREFIX_PATH, e.g.:
#   cmake -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
elseif (USE_TORCH)
    message(STATUS "Isomorphism: Configuring PyTorch (LibTorch) Backend")
    target_compile_definitions(isomorphism PUBLIC USE_TORCH)

    target_sources(isomorphism PRIVATE
            src/backends/torch/tensor_torch.cpp
            src/backends/torch/math_torch.cpp
    )

    # On Homebrew macOS, LibTorch's bundled protobuf declares absl:: link targets
    # but never calls find_package(absl) itself, so "absl::absl_check not found"
    # errors occur at configure time.  We pre-find absl before find_package(Torch).
    #
    # GUI tools like CLion don't inherit the shell PATH, so execute_process(brew)
    # may fail.  Instead we probe the Homebrew opt symlinks directly — Homebrew
    # always installs formula at:
    #   Apple Silicon : /opt/homebrew/opt/<formula>
    #   Intel Mac     : /usr/local/opt/<formula>
    if (APPLE AND NOT absl_FOUND)
        foreach (_absl_candidate
                "/opt/homebrew/opt/abseil/lib/cmake/absl"   # Apple Silicon
                "/usr/local/opt/abseil/lib/cmake/absl")     # Intel Mac
            if (EXISTS "${_absl_candidate}/abslConfig.cmake")
                set(absl_DIR "${_absl_candidate}" CACHE PATH "absl CMake config dir" FORCE)
                break()
            endif ()
        endforeach ()

        if (NOT absl_DIR)
            message(FATAL_ERROR
                "Isomorphism [USE_TORCH]: abseil not found.\n"
                "LibTorch's protobuf requires it.  Install via:\n"
                "  brew install abseil")
        endif ()
    endif ()
    find_package(absl QUIET)

    find_package(Torch REQUIRED)
    # Suppress warnings from LibTorch headers by marking them as system includes.
    target_include_directories(isomorphism SYSTEM PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(isomorphism PRIVATE "${TORCH_LIBRARIES}")
    # LibTorch may require C++17 ABI flags; propagate them if set.
    if (TORCH_CXX_FLAGS)
        target_compile_options(isomorphism PRIVATE ${TORCH_CXX_FLAGS})
    endif ()

# PC / Intel / NVIDIA SYCL Backend
elseif (USE_SYCL)
    message(STATUS "Isomorphism: Configuring SYCL/oneMKL Backend (PC GPU)")
    target_compile_definitions(isomorphism PUBLIC USE_SYCL)

    target_sources(isomorphism PRIVATE
            src/backends/sycl/tensor_sycl.cpp
            src/backends/sycl/math_sycl.cpp
    )

    find_package(MKL REQUIRED)
    target_compile_options(isomorphism PRIVATE -fsycl)
    target_link_options(isomorphism PRIVATE -fsycl)
    target_link_libraries(isomorphism PRIVATE MKL::MKL)
endif ()

# ==============================================================================
# 5. Installation & Export (For use in other projects)
# ==============================================================================
install(TARGETS isomorphism
        EXPORT IsomorphismTargets
        FILE_SET HEADERS
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(EXPORT IsomorphismTargets
        FILE IsomorphismTargets.cmake
        NAMESPACE isomorphism::
        DESTINATION lib/cmake/isomorphism
)

# Create a config file so find_package(isomorphism) works seamlessly
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/isomorphismConfig.cmake"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/IsomorphismTargets.cmake\")"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/isomorphismConfig.cmake"
        DESTINATION lib/cmake/isomorphism
)

# ==============================================================================
# Tests
# ==============================================================================
add_subdirectory(tests)