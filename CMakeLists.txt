cmake_minimum_required(VERSION 3.25)
project(isomorphism VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# 1. Global C++ Standards & Project Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

# Backend Selection Options — one or more may be enabled simultaneously.
# Each produces its own library: libisomorphism_mlx, libisomorphism_eigen, etc.
option(USE_MLX   "Build with Apple Silicon (Metal/MLX) backend" OFF)
option(USE_SYCL  "Build with PC/Intel/NVIDIA (SYCL/oneMKL) backend" OFF)
option(USE_EIGEN "Build with Eigen CPU backend" OFF)
option(USE_TORCH "Build with PyTorch (LibTorch) backend" OFF)
option(BUILD_TESTING "Build the Isomorphism test suite" ON)

set(_num_backends 0)
foreach (_b MLX SYCL EIGEN TORCH)
    if (USE_${_b})
        math(EXPR _num_backends "${_num_backends} + 1")
    endif ()
endforeach ()

if (_num_backends EQUAL 0)
    message(FATAL_ERROR
        "Isomorphism: No backend selected.\n"
        "Enable one or more: -DUSE_MLX=ON  -DUSE_EIGEN=ON  -DUSE_TORCH=ON  -DUSE_SYCL=ON")
endif ()

if (APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif ()
find_package(OpenMP REQUIRED)

# ==============================================================================
# 2. Per-Backend Library Macro
# ==============================================================================
# Creates one self-contained library target per backend.
# Each is named  isomorphism_<name>  and installs as  libisomorphism_<name>.
# It also registers a build-tree alias  isomorphism::<name>.
# The first backend registered becomes the default (used by tests).
#
# Usage: isomorphism_add_backend(<name> <COMPILE_DEFINE> <src1> [src2 ...])
# ==============================================================================
macro(isomorphism_add_backend _name _define)
    set(_sources ${ARGN})

    add_library(isomorphism_${_name} ${_sources})
    set_target_properties(isomorphism_${_name} PROPERTIES
        OUTPUT_NAME  "isomorphism_${_name}"   # library file stays libisomorphism_mlx.dylib
        EXPORT_NAME  "${_name}"               # exported as isomorphism::mlx (not isomorphism::isomorphism_mlx)
    )

    target_include_directories(isomorphism_${_name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    # The PUBLIC compile definition propagates to any downstream target that
    # links this library — no need for the consumer to set USE_MLX etc. manually.
    target_compile_definitions(isomorphism_${_name} PUBLIC ${_define})

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -O0 -g")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -g")
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
            #target_compile_options(isomorphism_${_name} PRIVATE  -g -fno-omit-frame-pointer -fstack-protector-strong)
        endif ()
    endif ()

    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
            target_compile_options(isomorphism_${_name} PRIVATE -O3 -march=native)
        endif ()
    endif ()

    # Alias for build-tree consumers (e.g. add_subdirectory users)
    add_library(isomorphism::${_name} ALIAS isomorphism_${_name})

    # Accumulate for the install() call below
    list(APPEND _iso_install_targets isomorphism_${_name})

    # The first registered backend becomes the default
    if (NOT DEFINED _iso_default_backend)
        set(_iso_default_backend ${_name})
    endif ()
endmacro()

# ==============================================================================
# 3. Backend Targets
# ==============================================================================

# --- Apple Silicon / MLX ---
if (USE_MLX)
    message(STATUS "Isomorphism: Building MLX backend (Apple Silicon)")
    find_package(MLX REQUIRED)

    # QR, eigh and SVD on the GPU come from metal-linalg
    # (https://github.com/c0rmac/metal-linalg). Nothing has to be installed by
    # hand: CMake takes the first of
    #
    #   1. METAL_LINALG_SOURCE_DIR, if set: that checkout, built in from source;
    #   2. a checkout beside this project (../metal-linalg), likewise;
    #   3. an installed metal-linalg, e.g. from Homebrew;
    #   4. otherwise metal-linalg is downloaded from METAL_LINALG_GIT_REPOSITORY
    #      and built along with this project, automatically.
    #
    # Built in from source (1, 2, 4) it is a static library folded into
    # libisomorphism_mlx, and its shaders and per-device routing tables are the
    # checkout's. -DMETAL_LINALG_USE_INSTALLED=ON insists on 3, which is what
    # the isomorphism-mlx formula does, since a Homebrew build has no network.
    # Either way the library carries its own compiled shaders.
    option(METAL_LINALG_USE_INSTALLED
           "Use only an installed metal-linalg (find_package), never a source checkout or a download" OFF)
    set(METAL_LINALG_SOURCE_DIR "" CACHE PATH
        "metal-linalg checkout to build from source (empty: look beside this project)")
    set(METAL_LINALG_GIT_REPOSITORY "https://github.com/c0rmac/metal-linalg.git" CACHE STRING
        "Where to download metal-linalg from when no checkout or installed copy is found")
    set(METAL_LINALG_GIT_TAG "main" CACHE STRING "Git tag or branch of metal-linalg to download")

    set(_iso_metal_linalg_installed OFF)
    if (METAL_LINALG_USE_INSTALLED)
        find_package(MetalLinalg REQUIRED)
        set(_iso_metal_linalg_installed ON)
        set(_iso_metal_linalg_from "the installed package (${MetalLinalg_DIR})")
    else ()
        if (NOT METAL_LINALG_SOURCE_DIR
            AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../metal-linalg/include/metal_linalg/qr.h")
            get_filename_component(METAL_LINALG_SOURCE_DIR
                                   "${CMAKE_CURRENT_SOURCE_DIR}/../metal-linalg" ABSOLUTE)
        endif ()
        if (METAL_LINALG_SOURCE_DIR)
            add_subdirectory("${METAL_LINALG_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/metal-linalg")
            set(_iso_metal_linalg_from "source at ${METAL_LINALG_SOURCE_DIR}")
        else ()
            find_package(MetalLinalg CONFIG QUIET)
            if (MetalLinalg_FOUND)
                set(_iso_metal_linalg_installed ON)
                set(_iso_metal_linalg_from "the installed package (${MetalLinalg_DIR})")
            else ()
                include(FetchContent)
                FetchContent_Declare(metal_linalg
                    GIT_REPOSITORY "${METAL_LINALG_GIT_REPOSITORY}"
                    GIT_TAG        "${METAL_LINALG_GIT_TAG}"
                    GIT_SHALLOW    TRUE)
                FetchContent_MakeAvailable(metal_linalg)
                set(_iso_metal_linalg_from
                    "${METAL_LINALG_GIT_REPOSITORY} (${METAL_LINALG_GIT_TAG}), downloaded and built in")
            endif ()
        endif ()
    endif ()
    message(STATUS "Isomorphism: metal-linalg from ${_iso_metal_linalg_from}")

    if (TARGET metal_linalg)
        # Built in from source: it still has to sit in the export set for a
        # static isomorphism build to be installable.
        install(TARGETS metal_linalg EXPORT IsomorphismTargets
                ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
    endif ()

    isomorphism_add_backend(mlx USE_MLX
        src/backends/mlx/tensor_mlx.cpp
        src/backends/mlx/math_mlx.cpp
    )
    target_link_libraries(isomorphism_mlx PRIVATE
        mlx
        metal_linalg::metal_linalg
    )
endif ()

# --- Eigen CPU ---
if (USE_EIGEN)
    message(STATUS "Isomorphism: Building Eigen backend (CPU)")
    find_package(Eigen3 REQUIRED NO_MODULE)
    isomorphism_add_backend(eigen USE_EIGEN
        src/backends/eigen/tensor_eigen.cpp
        src/backends/eigen/math_eigen.cpp
    )
    target_link_libraries(isomorphism_eigen PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
endif ()

# --- PyTorch / LibTorch ---
if (USE_TORCH)
    message(STATUS "Isomorphism: Building Torch backend (PyTorch/LibTorch)")

    # On Homebrew macOS, LibTorch's bundled protobuf declares absl:: link targets
    # but never calls find_package(absl) itself.  We probe the Homebrew opt
    # symlinks directly — GUI tools like CLion don't inherit the shell PATH so
    # execute_process(brew) is unreliable.
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
                "LibTorch's protobuf requires it.  Install via: brew install abseil")
        endif ()
    endif ()
    find_package(absl QUIET)
    find_package(Torch REQUIRED)

    isomorphism_add_backend(torch USE_TORCH
        src/backends/torch/tensor_torch.cpp
        src/backends/torch/math_torch.cpp
    )
    target_include_directories(isomorphism_torch SYSTEM PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(isomorphism_torch PRIVATE "${TORCH_LIBRARIES}")
    if (TORCH_CXX_FLAGS)
        target_compile_options(isomorphism_torch PRIVATE ${TORCH_CXX_FLAGS})
    endif ()
endif ()

# --- SYCL / oneMKL ---
if (USE_SYCL)
    message(STATUS "Isomorphism: Building SYCL backend (PC GPU / oneMKL)")
    find_package(MKL REQUIRED)
    isomorphism_add_backend(sycl USE_SYCL
        src/backends/sycl/tensor_sycl.cpp
        src/backends/sycl/math_sycl.cpp
    )
    target_compile_options(isomorphism_sycl PRIVATE -fsycl)
    target_link_options(isomorphism_sycl PRIVATE -fsycl)
    target_link_libraries(isomorphism_sycl PRIVATE MKL::MKL)
endif ()

# ==============================================================================
# 4. Build-Tree Default Alias
# ==============================================================================
# Plain "isomorphism" target used by tests/add_subdirectory consumers.
# Points to the first registered backend (MLX > Eigen > Torch > SYCL).
add_library(isomorphism ALIAS isomorphism_${_iso_default_backend})

# ==============================================================================
# 5. Installation & Export
# ==============================================================================

# Public headers (include/) are shared by all backends — install once.
install(DIRECTORY include/ DESTINATION include)

# Backend TensorImpl headers are required by the interop/ headers at consumer
# compile time.  Install each one only when its backend was actually built.
if (USE_MLX)
    install(FILES src/backends/mlx/tensor_impl_mlx.hpp
        DESTINATION include/isomorphism/src/backends/mlx)
endif ()
if (USE_EIGEN)
    install(FILES src/backends/eigen/tensor_impl_eigen.hpp
                  src/backends/eigen/memory_pool.hpp
        DESTINATION include/isomorphism/src/backends/eigen)
endif ()
if (USE_TORCH)
    install(FILES src/backends/torch/tensor_impl_torch.hpp
        DESTINATION include/isomorphism/src/backends/torch)
endif ()

# Install all built backend libraries under the same export set.
install(TARGETS ${_iso_install_targets}
    EXPORT IsomorphismTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT IsomorphismTargets
    FILE IsomorphismTargets.cmake
    NAMESPACE isomorphism::
    DESTINATION lib/cmake/isomorphism
)

# Config file:
#   • Loads the exported targets  (isomorphism::mlx, isomorphism::eigen, …)
#   • Exposes isomorphism_DEFAULT_TARGET pointing to whichever backend was
#     first registered in this build — consumers can use it directly or
#     override it with the explicit target name they want.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/isomorphismConfig.cmake"
[=[
include("${CMAKE_CURRENT_LIST_DIR}/IsomorphismTargets.cmake")

# isomorphism_DEFAULT_TARGET — the first backend compiled into this installation.
# Override in your own CMakeLists.txt to select a different backend:
#
#   target_link_libraries(my_app PRIVATE isomorphism::torch)
#   target_link_libraries(my_app PRIVATE isomorphism::eigen)
#   target_link_libraries(my_app PRIVATE isomorphism::mlx)
]=])
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/isomorphismConfig.cmake"
    "set(isomorphism_DEFAULT_TARGET isomorphism::${_iso_default_backend})\n")
if (USE_MLX AND _iso_metal_linalg_installed)
    # isomorphism_mlx links the installed metal-linalg, so a consumer of this
    # installation needs it found too (a static build references its target).
    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/isomorphismConfig.cmake"
        "include(CMakeFindDependencyMacro)\nfind_dependency(MetalLinalg)\n")
endif ()

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/isomorphismConfig.cmake"
    DESTINATION lib/cmake/isomorphism
)

# ==============================================================================
# 6. Tests
# ==============================================================================
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif ()
