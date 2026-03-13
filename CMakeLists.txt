cmake_minimum_required(VERSION 3.25)
project(isomorphism VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# 1. Global C++ Standards & Project Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend Selection Options
option(USE_MLX "Build with Apple Silicon (Metal/MLX) backend" OFF)
option(USE_SYCL "Build with PC/Intel/NVIDIA (SYCL/oneMKL) backend" OFF)
option(BUILD_TESTING "Build the Isomorphism test suite" ON)

if (USE_MLX AND USE_SYCL)
    message(FATAL_ERROR "Isomorphism: Choose either USE_MLX or USE_SYCL, not both.")
elseif (NOT USE_MLX AND NOT USE_SYCL)
    message(FATAL_ERROR "Isomorphism: No backend selected. Use -DUSE_MLX=ON or -DUSE_SYCL=ON.")
endif ()

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