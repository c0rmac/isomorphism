# Isomorphism

**Isomorphism** is a platform-agnostic C++ math wrapper that unifies **MLX** (Apple Silicon) and **SYCL** (oneMKL/PC) backends. It acts as a routing layer, translating high-level tensor operations into highly optimized native calls for the active hardware, ensuring seamless performance across disparate architectures.

By using Isomorphism, you write your mathematical logic once using a Domain Specific Language (DSL) and deploy it anywhere without worrying about the underlying silicon.

---

## Key Features
* **Platform-Agnostic DSL**: A unified interface for tensor operations including element-wise arithmetic and logical operations.
* **Intelligent Routing**: Automatically dispatches math to MLX on Apple devices or SYCL/oneMKL on Intel/NVIDIA/AMD hardware.
* **Heavy Linear Algebra**: Built-in support for SVD, QR decomposition, determinants, and linear system solvers.
* **Seamless Integration**: Designed to be a lightweight dependency for any C++ project through a modern CMake interface.

---

## Installation

The recommended way to install **Isomorphism** on macOS is via Homebrew. This ensures all backend dependencies, like MLX, are correctly configured.

```bash
# 1. Install MLX for macOS compatibility
brew install mlx

# 2. Tap the custom Isomorphism repository
brew tap c0rmac/homebrew-isomorphism

# 3. Install the library
brew install isomorphism
```

## Integrating with Your Project

Once installed, you can integrate Isomorphism into your project using CMake. The package exports a clean target that handles all include paths and backend-specific linking automatically.

### CMake Configuration
Add the following to your `CMakeLists.txt`:

```cmake
# 1. Locate the Isomorphism package
find_package(isomorphism REQUIRED)

# 2. Your executable
add_executable(my_app main.cpp)

# 3. Link the Isomorphism target
# This automatically handles include paths and backend dependencies (like MLX/SYCL)
target_link_libraries(my_app PRIVATE isomorphism::isomorphism)
```

### Basic Usage
```cmake
#include <isomorphism/math.hpp>

namespace iso = isomorphism::core::math;

int main() {
    // Operations are automatically routed to the active backend (MLX or SYCL)
    auto a = iso::full({3, 3}, 1.0f, DType::Float32);
    auto b = iso::eye(3, DType::Float32);
    
    // Perform a batched matrix multiplication
    auto result = iso::matmul(a, b);
    
    // Explicitly execute the computation graph
    iso::eval(result);
    
    return 0;
}
```

## Project Structure
* `include/isomorphism/`: Contains the public API and hardware-agnostic DSL headers.
* `src/backends/mlx/`: Implementation files for the Apple Silicon backend using the MLX framework.
* `src/backends/sycl/`: Implementation files for the PC/Linux backend using SYCL and oneMKL.

---

## License
This project is licensed under the MIT License.