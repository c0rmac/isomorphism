# Isomorphism

**Isomorphism** is a hardware-agnostic C++ tensor math library with a pluggable backend architecture. Write your mathematical logic once using a unified DSL and deploy it on any hardware — Apple Silicon, CPU, or GPU — by selecting a backend at compile time.

The Pimpl pattern keeps the public API entirely decoupled from the backend. Swapping from MLX to PyTorch is a single CMake flag change, with zero modifications to application code.

---

## Backends

| Backend | Flag | Hardware | Dependency |
|---|---|---|---|
| **MLX** | `-DUSE_MLX=ON` | Apple Silicon (Metal/GPU) | [mlx](https://github.com/ml-explore/mlx) |
| **Eigen** | `-DUSE_EIGEN=ON` | CPU (any platform) | [Eigen3](https://eigen.tuxfamily.org) |
| **Torch** | `-DUSE_TORCH=ON` | CPU / CUDA / MPS | [LibTorch](https://pytorch.org) |

Exactly one backend must be active per build.

---

## Installation via Homebrew

```bash
# Tap the repository
brew tap c0rmac/homebrew-isomorphism

# Auto-select (MLX on Apple Silicon, Eigen elsewhere)
brew install isomorphism

# Or choose a backend explicitly
brew install isomorphism --with-mlx      # MLX Library exceptionally well optimised for the Apple Silicon GPU
brew install isomorphism --with-eigen    # Eigen Library
brew install isomorphism --with-torch    # PyTorch / LibTorch
```

---

## Building from Source

```bash
git clone https://github.com/c0rmac/isomorphism.git
cd isomorphism
cmake -S . -B build -DUSE_MLX=ON -DCMAKE_BUILD_TYPE=Release   # or USE_EIGEN / USE_TORCH
cmake --build build
cmake --install build
```

**PyTorch backend** — point CMake at your LibTorch installation:
```bash
cmake -S . -B build \
  -DUSE_TORCH=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release
```
On macOS with Homebrew, `abseil` must also be installed (`brew install abseil`) so that LibTorch's protobuf dependency resolves correctly.

---

## Integrating with Your Project

```cmake
find_package(isomorphism REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE isomorphism::isomorphism)
```

That's it. Include paths and backend linking are handled by the exported target.

---

## API Overview

All operations live in the `isomorphism::math` namespace and operate on the opaque `isomorphism::Tensor` handle.

### Element-wise arithmetic
```cpp
math::add(a, b)        math::subtract(a, b)    math::multiply(a, b)
math::divide(a, b)     math::pow(a, exp)        math::clamp(a, lo, hi)
math::floor(a)         math::ceil(a)            math::round(a)
math::minimum(a, b)    math::mean(a)
```

### Matrix operations
```cpp
math::matmul(a, b)
math::transpose(a, axes)
math::reshape(a, shape)
math::broadcast_to(a, shape)
math::expand_dims(a, axes)
math::squeeze(a, axes)
math::eye(d, dtype)
math::full(shape, value, dtype)
math::stack(tensors, axis)
math::concatenate(tensors, axis)
math::slice(a, start, end, axis)
math::gather(a, indices, axis)
```

### Linear algebra
```cpp
math::matmul(a, b)                    // batched matrix multiply
math::solve(A, b)                     // AX = B
auto [U, S, Vh] = math::svd(a)        // singular value decomposition
auto [Q, R]     = math::qr(a)         // QR decomposition
math::inv(a)                          // matrix inverse
math::det(a)                          // determinant
math::trace(a)                        // sum of diagonal
math::matrix_exp(a)                   // matrix exponential
math::diag_embed(v)                   // 1D vector → diagonal matrix
math::diag_extract(a)                 // diagonal matrix → 1D vector
```

### Reductions
```cpp
math::sum(a, axes)     math::prod(a, axes)     math::mean(a)
math::min(a)           math::max(a)            math::argmax(a, axis)
math::cumsum(a, axis)  math::all(a, axes)      math::any(a, axes)
```

### Activation / transcendental
```cpp
math::exp(a)    math::log(a)     math::sqrt(a)    math::square(a)
math::abs(a)    math::sign(a)    math::sin(a)     math::cos(a)
math::tan(a)    math::asin(a)    math::acos(a)    math::atan(a)
math::atan2(y, x)
```

### Logical
```cpp
math::where(cond, x, y)
math::equal(a, b)        math::not_equal(a, b)
math::greater(a, b)      math::greater_equal(a, b)
math::less(a, b)         math::less_equal(a, b)
math::logical_and(a, b)  math::logical_or(a, b)
```

### Random
```cpp
math::random_normal(shape, dtype)
math::random_uniform(shape, dtype)
```

### Spectral
```cpp
math::rfft(a, n, axis)
math::irfft(a, n, axis)
```

### CPU / GPU bridge
```cpp
double            math::to_double(a)
int               math::to_int(a)
std::vector<float> math::to_float_vector(a)
void              math::eval(a)               // no-op on eager backends
void              math::set_default_device_cpu()
void              math::set_default_device_gpu()
```

---

## Quick Example

```cpp
#include <isomorphism/math.hpp>
#include <isomorphism/tensor.hpp>
#include <iostream>

namespace iso = isomorphism;
using namespace iso::math;

int main() {
    // Create a batch of 4 random 3×3 matrices
    iso::Tensor A = random_normal({4, 3, 3}, iso::DType::Float32);
    iso::Tensor I = eye(3, iso::DType::Float32);

    // Batched matmul — shape stays {4, 3, 3}
    iso::Tensor result = matmul(A, broadcast_to(I, {4, 3, 3}));

    // Pull a scalar to CPU
    std::cout << "trace[0] = " << to_double(slice(trace(result), 0, 1, 0)) << "\n";
    return 0;
}
```

---

## Project Structure

```
include/isomorphism/
    tensor.hpp          # Public Tensor handle (Pimpl)
    math.hpp            # Full DSL — backend-agnostic

src/backends/
    mlx/                # Apple Silicon (MLX/Metal)
    eigen/              # CPU (Eigen + OpenMP)
    torch/              # PyTorch (LibTorch)
    sycl/               # PC GPU (SYCL / oneMKL) — in progress
```

---

## License

MIT
