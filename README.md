# Isomorphism

**Isomorphism** is a hardware-agnostic C++ tensor math library with a pluggable backend architecture. Write your mathematical logic once using a unified DSL and deploy it on any hardware — Apple Silicon, CPU, or GPU — by selecting a backend at compile time.

The Pimpl pattern keeps the public API entirely decoupled from the backend. Swapping from MLX to PyTorch is a single word change in `target_link_libraries`, with zero modifications to application code.

---

## Installation via Homebrew

Each backend is a separate formula. Add the tap once, then install the formula that matches your hardware:

```bash
brew tap c0rmac/homebrew-isomorphism
```

**Apple MLX** — recommended on Apple Silicon (M1/M2/M3/M4), uses the Metal GPU:
```bash
brew install isomorphism-mlx
```

**Eigen** — lightweight CPU-only, no large framework dependency:
```bash
brew install isomorphism-eigen
```

**LibTorch** — for LibTorch / PyTorch users:
```bash
brew install isomorphism-torch
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
target_link_libraries(my_app PRIVATE isomorphism::mlx)    # or ::eigen  or ::torch
```

Each backend is a separate named target. Switching backend means changing one word:

```cmake
# Use Eigen on this machine
target_link_libraries(my_app PRIVATE isomorphism::eigen)

# Switch to Torch — no other changes needed
target_link_libraries(my_app PRIVATE isomorphism::torch)
```

Multiple backends can be installed simultaneously and a single project can link different backends into different executables:

```cmake
target_link_libraries(fast_app    PRIVATE isomorphism::mlx)
target_link_libraries(portable_app PRIVATE isomorphism::eigen)
```

**No backend flag is needed in your project's CMake.** The correct compile definition (`USE_MLX`, `USE_EIGEN`, `USE_TORCH`) is baked into each library at build time and propagated automatically to any target that links it.

If you don't want to hard-code a backend name, use the `isomorphism_DEFAULT_TARGET` variable set by `find_package`:

```cmake
find_package(isomorphism REQUIRED)
# isomorphism_DEFAULT_TARGET = whichever backend was compiled first (e.g. isomorphism::mlx)
target_link_libraries(my_app PRIVATE ${isomorphism_DEFAULT_TARGET})
```

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

### Interop: wrapping and unwrapping native types

If you already hold a native tensor from your backend, use the interop headers
to pass it into isomorphism and retrieve results without any data copy.

#### MLX

```cpp
#include <isomorphism/interop/mlx.hpp>
#include <mlx/mlx.h>

namespace iso_mlx = isomorphism::interop::mlx;

// Your existing MLX array — stays on the GPU, zero-copy
mlx::core::array my_arr = mlx::core::random::normal({4, 3, 3});

// Wrap: mlx::core::array → isomorphism::Tensor
isomorphism::Tensor A = iso_mlx::wrap(my_arr);

// Use the full isomorphism DSL
isomorphism::Tensor result = isomorphism::math::matmul(A, A);
isomorphism::math::eval(result);

// Unwrap: isomorphism::Tensor → mlx::core::array (zero-copy view)
mlx::core::array out = iso_mlx::unwrap(result);
```

#### PyTorch / LibTorch

```cpp
#include <isomorphism/interop/torch.hpp>
#include <torch/torch.h>

namespace iso_torch = isomorphism::interop::torch;

// Your existing torch tensor
torch::Tensor my_t = torch::randn({4, 3, 3});

// Wrap: torch::Tensor → isomorphism::Tensor (shares LibTorch storage)
isomorphism::Tensor A = iso_torch::wrap(my_t);

isomorphism::Tensor result = isomorphism::math::matmul(A, A);

// Unwrap: isomorphism::Tensor → torch::Tensor (zero-copy)
torch::Tensor out = iso_torch::unwrap(result);
```

#### Eigen

```cpp
#include <isomorphism/interop/eigen.hpp>
#include <Eigen/Dense>

namespace iso_eigen = isomorphism::interop::eigen;

// Your existing Eigen matrix — copied into isomorphism's pooled storage
Eigen::MatrixXf M = Eigen::MatrixXf::Random(3, 3);
isomorphism::Tensor A = iso_eigen::wrap(M);

isomorphism::Tensor result = isomorphism::math::matmul(A, A);

// Unwrap as a zero-copy Eigen::Map (contiguous tensors only)
auto out_map = iso_eigen::to_matrix_map(result);   // Eigen::Map<MatrixXf>

// Or copy into a new MatrixXf
Eigen::MatrixXf out = iso_eigen::to_matrix(result);
```

---

## Backends

| Backend | Flag | Hardware | Dependency |
|---|---|---|---|
| **MLX** | `-DUSE_MLX=ON` | Apple Silicon (Metal/GPU) | [mlx](https://github.com/ml-explore/mlx) |
| **Eigen** | `-DUSE_EIGEN=ON` | CPU (any platform) | [Eigen3](https://eigen.tuxfamily.org) |
| **Torch** | `-DUSE_TORCH=ON` | CPU / CUDA / MPS | [LibTorch](https://pytorch.org) |

One or more backends can be built simultaneously. Each installs as its own library (`libisomorphism_mlx`, `libisomorphism_eigen`, `libisomorphism_torch`) and exports as a separate CMake target (`isomorphism::mlx`, `isomorphism::eigen`, `isomorphism::torch`).

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
