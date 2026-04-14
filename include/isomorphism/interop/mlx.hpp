/**
 * @file interop/mlx.hpp
 * @brief Zero-copy interop between isomorphism::Tensor and mlx::core::array.
 *
 * Include this header when you are building against the MLX backend and need
 * to pass native MLX arrays into (or out of) the isomorphism API without going
 * through a CPU round-trip.
 *
 * Only available when compiled with USE_MLX.
 *
 * Usage:
 * @code
 *   #include <isomorphism/interop/mlx.hpp>
 *   namespace iso_mlx = isomorphism::interop::mlx;
 *
 *   // mlx → isomorphism
 *   mlx::core::array my_arr = ...;
 *   isomorphism::Tensor t = iso_mlx::wrap(my_arr);
 *
 *   // isomorphism → mlx
 *   mlx::core::array out = iso_mlx::unwrap(t);
 * @endcode
 */

#pragma once

#ifndef USE_MLX
#  error "isomorphism/interop/mlx.hpp requires the MLX backend (compile with USE_MLX defined)"
#endif

#include "isomorphism/tensor.hpp"
#include <mlx/mlx.h>

// Pull in the backend-specific TensorImpl definition.
// In the source tree this resolves via the relative path; after installation
// the CMake rules place it under include/isomorphism/src/backends/mlx/.
#if __has_include(<isomorphism/src/backends/mlx/tensor_impl_mlx.hpp>)
#  include <isomorphism/src/backends/mlx/tensor_impl_mlx.hpp>
#else
#  include "../../src/backends/mlx/tensor_impl_mlx.hpp"
#endif

namespace isomorphism::interop::mlx {

/**
 * @brief Wrap a native mlx::core::array as an isomorphism::Tensor.
 *
 * The resulting Tensor shares ownership of the underlying MLX buffer — no
 * data is copied.  The tensor's DType is inferred from the array's dtype.
 *
 * @param arr  The MLX array to wrap.
 * @return     An isomorphism::Tensor backed by @p arr.
 */
inline isomorphism::Tensor wrap(mlx::core::array arr) {
    auto dtype = [&]() -> isomorphism::DType {
        if (arr.dtype() == mlx::core::float16)  return isomorphism::DType::Float16;
        if (arr.dtype() == mlx::core::bfloat16) return isomorphism::DType::BFloat16;
        return isomorphism::DType::Float32;
    }();
    return isomorphism::Tensor(
        std::make_shared<isomorphism::TensorImpl>(std::move(arr)), dtype);
}

/**
 * @brief Unwrap an isomorphism::Tensor to its underlying mlx::core::array.
 *
 * This is a zero-copy view into the same GPU/unified-memory buffer.
 * The caller is responsible for ensuring the Tensor was created with the MLX
 * backend; behaviour is undefined if called on a Tensor from a different backend.
 *
 * @param t  A Tensor created by the MLX backend.
 * @return   The underlying mlx::core::array.
 */
inline mlx::core::array unwrap(const isomorphism::Tensor& t) {
    return t.get_impl()->data;
}

} // namespace isomorphism::interop::mlx
