/**
 * @file interop/torch.hpp
 * @brief Zero-copy interop between isomorphism::Tensor and torch::Tensor.
 *
 * Include this header when you are building against the PyTorch/LibTorch
 * backend and need to pass native torch tensors into (or out of) the
 * isomorphism API without a CPU round-trip.
 *
 * Only available when compiled with USE_TORCH.
 *
 * Usage:
 * @code
 *   #include <isomorphism/interop/torch.hpp>
 *   namespace iso_torch = isomorphism::interop::torch;
 *
 *   // torch → isomorphism
 *   torch::Tensor my_t = ...;
 *   isomorphism::Tensor t = iso_torch::wrap(my_t);
 *
 *   // isomorphism → torch
 *   torch::Tensor out = iso_torch::unwrap(t);
 * @endcode
 */

#pragma once

#ifndef USE_TORCH
#  error "isomorphism/interop/torch.hpp requires the Torch backend (compile with USE_TORCH defined)"
#endif

#include "isomorphism/tensor.hpp"
#include <torch/torch.h>

#if __has_include(<isomorphism/src/backends/torch/tensor_impl_torch.hpp>)
#  include <isomorphism/src/backends/torch/tensor_impl_torch.hpp>
#else
#  include "../../src/backends/torch/tensor_impl_torch.hpp"
#endif

namespace isomorphism::interop::torch {

/**
 * @brief Wrap a native torch::Tensor as an isomorphism::Tensor.
 *
 * Shares LibTorch's reference-counted storage — no data is copied.
 * The DType is inferred from the torch tensor's scalar type.
 *
 * @param t  The torch::Tensor to wrap.
 * @return   An isomorphism::Tensor backed by @p t.
 */
inline isomorphism::Tensor wrap(::torch::Tensor t) {
    auto dtype = [&]() -> isomorphism::DType {
        if (t.scalar_type() == ::torch::kFloat16)  return isomorphism::DType::Float16;
        if (t.scalar_type() == ::torch::kBFloat16) return isomorphism::DType::BFloat16;
        return isomorphism::DType::Float32;
    }();
    return isomorphism::Tensor(
        std::make_shared<isomorphism::TensorImpl>(std::move(t)), dtype);
}

/**
 * @brief Unwrap an isomorphism::Tensor to its underlying torch::Tensor.
 *
 * Zero-copy: returns a view into LibTorch's storage.
 * Behaviour is undefined if the Tensor was not created by the Torch backend.
 *
 * @param t  A Tensor created by the Torch backend.
 * @return   The underlying torch::Tensor.
 */
inline ::torch::Tensor unwrap(const isomorphism::Tensor& t) {
    return t.get_impl()->data;
}

} // namespace isomorphism::interop::torch
