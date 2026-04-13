// Path: isomorphism/src/backends/torch/tensor_impl_torch.hpp
#pragma once

#include "isomorphism/tensor.hpp"
#include <torch/torch.h>

namespace isomorphism {

    static inline torch::ScalarType get_torch_dtype(DType dtype) {
        switch (dtype) {
            case DType::Float16:  return torch::kFloat16;
            case DType::BFloat16: return torch::kBFloat16;
            case DType::Float32:
            default:              return torch::kFloat32;
        }
    }

    /**
     * @struct TensorImpl
     * @brief The hidden PyTorch (LibTorch) implementation of our Tensor.
     *
     * Holds a `torch::Tensor` which carries its own device, dtype, and
     * reference-counted storage. PyTorch is eager by default, so no explicit
     * eval() calls are needed.
     */
    struct TensorImpl {
        torch::Tensor data;

        TensorImpl() : data(torch::zeros({}, torch::kFloat32)) {}
        explicit TensorImpl(torch::Tensor t) : data(std::move(t)) {}
    };

} // namespace isomorphism
