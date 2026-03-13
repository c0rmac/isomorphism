// Path: isomorphism/src/backends/mlx/tensor_impl_mlx.hpp
#pragma once

#include "isomorphism/tensor.hpp"
#include <mlx/mlx.h>

// ==============================================================================
// THE PIMPL DEFINITION
// ==============================================================================

namespace isomorphism {

    // Translates isomorphism::DType to mlx::core::Dtype
    static inline mlx::core::Dtype get_mlx_dtype(DType dtype) {
        switch (dtype) {
            case DType::Float16:  return mlx::core::float16;
            case DType::BFloat16: return mlx::core::bfloat16;
            case DType::Float32:
            default:              return mlx::core::float32;
        }
    }

    /**
     * @struct TensorImpl
     * @brief The hidden MLX implementation of our Tensor.
     * * This struct is strictly internal to the MLX backend. It holds the actual
     * MLX array. Because `mlx::core::array` uses lazy evaluation and its own
     * internal reference counting, we are essentially piggybacking on Apple's
     * highly optimized memory management.
     */
    // The shared internal definition for the MLX backend
    struct TensorImpl {
        mlx::core::array data;

        TensorImpl() : data(mlx::core::array(0.0f)) {}
        explicit TensorImpl(mlx::core::array arr) : data(std::move(arr)) {}
    };

} // namespace isomorphism