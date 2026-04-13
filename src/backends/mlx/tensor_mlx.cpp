/**
 * @file tensor_mlx.cpp
 * @brief Apple Silicon (MLX) backend implementation for the Tensor class.
 * * This file defines the actual `TensorImpl` struct that was forward-declared
 * in `tensor.hpp`. It holds the `mlx::core::array` object, which represents
 * the multidimensional data living in the Mac's unified memory.
 */

#include <iomanip>
#include <vector>
#include <ostream>

#include "tensor_impl_mlx.hpp"
#include "isomorphism/tensor.hpp"
#include <mlx/mlx.h>

namespace isomorphism {

// ==============================================================================
// INTERNAL UTILITIES
// ==============================================================================

/**
 * @brief Helper to map our DType enum to MLX's internal type system.
 * By defining this locally, we avoid any dependency on the math namespace.
 */
static mlx::core::Dtype to_mlx_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float16:  return mlx::core::float16;
        case DType::BFloat16: return mlx::core::bfloat16;
        case DType::Float32:
        default:              return mlx::core::float32;
    }
}

// ==============================================================================
// TENSOR CLASS IMPLEMENTATION
// ==============================================================================

/** @brief Default constructor: Initializes an empty/null tensor.
 * Initializes pimpl_ to nullptr to represent an uninitialized state.
 */
Tensor::Tensor() : pimpl_(nullptr), dtype_(DType::Float32) {}

/** * @brief Scalar constructor: Initializes a 0D array with a specific double value.
 * Crucial for broadcasting, like when we do `math::multiply(X, Tensor(0.5))`.
 */
Tensor::Tensor(double scalar_value, DType dtype) : dtype_(dtype) {
    // Map our DType to MLX's internal type system via our local helper
    mlx::core::Dtype mlx_type = to_mlx_dtype(dtype);
    // Create the MLX array and wrap it in the Pimpl container
    pimpl_ = std::make_shared<TensorImpl>(mlx::core::array(static_cast<float>(scalar_value), mlx_type));
}

/**
 * @brief Unified backend constructor.
 * Required by math_mlx.cpp to wrap native MLX arrays into our public Tensor handle.
 * This resolves the "Undefined symbols" linker error.
 */
Tensor::Tensor(std::shared_ptr<TensorImpl> pimpl, DType dtype)
    : pimpl_(std::move(pimpl)), dtype_(dtype) {}

/** * @brief Destructor: Explicitly defaulted here where TensorImpl is fully defined.
 * The shared_ptr handles the actual memory cleanup.
 */
Tensor::~Tensor() = default;


// ==============================================================================
// INTROSPECTION METHODS
// ==============================================================================

std::vector<int> Tensor::shape() const {
    if (!pimpl_) return {};
    // MLX returns a custom 'SmallVector<int>', so we explicitly
    // construct a standard std::vector<int> from its iterators.
    const auto& s = pimpl_->data.shape();
    return std::vector<int>(s.begin(), s.end());
}

int Tensor::ndim() const {
    if (!pimpl_) return 0;
    return pimpl_->data.ndim();
}

int Tensor::size() const {
    if (!pimpl_) return 0;
    return pimpl_->data.size();
}


// ==============================================================================
// PRINTING & FORMATTING (N-Dimensional Traversal)
// ==============================================================================

/**
 * @brief Recursively prints N-dimensional tensor data with appropriate bracket nesting.
 */
static void print_recursive(std::ostream& os, const float* data, const std::vector<int>& shape,
                            int depth, size_t offset, const std::vector<size_t>& strides) {
    // Base case 1: 0D Tensor (Scalar)
    if (shape.empty()) {
        os << data[0];
        return;
    }

    // Base case 2: Innermost dimension (Print actual values)
    if (depth == static_cast<int>(shape.size()) - 1) {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            os << std::setprecision(6) << data[offset + i * strides[depth]];
            if (i < shape[depth] - 1) os << ", ";
        }
        os << "]";
    }
    // Recursive case: Outer dimensions (Print brackets and recurse)
    else {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            if (i > 0) {
                os << ",\n";
                // Add indentation based on depth for readability
                os << std::string(depth + 1, ' ');
            }
            print_recursive(os, data, shape, depth + 1, offset + i * strides[depth], strides);
        }
        os << "]";
    }
}

/**
 * @brief Overload for standard output streams to print the underlying tensor data.
 */
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    auto impl = tensor.get_impl();
    if (!impl) {
        return os << "Tensor(Null)";
    }

    // 1. Force the MLX backend to execute pending operations in the compute graph
    mlx::core::eval({impl->data});

    // 2. Extract shape
    auto shape = tensor.shape();

    // 3. Compute flat memory strides (assuming contiguous C-style memory for printing)
    std::vector<size_t> strides(shape.size(), 1);
    if (!shape.empty()) {
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // 4. Safely grab the raw float pointer from the MLX array
    const float* ptr = impl->data.data<float>();

    // 5. Print the header
    os << "Tensor(shape={";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i] << (i == static_cast<size_t>(shape.size()) - 1 ? "" : ", ");
    }
    os << "}, data=\n";

    // 6. Trigger recursive printing
    print_recursive(os, ptr, shape, 0, 0, strides);

    os << "\n)";
    return os;
}

} // namespace isomorphism