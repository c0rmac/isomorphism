/**
 * @file tensor_mlx.cpp
 * @brief Apple Silicon (MLX) backend implementation for the Tensor class.
 * * This file defines the actual `TensorImpl` struct that was forward-declared
 * in `tensor.hpp`. It holds the `mlx::core::array` object, which represents
 * the multidimensional data living in the Mac's unified memory.
 */

#include <iomanip>

#include "tensor_impl_mlx.hpp"
#include "isomorphism/tensor.hpp"
#include <mlx/mlx.h>
#include <vector>
#include <ostream>

namespace isomorphism {

// ==============================================================================
// TENSOR CLASS IMPLEMENTATION
// ==============================================================================

// Default constructor: Initializes an empty/zero scalar tensor
Tensor::Tensor() : pimpl_(std::make_shared<TensorImpl>()) {}

// Scalar constructor: Initializes a 0D array with a specific double value.
// Crucial for broadcasting, like when we do `math::multiply(X, Tensor(0.5))`
    Tensor::Tensor(double scalar_value, DType dtype) {
    mlx::core::Dtype mlx_type = get_mlx_dtype(dtype);
    pimpl_ = std::make_shared<TensorImpl>(mlx::core::array(scalar_value, mlx_type));
    dtype_ = dtype;
}

// Destructor: We must explicitly default it here in the .cpp file where
// TensorImpl is fully defined, otherwise the compiler complains about
// destroying an incomplete type.
Tensor::~Tensor() = default;

// ==============================================================================
// INTROSPECTION METHODS
// ==============================================================================

    std::vector<int> Tensor::shape() const {
        // MLX returns a custom 'SmallVector<int>', so we explicitly
        // construct a standard std::vector<int> from its iterators.
        const auto& s = pimpl_->data.shape();
        return std::vector<int>(s.begin(), s.end());
    }

int Tensor::ndim() const {
    return pimpl_->data.ndim();
}

int Tensor::size() const {
    return pimpl_->data.size();
}

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
    if (depth == shape.size() - 1) {
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

// ==============================================================================
// STREAM OUTPUT OPERATOR
// ==============================================================================

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    auto impl = tensor.get_impl(); //
    if (!impl) {
        return os << "[Null Tensor]";
    }

    // 1. Force the MLX backend to execute pending operations
    mlx::core::eval({impl->data});

    // 2. Extract shape
    auto shape = tensor.shape();

    // 3. Compute flat memory strides (assuming contiguous C-style memory)
    std::vector<size_t> strides(shape.size(), 1);
    if (!shape.empty()) {
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // 4. Safely grab the raw float pointer
    const float* ptr = impl->data.data<float>();

    // 5. Print the header
    os << "Tensor(shape={";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    os << "}, data=\n";

    // 6. Trigger recursive printing
    print_recursive(os, ptr, shape, 0, 0, strides);

    os << "\n)";
    return os;
}

} // namespace isomorphism