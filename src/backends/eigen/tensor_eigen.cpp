/**
 * @file tensor_eigen.cpp
 * @brief Implementation of the Tensor class for the Eigen CPU backend.
 * * This file implements the hardware-agnostic Tensor interface using
 * Eigen-compatible strided memory management.
 */

#include "isomorphism/tensor.hpp"
#include "tensor_impl_eigen.hpp"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace isomorphism {

// ============================================================================
// CONSTRUCTORS & DESTRUCTOR
// ============================================================================

/**
 * @brief Default constructor creating an empty/null tensor.
 */
Tensor::Tensor() : pimpl_(nullptr), dtype_(DType::Float32) {}

/**
 * @brief Constructs a 0D (scalar) tensor from a double.
 * Internally, the value is stored as a float to maintain SIMD
 * compatibility with the Eigen backend.
 */
Tensor::Tensor(double scalar_value, DType dtype) : dtype_(dtype) {
    // Allocate a TensorImpl with a rank-1 shape of size 1
    pimpl_ = std::make_shared<TensorImpl>(std::vector<int>{1});
    pimpl_->data_ptr[0] = static_cast<float>(scalar_value);
}

/**
 * @brief Unified backend constructor.
 * Allows the math layer to wrap an existing TensorImpl (view or allocation).
 */
Tensor::Tensor(std::shared_ptr<TensorImpl> pimpl, DType dtype)
    : pimpl_(std::move(pimpl)), dtype_(dtype) {}

/**
 * @brief Destructor.
 * Managed automatically by std::shared_ptr reference counting.
 */
Tensor::~Tensor() = default;


// ============================================================================
// INTROSPECTION
// ============================================================================

/**
 * @brief Returns the number of dimensions (rank) of the tensor.
 */
int Tensor::ndim() const {
    if (!pimpl_) return 0;
    return static_cast<int>(pimpl_->shape.size());
}

/**
 * @brief Returns the dimensions of the tensor as a vector of integers.
 */
std::vector<int> Tensor::shape() const {
    if (!pimpl_) return {};
    return pimpl_->shape;
}

/**
 * @brief Returns the total number of individual elements in the tensor view.
 */
int Tensor::size() const {
    if (!pimpl_) return 0;
    return pimpl_->size();
}


// ============================================================================
// PRINTING & FORMATTING (Strided Memory Traversal)
// ============================================================================

/**
 * @brief Helper to recursively traverse the N-Dimensional shape using Strides.
 * * This is essential because the data_ptr may represent a non-contiguous view
 * (like a transpose). We calculate the flat index for each element using:
 * flat_idx = offset + sum(coord[i] * stride[i]).
 */
static void print_recursive(std::ostream& os,
                            const TensorImpl& impl,
                            std::vector<int>& coords,
                            int current_dim)
{
    // Base Case: We've reached the innermost dimension
    if (current_dim == static_cast<int>(impl.shape.size())) {
        int flat_idx = impl.offset;
        for (size_t i = 0; i < coords.size(); ++i) {
            flat_idx += coords[i] * impl.strides[i];
        }

        os << std::defaultfloat << impl.data_ptr[flat_idx];
        return;
    }

    // Recursive Case: Open a new dimension level
    os << "[";
    for (int i = 0; i < impl.shape[current_dim]; ++i) {
        coords.push_back(i);
        print_recursive(os, impl, coords, current_dim + 1);
        coords.pop_back();

        // Add separators and handle multi-line indentation
        if (i < impl.shape[current_dim] - 1) {
            os << ", ";
            if (current_dim < static_cast<int>(impl.shape.size()) - 1) {
                os << "\n" << std::string(current_dim + 1, ' ');
            }
        }
    }
    os << "]";
}

/**
 * @brief Overload for standard output streams.
 */
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    auto impl = t.get_impl();
    if (!impl) {
        os << "Tensor(Uninitialized)";
        return os;
    }

    // Scalar fast-path
    if (impl->shape.empty() || (impl->size() == 1 && impl->shape.size() <= 1)) {
        os << "Tensor(" << impl->data_ptr[impl->offset] << ", dtype=Float32)";
        return os;
    }

    os << "Tensor(\n";
    std::vector<int> current_coords;
    current_coords.reserve(impl->shape.size());

    // Execute strided recursive printer
    print_recursive(os, *impl, current_coords, 0);

    os << ", dtype=Float32)";
    return os;
}

} // namespace isomorphism