/**
 * @file tensor_impl_eigen.hpp
 * @brief Private implementation of the Tensor handle for the Eigen CPU backend.
 */

#pragma once

#include "isomorphism/tensor.hpp"
#include "memory_pool.hpp"

#include <vector>
#include <memory>
#include <numeric>

namespace isomorphism {

/**
 * @brief The internal "Pimpl" for isomorphism::Tensor on CPU.
 * * Uses a MemoryPool to avoid the performance hindrance of frequent allocations.
 * * Supports O(1) slicing and transposing via strides and offsets.
 */
struct TensorImpl {
    // Shared pointer to the raw float buffer, managed by the MemoryPool
    std::shared_ptr<float[]> data_ptr;

    std::vector<int> shape;
    std::vector<int> strides;
    int offset = 0;

    // Metadata for FFT and SVD operations
    bool is_complex = false;

    /**
     * @brief Standard constructor for NEW tensors.
     * Requests memory from the thread-safe pool instead of the system heap.
     */
    TensorImpl(const std::vector<int>& s, bool complex = false)
        : shape(s), is_complex(complex), offset(0)
    {
        size_t total_elements = 1;
        for (int d : shape) total_elements *= d;

        // If complex, we need 2 floats per element (real, imag)
        size_t alloc_size = is_complex ? total_elements * 2 : total_elements;

        // Acquire from pool to bypass "Allocation Overhead"
        data_ptr = math::MemoryPool::instance().acquire(alloc_size);
        strides = compute_contiguous_strides(shape);
    }

    /**
     * @brief Zero-Copy view constructor.
     * Used for O(1) operations like slice, transpose, and reshape.
     */
    TensorImpl(std::shared_ptr<float[]> ptr,
               std::vector<int> s,
               std::vector<int> st,
               int off,
               bool complex = false)
        : data_ptr(ptr), shape(s), strides(st), offset(off), is_complex(complex) {}

    /**
     * @brief Returns the total number of logical elements in the tensor.
     */
    size_t size() const {
        if (shape.empty()) return 0;
        size_t s = 1;
        for (int d : shape) s *= d;
        return s;
    }

    /**
     * @brief Checks if the memory layout is contiguous (no gaps or non-standard strides).
     * Used by math_eigen.cpp to determine if a copy is required before certain operations.
     */
    bool is_contiguous() const {
        auto expected = compute_contiguous_strides(shape);
        return strides == expected;
    }

    /**
     * @brief Static helper to calculate Row-Major contiguous strides.
     */
    static std::vector<int> compute_contiguous_strides(const std::vector<int>& shape) {
        int ndim = static_cast<int>(shape.size());
        if (ndim == 0) return {};

        std::vector<int> st(ndim);
        int current_stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            st[i] = current_stride;
            current_stride *= shape[i];
        }
        return st;
    }
};

} // namespace isomorphism