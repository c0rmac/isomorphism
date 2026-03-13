/**
 * @file tensor.hpp
 * @brief The core data structure for batched manifold optimization.
 * * This file defines the `Tensor` class using the Pimpl (Pointer to Implementation)
 * idiom. It acts as an opaque handle to the underlying hardware memory
 * (Apple MLX `mlx::core::array` or PC SYCL `sycl::buffer`).
 * * Because it uses `std::shared_ptr` internally, copying a `Tensor` object
 * is virtually free—it simply increments a reference count. Deep copies of
 * GPU memory are strictly avoided unless explicitly requested.
 */

#pragma once

#include <memory>
#include <vector>
#include <ostream>

namespace involute {
    /**
 * @enum DType
 * @brief Represents the data type of the underlying tensor memory.
 */
    enum class DType {
        Float32,
        Float16,
        BFloat16
    };

    // Forward declaration of the internal implementation struct.
    // The actual definition lives exclusively inside `src/backends/mlx/tensor_mlx.cpp`
    // or `src/backends/sycl/tensor_sycl.cpp`.
    struct TensorImpl;

    /**
     * @class Tensor
     * @brief Hardware-agnostic wrapper for multi-dimensional arrays.
     * * @example
     * // Creating a scalar tensor for math operations
     * involute::Tensor beta_tensor(20.0);
     * * // Introspecting a tensor returned from a solver
     * std::vector<int> s = my_tensor.shape();
     * std::cout << "Dimensions: " << my_tensor.ndim() << "\n";
     */
    class Tensor {
    private:
        /** * @brief The opaque pointer hiding the hardware-specific array.
         * We use shared_ptr so that passing Tensors by value into math functions
         * does not trigger massive GPU-to-GPU memory copies.
         */
        std::shared_ptr<TensorImpl> pimpl_;
        DType dtype_;

    public:
        // ==============================================================================
        // CONSTRUCTORS & DESTRUCTOR
        // ==============================================================================

        /** @brief Default constructor creating an empty/null tensor. */
        Tensor();

        /** * @brief Constructs a 0D (scalar) tensor from a double.
         * Essential for broadcasting scalars in operations like exp(-beta * costs).
         */
        explicit Tensor(double scalar_value, DType dtype);

        /** @brief Destructor. Automatically frees GPU memory when ref count hits 0. */
        ~Tensor();

        // ==============================================================================
        // COPY & MOVE SEMANTICS
        // ==============================================================================

        // Defaulting these relies on std::shared_ptr's built-in reference counting.
        Tensor(const Tensor &other) = default;

        Tensor &operator=(const Tensor &other) = default;

        Tensor(Tensor &&other) noexcept = default;

        Tensor &operator=(Tensor &&other) noexcept = default;

        /**
         * @brief Advanced constructor used exclusively by the backend implementation.
         * Users should never need to call this directly.
         */
        explicit Tensor(std::shared_ptr<TensorImpl> impl) : pimpl_(std::move(impl)) {
        }

        // ==============================================================================
        // INTROSPECTION & UTILITY
        // ==============================================================================

        /** @brief Returns the dimensions of the tensor (e.g., {80000, 100, 100}). */
        std::vector<int> shape() const;

        /** @brief Returns the number of dimensions (e.g., 3 for [N, d, d]). */
        int ndim() const;

        /** @brief Returns the total number of individual elements in the tensor. */
        int size() const;

        DType dtype() const {
            return dtype_;
        }

        /** * @brief Accessor for the backend to retrieve the raw MLX/SYCL object.
         * Used by functions inside `involute::math` to unwrap the tensor before math.
         */
        std::shared_ptr<TensorImpl> get_impl() const { return pimpl_; }
    };

    /**
         * @brief Overload for standard output streams to print the underlying tensor data.
         */
    std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
} // namespace involute
