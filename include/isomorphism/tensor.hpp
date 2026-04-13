/**
 * @file tensor.hpp
 * @brief The core data structure for batched manifold optimization.
 * * This file defines the `Tensor` class using the Pimpl (Pointer to Implementation)
 * idiom. It acts as an opaque handle to the underlying hardware memory
 * (Apple MLX `mlx::core::array`, PC SYCL `sycl::buffer`, or Eigen CPU buffers).
 * * Because it uses `std::shared_ptr` internally, copying a `Tensor` object
 * is virtually free—it simply increments a reference count. Deep copies of
 * memory are strictly avoided unless explicitly requested.
 */

#pragma once

#include <memory>
#include <vector>
#include <ostream>

namespace isomorphism {

    /**
     * @enum DType
     * @brief Represents the data type of the underlying tensor memory.
     */
    enum class DType {
        Float32,
        Float16,
        BFloat16
    };

    /**
     * @brief Forward declaration of the internal implementation struct.
     * The actual definition lives exclusively inside backend-specific files
     * (e.g., `tensor_impl_eigen.hpp`).
     */
    struct TensorImpl;

    /**
     * @class Tensor
     * @brief Hardware-agnostic wrapper for multi-dimensional arrays.
     * * @example
     * // Creating a scalar tensor for math operations
     * isomorphism::Tensor beta_tensor(20.0);
     * * // Introspecting a tensor returned from a solver
     * std::vector<int> s = my_tensor.shape();
     * std::cout << "Dimensions: " << my_tensor.ndim() << "\n";
     */
    class Tensor {
    private:
        /** * @brief The opaque pointer hiding the hardware-specific array.
         * We use shared_ptr so that passing Tensors by value into math functions
         * does not trigger massive memory copies.
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
        explicit Tensor(double scalar_value, DType dtype = DType::Float32);

        /**
         * @brief Unified constructor used by backend implementations (MLX, Eigen).
         * This constructor allows the math layer to wrap existing hardware-specific
         * implementations into the public Tensor handle.
         */
        explicit Tensor(std::shared_ptr<TensorImpl> pimpl, DType dtype = DType::Float32);

        /** @brief Destructor. Automatically frees memory when ref count hits 0. */
        ~Tensor();

        // ==============================================================================
        // COPY & MOVE SEMANTICS
        // ==============================================================================

        // Defaulting these relies on std::shared_ptr's built-in reference counting.
        Tensor(const Tensor &other) = default;
        Tensor &operator=(const Tensor &other) = default;
        Tensor(Tensor &&other) noexcept = default;
        Tensor &operator=(Tensor &&other) noexcept = default;

        // ==============================================================================
        // INTROSPECTION & UTILITY
        // ==============================================================================

        /** @brief Returns the dimensions of the tensor (e.g., {80000, 100, 100}). */
        std::vector<int> shape() const;

        /** @brief Returns the number of dimensions (e.g., 3 for [N, d, d]). */
        int ndim() const;

        /** @brief Returns the total number of individual elements in the tensor. */
        int size() const;

        /** @brief Returns the data type of the tensor. */
        DType dtype() const {
            return dtype_;
        }

        /** * @brief Accessor for the backend to retrieve the raw implementation object.
         * Used by functions inside `isomorphism::math` to unwrap the tensor before math.
         */
        std::shared_ptr<TensorImpl> get_impl() const { return pimpl_; }
    };

    /**
     * @brief Overload for standard output streams to print the underlying tensor data.
     */
    std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

} // namespace isomorphism