/**
 * @file math.hpp
 * @brief Hardware-agnostic mathematical operations for isomorphism.
 * * This namespace provides a Domain Specific Language (DSL) for batched tensor
 * operations. These functions act as a routing layer: they take opaque
 * isomorphism::Tensor objects and translate the operation into highly optimized
 * native calls for the active backend (MLX on Apple, oneMKL/SYCL on PC).
 */

#pragma once

#include "isomorphism/tensor.hpp"
#include <vector>
#include <tuple>


namespace isomorphism::math {
    // ==============================================================================
    // 1. ELEMENT-WISE ARITHMETIC
    // ==============================================================================

    /** @brief Element-wise addition of two tensors. Supports broadcasting. */
    Tensor add(const Tensor &a, const Tensor &b);

    /** @brief Element-wise subtraction (a - b). Supports broadcasting. */
    Tensor subtract(const Tensor &a, const Tensor &b);

    /** @brief Element-wise multiplication. Used for scaling by beta, lambda, etc. */
    Tensor multiply(const Tensor &a, const Tensor &b);

    /** @brief Element-wise division (a / b). Supports broadcasting. */
    Tensor divide(const Tensor &a, const Tensor &b);

    /** @brief Element-wise floor function. */
    Tensor floor(const Tensor &a);

    Tensor mean(const Tensor &a);

    /** @brief Element-wise minimum of two tensors. Supports broadcasting. */
    Tensor minimum(const Tensor &a, const Tensor &b);

    /** @brief Element-wise ceiling: Returns the smallest integer greater than or equal to each element. */
    Tensor ceil(const Tensor &a);

    /** @brief Element-wise rounding: Rounds elements to the nearest integer. */
    Tensor round(const Tensor &a);

    /** * @brief Constraints tensor values to the range [min, max].
     * Essential for numerical stability and preventing NaN/Inf propagation.
     */
    Tensor clamp(const Tensor &a, float min, float max);

    // ==============================================================================
    // 2. CORE MATRIX OPERATIONS
    // ==============================================================================

    /** * @brief Batched matrix multiplication.
     * Evaluates A * B. If A is [N, d, d] and B is [N, d, d], performs N matmuls.
     */
    Tensor matmul(const Tensor &a, const Tensor &b);

    /** * @brief Swaps the last two dimensions of a tensor.
     * Essential for creating skew-symmetric Lie algebra matrices: 0.5 * (W - W^T).
     */
    Tensor transpose(const Tensor &a, const std::vector<int> &axes);

    /** * @brief Generates a 2D Identity matrix of size [d, d].
     * Can be broadcasted against batched tensors of shape [N, d, d].
     */
    Tensor eye(int d, DType dtype);

    /** * @brief Expands the dimensions of a tensor.
     * Essential for broadcasting. For example, expanding a [N] tensor of
     * norms to [N, 1, 1] so it can scale a batch of [N, d, d] matrices.
     */
    Tensor expand_dims(const Tensor &a, const std::vector<int> &axes);

    /** @brief Reshapes a tensor to a new shape. */
    Tensor reshape(const Tensor &a, const std::vector<int> &shape);

    /** @brief Broadcasts a tensor to a new target shape. */
    Tensor broadcast_to(const Tensor &a, const std::vector<int> &shape);

    /** @brief Creates a tensor from a raw C++ vector of floats. */
    Tensor array(const std::vector<float> &data, const std::vector<int> &shape, DType dtype);

    /** @brief Casts a tensor to Int32 for index lookups on the GPU. */
    Tensor astype_int32(const Tensor &a);

    /** @brief Gathers values along an axis based on an array of indices. */
    Tensor gather(const Tensor &a, const Tensor &indices, int axis = 0);

    /** * @brief Joins a sequence of tensors along a NEW axis.
     * Contrast with concatenate, which joins along an existing axis.
     */
    Tensor stack(const std::vector<Tensor> &tensors, int axis = 0);

    /** @brief Removes dimensions of size 1 from the tensor shape. */
    Tensor squeeze(const Tensor &a, const std::vector<int> &axes = {});

    /** @brief Creates a tensor of given shape filled with a constant value. */
    Tensor full(const std::vector<int> &shape, float value, DType dtype);

    // ==============================================================================
    // 3. LOGICAL OPERATIONS
    // ==============================================================================

    /** * @brief Element-wise conditional selection: (condition) ? x : y.
     * Vital for keeping branching logic on the GPU without a CPU synchronization event.
     */
    Tensor where(const Tensor &condition, const Tensor &x, const Tensor &y);

    /** @brief Element-wise equality comparison. */
    Tensor equal(const Tensor &a, const Tensor &b);

    /** @brief Element-wise inequality comparison. */
    Tensor not_equal(const Tensor &a, const Tensor &b);

    /** @brief Element-wise 'greater than' comparison. */
    Tensor greater(const Tensor &a, const Tensor &b);

    /** @brief Element-wise 'less than' comparison. */
    Tensor less(const Tensor &a, const Tensor &b);

    /** @brief Element-wise logical AND. */
    Tensor logical_and(const Tensor &a, const Tensor &b);

    /** @brief Element-wise logical OR. */
    Tensor logical_or(const Tensor &a, const Tensor &b);

    // ==============================================================================
    // 4. REDUCTIONS & NON-LINEARITIES
    // ==============================================================================

    /** * @brief Sums elements of a tensor along specified axes.
     * If axes is empty, sums all elements into a scalar tensor.
     */
    Tensor sum(const Tensor &a, const std::vector<int> &axes = {});

    /** @brief Returns the minimum element of the entire tensor as a scalar tensor. */
    Tensor min(const Tensor &a);

    /** @brief Element-wise exponential: exp(a). */
    Tensor exp(const Tensor &a);

    /** @brief Element-wise exponential: log(a). */
    Tensor log(const Tensor &a);

    /** @brief Element-wise square: a^2. Used for calculating Frobenius norms. */
    Tensor square(const Tensor &a);

    /** @brief Element-wise square root: sqrt(a). */
    Tensor sqrt(const Tensor &a);

    /** @brief Element-wise absolute value: |a|. Used as a numerical safety net before sqrt. */
    Tensor abs(const Tensor &a);

    /** @brief Element-wise sine: sin(a). Required for the geometric generator. */
    Tensor sin(const Tensor &a);

    /** @brief Element-wise cosine: cos(a). Required for the geometric generator and Ackley. */
    Tensor cos(const Tensor &a);

    /** @brief Element-wise inverse sine: asin(a). */
    Tensor asin(const Tensor &a);

    /** @brief Element-wise inverse cosine: acos(a). */
    Tensor acos(const Tensor &a);

    /** @brief Element-wise inverse tangent: atan(a). */
    Tensor atan(const Tensor &a);

    /** @brief Returns the index of the maximum element along an axis. */
    Tensor argmax(const Tensor &a, int axis = 0);

    /** @brief Returns the maximum element of the entire tensor as a scalar tensor. */
    Tensor max(const Tensor &a);

    /** @brief Computes the product of elements along specified axes. */
    Tensor prod(const Tensor &a, const std::vector<int> &axes = {});

    /** @brief Returns true if all elements along specified axes are non-zero. */
    Tensor all(const Tensor &a, const std::vector<int> &axes = {});

    /** @brief Returns true if any element along specified axes is non-zero. */
    Tensor any(const Tensor &a, const std::vector<int> &axes = {});

    /** @brief Element-wise power function: a^exponent. */
    Tensor pow(const Tensor &a, float exponent);

    /** @brief Element-wise tangent function. */
    Tensor tan(const Tensor &a);

    /** * @brief Element-wise four-quadrant inverse tangent.
     * Essential for robustly recovering angles in geometric and manifold generators.
     */
    Tensor atan2(const Tensor &y, const Tensor &x);

    // ==============================================================================
    // 5. HEAVY LINEAR ALGEBRA
    // ==============================================================================

    /** * @brief Solves the linear system AX = B for X.
     * Heavily utilized in the Cayley Transform to avoid matrix inversion.
     */
    Tensor solve(const Tensor &a, const Tensor &b);

    /** * @brief Batched Singular Value Decomposition (SVD).
     * @return A tuple of 3 tensors: {U, S, V_transpose}.
     * Required for projecting the ambient Fréchet mean back onto the SO(d) manifold.
     */
    std::tuple<Tensor, Tensor, Tensor> svd(const Tensor &a);

    /** * @brief Batched QR Decomposition.
     * @return A tuple of 2 tensors: {Q, R}.
     * The Q matrix represents the direct chordal projection of an ambient
     * matrix onto the orthogonal group in a single, non-iterative pass.
     */
    std::tuple<Tensor, Tensor> qr(const Tensor &a);

    /** * @brief Computes the determinant of a square matrix or a batch of square matrices.
     * @return A tensor containing the determinant(s).
     */
    Tensor det(const Tensor &a);

    /** @brief Computes the explicit matrix inverse. Prefer 'solve' for systems of equations. */
    Tensor inv(const Tensor &a);

    /** @brief Computes the sum of diagonal elements (the trace) of a matrix. */
    Tensor trace(const Tensor &a);

    // ==============================================================================
    // 6. STOCHASTIC GENERATION
    // ==============================================================================

    /** * @brief Generates a tensor filled with random normal (Gaussian) values.
     * Used for initializing particles and synthesizing anisotropic noise.
     */
    Tensor random_normal(const std::vector<int> &shape, DType dtype);

    /** @brief Generates a tensor filled with uniform random values in [0, 1). */
    Tensor random_uniform(const std::vector<int> &shape, DType dtype);

    // ==============================================================================
    // 7. CPU-GPU BRIDGE
    // ==============================================================================


    /** * @brief Pulls a scalar 0D or 1D tensor back to the CPU as a standard C++ double.
     * @warning This triggers a synchronization event between the CPU and GPU.
     * Use sparingly (e.g., only once per step for checking convergence energies).
     */
    double to_double(const Tensor &a);

    std::vector<float> to_float_vector(const Tensor &a);

    /** @brief Explicitly executes the pending computation graph for this tensor. */
    void eval(const Tensor &a);

    /** @brief Concatenates a vector of tensors along a specified axis. */
    Tensor concatenate(const std::vector<Tensor> &tensors, int axis = 0);

    /** @brief Slices a tensor along a specified axis from start to end indices. */
    Tensor slice(const Tensor &a, int start, int end, int axis = 0);

    /** * @brief Pulls a scalar integer tensor back to the CPU as a standard C++ int.
     * Essential for index-based operations like selecting the global best particle.
     */
    int to_int(const Tensor &a);
} // namespace isomorphism::math
