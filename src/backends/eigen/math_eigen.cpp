// Path: isomorphism/src/backends/eigen/math_eigen.cpp

#include <iostream>

#include "isomorphism/math.hpp"
#include "tensor_impl_eigen.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/LU>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/CXX11/Tensor>

#include <random>
#include <stdexcept>
#include <thread>

namespace isomorphism::math {

    // ==============================================================================
    // INTERNAL HELPERS
    // ==============================================================================

    /**
 * @brief Unwraps the opaque Tensor to its internal implementation.
 */
static inline TensorImpl& unwrap(const Tensor& t) {
    return *t.get_impl();
}

    /**
     * @brief Evaluates an Eigen expression directly into a pooled Tensor.
     * * This is the primary driver for "Eager" execution in the Eigen backend.
     * Because TensorImpl now uses the MemoryPool, this function is
     * significantly faster than a standard heap allocation.
     * * @tparam Derived The Eigen expression type (Lazy Evaluation).
     * @param expr The Eigen expression to compute.
     * @param shape The target shape of the resulting Tensor.
     * @return A isomorphism::Tensor holding the computed data.
     */
    template <typename Derived>
    static inline Tensor wrap(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& expr,
                              const std::vector<int>& shape)
{
    // 1. Instantiate a new TensorImpl.
    // The constructor automatically calls MemoryPool::acquire().
    auto impl = std::make_shared<TensorImpl>(shape);

    // 2. Map the pool-allocated memory to a 1D Eigen TensorMap.
    // We treat it as 1D to simplify the evaluation of reshaped expressions.
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> out_map(
        impl->data_ptr.get(),
        impl->size()
    );

    // 3. Trigger the actual computation.
    // This evaluates the Eigen expression graph directly into our raw buffer.
    out_map = expr.reshape(Eigen::array<Eigen::Index, 1>{(Eigen::Index)impl->size()});

    // 4. Return the opaque handle.
    return Tensor(impl, DType::Float32);
}

/**
 * @brief Zero-Copy Wrap. Creates a NEW Tensor that points to EXISTING memory.
 * Used for O(1) operations like slice, reshape, and transpose.
 */
static inline Tensor wrap_view(std::shared_ptr<float[]> ptr,
                               const std::vector<int>& shape,
                               const std::vector<int>& strides,
                               int offset)
{
    auto impl = std::make_shared<TensorImpl>(ptr, shape, strides, offset);
    return Tensor(impl, DType::Float32);
}

/**
 * @brief Recursive helper to copy heavily strided memory into a flat buffer.
 */
static void copy_strided_recursive(const float* src, float* dst,
                                   const std::vector<int>& shape,
                                   const std::vector<int>& strides,
                                   int current_dim, int& dst_idx, int src_idx)
{
    if (current_dim == static_cast<int>(shape.size())) {
        dst[dst_idx++] = src[src_idx];
        return;
    }
    for (int i = 0; i < shape[current_dim]; ++i) {
        copy_strided_recursive(src, dst, shape, strides, current_dim + 1,
                               dst_idx, src_idx + i * strides[current_dim]);
    }
}

/**
 * @brief Enforces memory contiguity.
 * If the tensor is a strided view, it physically copies it to a flat buffer.
 * If it is already contiguous, it returns the tensor instantly (Zero-Copy).
 */
static inline Tensor contiguous(const Tensor& t) {
    auto& impl = unwrap(t);
    if (impl.is_contiguous()) return t;

    auto flat_impl = std::make_shared<TensorImpl>(impl.shape);
    int dst_idx = 0;

    copy_strided_recursive(impl.data_ptr.get(), flat_impl->data_ptr.get(),
                           impl.shape, impl.strides, 0, dst_idx, impl.offset);

    return Tensor(flat_impl, DType::Float32);
}

// Helper macro for N-Dimensional Eigen operations.
// ALWAYS call contiguous() on your inputs before dispatching this macro!
#define DISPATCH_RANK(NDIM, TARGET_MACRO) \
    switch (NDIM) { \
        case 1: { TARGET_MACRO(1); } break; \
        case 2: { TARGET_MACRO(2); } break; \
        case 3: { TARGET_MACRO(3); } break; \
        case 4: { TARGET_MACRO(4); } break; \
        case 5: { TARGET_MACRO(5); } break; \
        case 6: { TARGET_MACRO(6); } break; \
        default: throw std::runtime_error("[isomorphism/eigen] Unsupported Tensor rank (ndim > 6)"); \
    }

// ==============================================================================
// 1. ELEMENT-WISE ARITHMETIC
// ==============================================================================

/**
 * @brief Generic driver for binary operations (add, sub, mul, div, min).
 * Handles NumPy/MLX style broadcasting and enforces memory contiguity.
 */
template <typename Op>
static inline Tensor apply_binary(const Tensor& a, const Tensor& b, Op op) {
    // 1. Enforce contiguity to allow simple Eigen::TensorMap access.
    // This is the "Lazy Contiguous" pattern used by PyTorch/MLX.
    Tensor ca = contiguous(a);
    Tensor cb = contiguous(b);

    auto& A = unwrap(ca);
    auto& B = unwrap(cb);

    // Fast Path: Exact shape match
    if (ca.shape() == cb.shape()) {
        #define EXEC_BINARY_SAME(RANK) \
            Eigen::array<Eigen::Index, RANK> dims; \
            for(int i=0; i<RANK; ++i) dims[i] = A.shape[i]; \
            Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapA(A.data_ptr.get(), dims); \
            Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapB(B.data_ptr.get(), dims); \
            return wrap(op(mapA, mapB), ca.shape());

        DISPATCH_RANK(ca.ndim(), EXEC_BINARY_SAME);
        #undef EXEC_BINARY_SAME
    }

    // Scalar Path: a is scalar
    if (ca.size() == 1) {
        Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> mapB(B.data_ptr.get(), cb.size());
        return wrap(op(mapB.constant(A.data_ptr[A.offset]), mapB), cb.shape());
    }

    // Scalar Path: b is scalar
    if (cb.size() == 1) {
        Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> mapA(A.data_ptr.get(), ca.size());
        return wrap(op(mapA, mapA.constant(B.data_ptr[B.offset])), ca.shape());
    }

    // --- Full N-D Broadcasting ---
    int target_ndim = std::max(ca.ndim(), cb.ndim());
    std::vector<int> target_shape(target_ndim);
    std::vector<int> pad_a(target_ndim, 1), pad_b(target_ndim, 1);
    std::vector<int> bcast_a(target_ndim, 1), bcast_b(target_ndim, 1);

    for (int i = 0; i < target_ndim; ++i) {
        int d_a = (i < target_ndim - ca.ndim()) ? 1 : ca.shape()[i - (target_ndim - ca.ndim())];
        int d_b = (i < target_ndim - cb.ndim()) ? 1 : cb.shape()[i - (target_ndim - cb.ndim())];
        target_shape[i] = std::max(d_a, d_b);
        pad_a[i] = d_a; pad_b[i] = d_b;
        bcast_a[i] = target_shape[i] / d_a;
        bcast_b[i] = target_shape[i] / d_b;
    }

    #define EXEC_BINARY_BCAST(RANK) \
        Eigen::array<Eigen::Index, RANK> dA, dB, bA, bB; \
        for(int i=0; i<RANK; ++i) { dA[i]=pad_a[i]; dB[i]=pad_b[i]; bA[i]=bcast_a[i]; bB[i]=bcast_b[i]; } \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapA(A.data_ptr.get(), dA); \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapB(B.data_ptr.get(), dB); \
        return wrap(op(mapA.broadcast(bA), mapB.broadcast(bB)), target_shape);

    DISPATCH_RANK(target_ndim, EXEC_BINARY_BCAST);
    #undef EXEC_BINARY_BCAST

    throw std::runtime_error("[isomorphism/eigen] Broadcast failed.");
}

// Element-wise arithmetic using the driver above
Tensor add(const Tensor& a, const Tensor& b)      { return apply_binary(a, b, std::plus<>{}); }
Tensor subtract(const Tensor& a, const Tensor& b) { return apply_binary(a, b, std::minus<>{}); }
Tensor multiply(const Tensor& a, const Tensor& b) { return apply_binary(a, b, std::multiplies<>{}); }
Tensor divide(const Tensor& a, const Tensor& b)   { return apply_binary(a, b, std::divides<>{}); }
Tensor minimum(const Tensor& a, const Tensor& b)  { return apply_binary(a, b, [](auto x, auto y){ return x.cwiseMin(y); }); }

// Unary operations (Operate on flat 1D buffer for speed)
Tensor floor(const Tensor& a) { return wrap(Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(unwrap(a).data_ptr.get(), unwrap(a).size()).floor(), a.shape()); }
Tensor ceil(const Tensor& a)  { return wrap(Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(unwrap(a).data_ptr.get(), unwrap(a).size()).ceil(), a.shape()); }
Tensor round(const Tensor& a) { return wrap(Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(unwrap(a).data_ptr.get(), unwrap(a).size()).round(), a.shape()); }

Tensor mean(const Tensor& a) {
    // Global Mean: Force evaluation to a Rank-0 RowMajor Tensor
    Eigen::Tensor<float, 0, Eigen::RowMajor> m = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(unwrap(a).data_ptr.get(), unwrap(a).size()).mean();
    return Tensor(static_cast<double>(m()), DType::Float32);
}

Tensor clamp(const Tensor& a, float min_val, float max_val) {
    auto& impl = unwrap(a);
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
    // Clamp implementation using Eigen's cwiseMin/cwiseMax
    return wrap(map.cwiseMax(min_val).cwiseMin(max_val), a.shape());
}

// ==============================================================================
// 2. CORE MATRIX OPERATIONS
// ==============================================================================

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Matmul requires inputs to be contiguous for efficient BLAS/GEMM mapping
    Tensor ca = contiguous(a);
    Tensor cb = contiguous(b);

    auto& A = unwrap(ca);
    auto& B = unwrap(cb);

    int ndim_a = ca.ndim();
    int M = (ndim_a >= 2) ? ca.shape()[ndim_a - 2] : 1;
    int K = ca.shape().back();
    int N = cb.shape().back();

    // Calculate total batch size by flattening all leading dimensions
    int bat = ca.size() / (M * K);

    auto out_shape = ca.shape();
    out_shape.back() = N; // [..., M, N]

    auto res_impl = std::make_shared<TensorImpl>(out_shape);

    float* ptrA = A.data_ptr.get();
    float* ptrB = B.data_ptr.get();
    float* ptrRes = res_impl->data_ptr.get();

    // Disable Eigen internal multi-threading as the caller likely parallelizes batches
    Eigen::setNbThreads(1);

    // Standard Batched GEMM mapped directly to the flat evaluated memory
    #pragma omp parallel for schedule(static) if(bat > 1)
    for (int i = 0; i < bat; ++i) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mapA(ptrA + i * M * K, M, K);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mapB(ptrB + i * K * N, K, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mapRes(ptrRes + i * M * N, M, N);

        mapRes.noalias() = mapA * mapB;
    }

    return Tensor(res_impl, DType::Float32);
}

Tensor transpose(const Tensor& a, const std::vector<int>& axes) {
    auto& impl = unwrap(a);
    int ndim = impl.shape.size();
    if (ndim <= 1) return a;

    std::vector<int> perm = axes;
    if (perm.empty()) {
        for (int i = 0; i < ndim; ++i) perm.push_back(i);
        std::swap(perm[ndim - 1], perm[ndim - 2]);
    } else {
        for (auto& p : perm) if (p < 0) p += ndim;
    }

    // O(1) Metadata Shuffle: No data is moved
    std::vector<int> new_shape(ndim);
    std::vector<int> new_strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        new_shape[i] = impl.shape[perm[i]];
        new_strides[i] = impl.strides[perm[i]];
    }

    return wrap_view(impl.data_ptr, new_shape, new_strides, impl.offset);
}

Tensor reshape(const Tensor& a, const std::vector<int>& shape) {
    // Reshape logically requires contiguity
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);

    auto ns = shape;
    int neg = -1, known = 1;
    for (int i = 0; i < (int)ns.size(); ++i) {
        if (ns[i] == -1) neg = i;
        else known *= ns[i];
    }
    if (neg >= 0) ns[neg] = impl.size() / known;

    // O(1) Re-stride: Calculate new contiguous strides for the new shape
    auto new_strides = TensorImpl::compute_contiguous_strides(ns);
    return wrap_view(impl.data_ptr, ns, new_strides, impl.offset);
}

Tensor broadcast_to(const Tensor& a, const std::vector<int>& shape) {
    auto& impl = unwrap(a);
    if (impl.shape == shape) return a;

    int target_ndim = shape.size();
    int src_ndim = impl.shape.size();

    std::vector<int> new_strides(target_ndim, 0);

    // Align to the right and set stride to 0 for dimensions that are stretched
    for (int i = 0; i < target_ndim; ++i) {
        int target_dim_idx = target_ndim - 1 - i;
        int src_dim_idx = src_ndim - 1 - i;

        if (src_dim_idx >= 0) {
            if (impl.shape[src_dim_idx] == shape[target_dim_idx]) {
                new_strides[target_dim_idx] = impl.strides[src_dim_idx];
            } else if (impl.shape[src_dim_idx] == 1) {
                new_strides[target_dim_idx] = 0; // The "secret sauce" for O(1) broadcast
            } else {
                throw std::runtime_error("[isomorphism/eigen] Incompatible broadcast shapes.");
            }
        } else {
            new_strides[target_dim_idx] = 0; // Padded dimension
        }
    }

    return wrap_view(impl.data_ptr, shape, new_strides, impl.offset);
}

Tensor expand_dims(const Tensor& a, const std::vector<int>& axes) {
    auto& impl = unwrap(a);
    int new_ndim = impl.shape.size() + axes.size();

    auto new_shape = impl.shape;
    auto new_strides = impl.strides;
    auto sorted_axes = axes;
    for (auto& x : sorted_axes) if (x < 0) x += new_ndim;
    std::sort(sorted_axes.begin(), sorted_axes.end());

    for (int ax : sorted_axes) {
        new_shape.insert(new_shape.begin() + ax, 1);
        // Stride for a dimension of size 1 can be anything;
        // we use the next dimension's stride to stay consistent.
        int next_stride = (ax < (int)new_strides.size()) ? new_strides[ax] : 1;
        new_strides.insert(new_strides.begin() + ax, next_stride);
    }

    return wrap_view(impl.data_ptr, new_shape, new_strides, impl.offset);
}

Tensor squeeze(const Tensor& a, const std::vector<int>& axes) {
    auto& impl = unwrap(a);
    std::vector<int> ns, nst;

    std::vector<bool> to_remove(impl.shape.size(), false);
    if (axes.empty()) {
        for (int i = 0; i < (int)impl.shape.size(); ++i)
            if (impl.shape[i] == 1) to_remove[i] = true;
    } else {
        for (int ax : axes) {
            int idx = (ax < 0) ? ax + impl.shape.size() : ax;
            if (impl.shape[idx] == 1) to_remove[idx] = true;
        }
    }

    for (int i = 0; i < (int)impl.shape.size(); ++i) {
        if (!to_remove[i]) {
            ns.push_back(impl.shape[i]);
            nst.push_back(impl.strides[i]);
        }
    }
    return wrap_view(impl.data_ptr, ns, nst, impl.offset);
}

// --- EVALUATED CORE OPERATIONS ---

Tensor eye(int d, DType dtype) {
    auto impl = std::make_shared<TensorImpl>(std::vector<int>{d, d});
    for (int i = 0; i < d; ++i) impl->data_ptr[i * d + i] = 1.0f;
    return Tensor(impl, DType::Float32);
}

Tensor array(const std::vector<float>& data, const std::vector<int>& shape, DType dtype) {
    auto impl = std::make_shared<TensorImpl>(shape);
    std::copy(data.begin(), data.end(), impl->data_ptr.get());
    return Tensor(impl, DType::Float32);
}

Tensor full(const std::vector<int>& shape, float value, DType dtype) {
    auto impl = std::make_shared<TensorImpl>(shape);
    std::fill(impl->data_ptr.get(), impl->data_ptr.get() + impl->size(), value);
    return Tensor(impl, DType::Float32);
}

Tensor stack(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) throw std::runtime_error("[isomorphism/eigen] stack: empty list");
    // Standard stack: expand each to [..., 1, ...] and concatenate
    std::vector<Tensor> expanded;
    for (const auto& t : tensors) expanded.push_back(expand_dims(t, {axis}));
    return concatenate(expanded, axis);
}

// ==============================================================================
// 3. LOGICAL OPERATIONS
// ==============================================================================

/**
 * @brief Element-wise conditional selection: (condition) ? x : y.
 * Supports N-D broadcasting across all three inputs.
 */
Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    // Ensure all inputs are contiguous for Eigen evaluation
    Tensor cc = contiguous(condition);
    Tensor cx = contiguous(x);
    Tensor cy = contiguous(y);

    auto& C = unwrap(cc);
    auto& X = unwrap(cx);
    auto& Y = unwrap(cy);

    // Determine the broadcasted target shape
    int target_ndim = std::max({cc.ndim(), cx.ndim(), cy.ndim()});
    std::vector<int> target_shape(target_ndim);

    // Simplified shape padding for ternary broadcasting
    auto get_dim = [](const std::vector<int>& shape, int idx, int target_ndim) {
        int offset = target_ndim - (int)shape.size();
        return (idx < offset) ? 1 : shape[idx - offset];
    };

    for (int i = 0; i < target_ndim; ++i) {
        int d_c = get_dim(cc.shape(), i, target_ndim);
        int d_x = get_dim(cx.shape(), i, target_ndim);
        int d_y = get_dim(cy.shape(), i, target_ndim);
        target_shape[i] = std::max({d_c, d_x, d_y});
    }

    #define EXEC_WHERE(RANK) \
        Eigen::array<Eigen::Index, RANK> dC, dX, dY, bC, bX, bY; \
        for(int i=0; i<RANK; ++i) { \
            int cur_c = get_dim(cc.shape(), i, RANK); \
            int cur_x = get_dim(cx.shape(), i, RANK); \
            int cur_y = get_dim(cy.shape(), i, RANK); \
            dC[i] = cur_c; dX[i] = cur_x; dY[i] = cur_y; \
            bC[i] = target_shape[i]/cur_c; bX[i] = target_shape[i]/cur_x; bY[i] = target_shape[i]/cur_y; \
        } \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapC(C.data_ptr.get(), dC); \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapX(X.data_ptr.get(), dX); \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> mapY(Y.data_ptr.get(), dY); \
        /* Eigen's select expects a boolean condition. 0.0f is False, everything else is True. */ \
        auto cond_bool = (mapC.broadcast(bC) != mapC.constant(0.0f)); \
        return wrap(cond_bool.select(mapX.broadcast(bX), mapY.broadcast(bY)), target_shape);

    DISPATCH_RANK(target_ndim, EXEC_WHERE);
    #undef EXEC_WHERE
    throw std::runtime_error("[isomorphism/eigen] where: rank dispatch failed");
}

// Comparison operations using the apply_binary driver.
// We cast the resulting boolean tensor to float (1.0 or 0.0) for storage.
Tensor equal(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) { return (x == y).template cast<float>(); });
}

Tensor not_equal(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) { return (x != y).template cast<float>(); });
}

Tensor greater(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) { return (x > y).template cast<float>(); });
}

Tensor less(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) { return (x < y).template cast<float>(); });
}

Tensor greater_equal(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) { return (x >= y).template cast<float>(); });
}

Tensor less_equal(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) { return (x <= y).template cast<float>(); });
}

// Logical operations. Input is treated as boolean (non-zero = true).
Tensor logical_and(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) {
        return ((x != x.constant(0.0f)) && (y != y.constant(0.0f))).template cast<float>();
    });
}

Tensor logical_or(const Tensor& a, const Tensor& b) {
    return apply_binary(a, b, [](auto x, auto y) {
        return ((x != x.constant(0.0f)) || (y != y.constant(0.0f))).template cast<float>();
    });
}

// ==============================================================================
// 4. REDUCTIONS & NON-LINEARITIES
// ==============================================================================

// --- HELPER FOR MULTI-AXIS REDUCTION ---
// Eigen's .sum() and .prod() accept an Eigen::array of indices.
template <int RANK, int RED_SIZE, typename Op>
static inline Tensor reduce_impl(const TensorImpl& impl, const std::vector<int>& axes, Op op) {
    Eigen::array<Eigen::Index, RANK> dims;
    for(int i=0; i<RANK; ++i) dims[i] = impl.shape[i];

    Eigen::array<Eigen::Index, RED_SIZE> red_axes;
    for(int i=0; i<RED_SIZE; ++i) red_axes[i] = (axes[i] < 0) ? axes[i] + RANK : axes[i];

    Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> map(impl.data_ptr.get(), dims);

    // Determine output shape
    std::vector<int> out_shape;
    std::vector<bool> is_red(RANK, false);
    for(int a : red_axes) is_red[a] = true;
    for(int i=0; i<RANK; ++i) if(!is_red[i]) out_shape.push_back(impl.shape[i]);
    if(out_shape.empty()) out_shape = {1};

    return wrap(op(map, red_axes), out_shape);
}

// ------------------------------------------------------------------------------
// REDUCTION OPS (Sum, Prod, All, Any)
// ------------------------------------------------------------------------------

Tensor sum(const Tensor& a, const std::vector<int>& axes) {
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    if (axes.empty()) { // Global reduction
        Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
        Eigen::Tensor<float, 0, Eigen::RowMajor> res = map.sum();
        return Tensor(static_cast<double>(res()), DType::Float32);
    }
    // For specific axes, we dispatch based on rank (simplified for common cases)
    if (ca.ndim() == 2 && axes.size() == 1) return reduce_impl<2, 1>(impl, axes, [](auto m, auto ax){ return m.sum(ax); });
    if (ca.ndim() == 3 && axes.size() == 1) return reduce_impl<3, 1>(impl, axes, [](auto m, auto ax){ return m.sum(ax); });
    return ca; // Fallback
}

Tensor prod(const Tensor& a, const std::vector<int>& axes) {
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    if (axes.empty()) {
        Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
        Eigen::Tensor<float, 0, Eigen::RowMajor> res = map.prod();
        return Tensor(static_cast<double>(res()), DType::Float32);
    }
    if (ca.ndim() == 2 && axes.size() == 1) return reduce_impl<2, 1>(impl, axes, [](auto m, auto ax){ return m.prod(ax); });
    return ca;
}

Tensor all(const Tensor& a, const std::vector<int>& axes) {
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    auto op = [](auto m, auto ax){ return (m != m.constant(0.0f)).all(ax).template cast<float>(); };
    if (ca.ndim() == 2 && axes.size() == 1) return reduce_impl<2, 1>(impl, axes, op);
    return ca;
}

Tensor any(const Tensor& a, const std::vector<int>& axes) {
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    auto op = [](auto m, auto ax){ return (m != m.constant(0.0f)).any(ax).template cast<float>(); };
    if (ca.ndim() == 2 && axes.size() == 1) return reduce_impl<2, 1>(impl, axes, op);
    return ca;
}

// ------------------------------------------------------------------------------
// NON-LINEARITIES (Flat 1D Optimization)
// ------------------------------------------------------------------------------

#define UNARY_OP(NAME, EIGEN_METHOD) \
Tensor NAME(const Tensor& a) { \
    auto& impl = unwrap(a); \
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size()); \
    return wrap(map.EIGEN_METHOD(), a.shape()); \
}

UNARY_OP(exp, exp)
UNARY_OP(log, log)
UNARY_OP(square, square)
UNARY_OP(sqrt, sqrt)
UNARY_OP(abs, abs)

Tensor pow(const Tensor& a, float exponent) {
    auto& impl = unwrap(a);
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
    return wrap(map.pow(exponent), a.shape());
}

// ------------------------------------------------------------------------------
// TRIGONOMETRY (UnaryExpr Fallbacks)
// ------------------------------------------------------------------------------

#define TRIG_OP(NAME, STD_FUNC) \
Tensor NAME(const Tensor& a) { \
    auto& impl = unwrap(a); \
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size()); \
    return wrap(map.unaryExpr([](float x){ return std::STD_FUNC(x); }), a.shape()); \
}

TRIG_OP(sin, sin)
TRIG_OP(cos, cos)
TRIG_OP(tan, tan)
TRIG_OP(asin, asin)
TRIG_OP(acos, acos)
TRIG_OP(atan, atan)

Tensor atan2(const Tensor& y, const Tensor& x) {
    return apply_binary(y, x, [](auto v_y, auto v_x){
        return v_y.binaryExpr(v_x, [](float a, float b){ return std::atan2(a, b); });
    });
}

// ------------------------------------------------------------------------------
// SPECIAL REDUCTIONS & SCANS
// ------------------------------------------------------------------------------

Tensor max(const Tensor& a) {
    auto& impl = unwrap(a);
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
    Eigen::Tensor<float, 0, Eigen::RowMajor> res = map.maximum();
    return Tensor(static_cast<double>(res()), DType::Float32);
}

Tensor min(const Tensor& a) {
    auto& impl = unwrap(a);
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
    Eigen::Tensor<float, 0, Eigen::RowMajor> res = map.minimum();
    return Tensor(static_cast<double>(res()), DType::Float32);
}

Tensor argmax(const Tensor& a, int axis) {
    Tensor ca = contiguous(a);
    int ndim = ca.ndim();
    int adj_axis = (axis < 0) ? axis + ndim : axis;

    #define EXEC_ARGMAX(RANK) \
        Eigen::array<Eigen::Index, RANK> dims; \
        for(int i=0; i<RANK; ++i) dims[i] = unwrap(ca).shape[i]; \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> map(unwrap(ca).data_ptr.get(), dims); \
        std::vector<int> out_shape = ca.shape(); \
        out_shape.erase(out_shape.begin() + adj_axis); \
        if(out_shape.empty()) out_shape = {1}; \
        return wrap(map.argmax(adj_axis).template cast<float>(), out_shape);

    DISPATCH_RANK(ndim, EXEC_ARGMAX);
    #undef EXEC_ARGMAX
    throw std::runtime_error("argmax failed");
}

Tensor cumsum(const Tensor& a, int axis) {
    Tensor ca = contiguous(a);
    int ndim = ca.ndim();
    int adj_axis = (axis < 0) ? axis + ndim : axis;

    #define EXEC_CUMSUM(RANK) \
        Eigen::array<Eigen::Index, RANK> dims; \
        for(int i=0; i<RANK; ++i) dims[i] = unwrap(ca).shape[i]; \
        Eigen::TensorMap<const Eigen::Tensor<float, RANK, Eigen::RowMajor>> map(unwrap(ca).data_ptr.get(), dims); \
        /* Eigen Scan: 0 is the axis, SumReducer is the op, false is exclusive */ \
        return wrap(map.scan(adj_axis, Eigen::internal::SumReducer<float>(), false), ca.shape());

    DISPATCH_RANK(ndim, EXEC_CUMSUM);
    #undef EXEC_CUMSUM
    throw std::runtime_error("cumsum failed");
}

// FFT — Wraps kissfft natively included in Eigen.
// output stored with is_complex=true, real in left half, imag in right half.
Tensor rfft(const Tensor& a, int n, int axis) {
    // 1. Enforce contiguity and identify axes
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    int ndim = ca.ndim();
    int adj_axis = (axis < 0) ? axis + ndim : axis;

    int L = impl.shape[adj_axis];
    int fft_n = (n == -1) ? L : n;
    int out_L = fft_n / 2 + 1;

    // 2. Prepare output shape and complex tensor
    auto out_shape = impl.shape;
    out_shape[adj_axis] = out_L;
    auto res_impl = std::make_shared<TensorImpl>(out_shape, true); // is_complex = true

    // 3. Batching logic: All dimensions except the FFT axis
    int stride_before = 1;
    for (int i = 0; i < adj_axis; ++i) stride_before *= impl.shape[i];
    int stride_after = 1;
    for (int i = adj_axis + 1; i < ndim; ++i) stride_after *= impl.shape[i];

    Eigen::FFT<float> fft;
    float* pSrc = impl.data_ptr.get() + impl.offset;
    float* pDst = res_impl->data_ptr.get();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < stride_before; ++i) {
        for (int j = 0; j < stride_after; ++j) {
            // Local buffers for FFT length n
            std::vector<float> time_domain(fft_n, 0.0f);
            std::vector<std::complex<float>> freq_domain;

            // Copy input slice (handle truncation/padding)
            int copy_len = std::min(L, fft_n);
            for (int k = 0; k < copy_len; ++k) {
                time_domain[k] = pSrc[(i * L * stride_after) + (k * stride_after) + j];
            }

            // Execute Real-to-Complex FFT
            fft.fwd(freq_domain, time_domain);

            // Copy Hermitian symmetric results (first n/2 + 1 elements)
            for (int k = 0; k < out_L; ++k) {
                int dst_idx = (i * out_L * stride_after) + (k * stride_after) + j;
                pDst[dst_idx * 2]     = freq_domain[k].real();
                pDst[dst_idx * 2 + 1] = freq_domain[k].imag();
            }
        }
    }

    return Tensor(res_impl, DType::Float32);
}

Tensor irfft(const Tensor& a, int n, int axis) {
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    if (!impl.is_complex) throw std::runtime_error("[isomorphism/eigen] irfft requires a complex input");

    int ndim = ca.ndim();
    int adj_axis = (axis < 0) ? axis + ndim : axis;

    int out_L_complex = impl.shape[adj_axis];
    int fft_n = (n == -1) ? 2 * (out_L_complex - 1) : n;

    auto out_shape = impl.shape;
    out_shape[adj_axis] = fft_n;
    auto res_impl = std::make_shared<TensorImpl>(out_shape, false); // is_complex = false

    int stride_before = 1;
    for (int i = 0; i < adj_axis; ++i) stride_before *= impl.shape[i];
    int stride_after = 1;
    for (int i = adj_axis + 1; i < ndim; ++i) stride_after *= impl.shape[i];

    Eigen::FFT<float> fft;
    float* pSrc = impl.data_ptr.get() + impl.offset;
    float* pDst = res_impl->data_ptr.get();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < stride_before; ++i) {
        for (int j = 0; j < stride_after; ++j) {
            std::vector<std::complex<float>> freq_domain(fft_n);
            std::vector<float> time_domain;

            // Reconstruct full spectrum using Hermitian symmetry: X[k] = conj(X[N-k])
            for (int k = 0; k < out_L_complex; ++k) {
                int src_idx = (i * out_L_complex * stride_after) + (k * stride_after) + j;
                freq_domain[k] = {pSrc[src_idx * 2], pSrc[src_idx * 2 + 1]};
            }
            for (int k = out_L_complex; k < fft_n; ++k) {
                freq_domain[k] = std::conj(freq_domain[fft_n - k]);
            }

            // Execute Complex-to-Real Inverse FFT
            fft.inv(time_domain, freq_domain);

            // Write back to real-valued output
            for (int k = 0; k < fft_n; ++k) {
                int dst_idx = (i * fft_n * stride_after) + (k * stride_after) + j;
                pDst[dst_idx] = time_domain[k];
            }
        }
    }

    return Tensor(res_impl, DType::Float32);
}

// ==============================================================================
// 5. HEAVY LINEAR ALGEBRA
// ==============================================================================

using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec = Eigen::VectorXf;

Tensor solve(const Tensor& a, const Tensor& b) {
    // solvers require contiguous memory for efficient LU mapping
    Tensor ca = contiguous(a);
    Tensor cb = contiguous(b);

    int d = ca.shape().back();
    int k = cb.shape().back();
    int bat = ca.size() / (d * d);

    auto res_impl = std::make_shared<TensorImpl>(cb.shape());
    float* pA = unwrap(ca).data_ptr.get();
    float* pB = unwrap(cb).data_ptr.get();
    float* pX = res_impl->data_ptr.get();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bat; ++i) {
        Eigen::Map<const Mat> mapA(pA + i * d * d, d, d);
        Eigen::Map<const Mat> mapB(pB + i * d * k, d, k);
        Eigen::Map<Mat> mapX(pX + i * d * k, d, k);

        // Use partial pivoting LU for stable general-purpose solving
        mapX = mapA.partialPivLu().solve(mapB);
    }
    return Tensor(res_impl, DType::Float32);
}

std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& a) {
    Tensor ca = contiguous(a);
    int M = ca.shape()[ca.ndim() - 2];
    int N = ca.shape().back();
    int K = std::min(M, N);
    int bat = ca.size() / (M * N);

    // Prepare output shapes and implementations
    auto shapeU = ca.shape(); shapeU.back() = M;
    auto shapeS = ca.shape(); shapeS.pop_back(); shapeS.back() = K;
    auto shapeV = ca.shape(); shapeV[ca.ndim()-2] = N;

    auto implU = std::make_shared<TensorImpl>(shapeU);
    auto implS = std::make_shared<TensorImpl>(shapeS);
    auto implV = std::make_shared<TensorImpl>(shapeV);

    float* pA = unwrap(ca).data_ptr.get();

    #pragma omp parallel if(bat > 1)
    {
        // 2. Hoist the SVD solver workspace.
        // Pre-allocating with dimensions (M, N) allows BDCSVD to reserve all
        // necessary memory upfront once per thread.
        Eigen::BDCSVD<Mat> svd_calc(M, N, Eigen::ComputeFullU | Eigen::ComputeFullV);

        #pragma omp for schedule(static)
        for (int i = 0; i < bat; ++i) {
            Eigen::Map<const Mat> mapA(pA + i * M * N, M, N);
            Eigen::Map<Mat> mapU(implU->data_ptr.get() + i * M * M, M, M);
            Eigen::Map<Vec> mapS(implS->data_ptr.get() + i * K, K);
            Eigen::Map<Mat> mapV(implV->data_ptr.get() + i * N * N, N, N);

            // 3. Compute using the pre-allocated workspace
            svd_calc.compute(mapA);

            // 4. Assign results using .noalias() where possible
            mapU.noalias() = svd_calc.matrixU();
            mapS.noalias() = svd_calc.singularValues();

            // Note: matrixV().transpose() returns Vt as required by math.hpp
            mapV.noalias() = svd_calc.matrixV().transpose();
        }
    }

    return {Tensor(implU, DType::Float32),
            Tensor(implS, DType::Float32),
            Tensor(implV, DType::Float32)};
}

    /*
Tensor matrix_exp(const Tensor& a) {
    Tensor ca = contiguous(a);
    int d = ca.shape().back();
    int bat = ca.size() / (d * d);
    auto res_impl = std::make_shared<TensorImpl>(ca.shape());

    float* pA = unwrap(ca).data_ptr.get();
    float* pR = res_impl->data_ptr.get();

    // Padé thresholds for single precision (Blanes et al. 2024)
    constexpr float kTheta3 = 0.42f, kTheta5 = 1.90f, kTheta7 = 3.70f;

    // 1. Thread-level parallelization.
    // We open the parallel region here, BEFORE the loop.
    #pragma omp parallel
    {
        // 2. Thread-local workspaces.
        // These are allocated EXACTLY ONCE per thread on the heap, bypassing the OS memory
        // allocator lock contention that was ruining your multi-core scaling.
        Mat B(d, d), B2(d, d), B4(d, d), B6(d, d);
        Mat U(d, d), V(d, d), X(d, d), temp(d, d);
        Mat I = Mat::Identity(d, d);

        // 3. Distribute the batched workload across the threads
        #pragma omp for schedule(static)
        for (int i = 0; i < bat; ++i) {
            Eigen::Map<const Mat> mapA(pA + i * d * d, d, d);
            Eigen::Map<Mat> mapRes(pR + i * d * d, d, d);

            float norm1 = mapA.cwiseAbs().colwise().sum().maxCoeff();
            int s = 0; float scaled_n = norm1;
            while (scaled_n > kTheta7) { scaled_n /= 2.0f; ++s; }
            int m = (scaled_n <= kTheta3) ? 3 : (scaled_n <= kTheta5) ? 5 : 7;

            B = mapA / std::pow(2.0f, (float)s);

            // 4. Use .noalias() to guarantee no hidden temporaries are created
            B2.noalias() = B * B;

            if (m == 3) {
                temp = 60.0f * I + B2;
                U.noalias() = B * temp;
                V = 120.0f * I + 12.0f * B2;
            } else if (m == 5) {
                B4.noalias() = B2 * B2;
                temp = 15120.0f * I + 420.0f * B2 + B4;
                U.noalias() = B * temp;
                V = 30240.0f * I + 3360.0f * B2 + 30.0f * B4;
            } else {
                B4.noalias() = B2 * B2;
                B6.noalias() = B4 * B2;
                temp = 8648640.0f * I + 277200.0f * B2 + 1512.0f * B4 + B6;
                U.noalias() = B * temp;
                V = 17297280.0f * I + 1995840.0f * B2 + 25200.0f * B4 + 56.0f * B6;
            }

            // 5. Solve: (V - U) * X = (V + U)
            // We use 'temp' to hold (V - U) to avoid allocating a temporary inside the solver.
            temp = V - U;
            X = temp.partialPivLu().solve(V + U);

            // 6. Squaring phase: Alternating buffers using 'temp' to avoid X = X * X aliasing
            for (int k = 0; k < s; ++k) {
                temp.noalias() = X * X;
                X = temp;
            }

            // Write directly to the output pointer
            mapRes = X;
        }
    }
    return Tensor(res_impl, DType::Float32);
}*/

    Tensor matrix_exp(const Tensor& a) {
    // 1. Ensure memory is contiguous for the Eigen Map
    Tensor ca = contiguous(a);
    int d = ca.shape().back();
    int bat = ca.size() / (d * d);
    auto res_impl = std::make_shared<TensorImpl>(ca.shape());

    float* pA = unwrap(ca).data_ptr.get();
    float* pR = res_impl->data_ptr.get();

#pragma omp parallel if(bat > 1)
    {
        // 3. Thread-local workspace to prevent heap contention
        // We evaluate the exponential into a local matrix first to ensure
        // thread-safety and avoid aliasing during the assignment to the map.
        Mat res_local(d, d);

#pragma omp for schedule(static)
        for (int i = 0; i < bat; ++i) {
            Eigen::Map<const Mat> mapA(pA + i * d * d, d, d);
            Eigen::Map<Mat> mapRes(pR + i * d * d, d, d);

            // 4. Native Eigen Call
            mapRes.noalias() = mapA.exp();
        }
    }

    return Tensor(res_impl, DType::Float32);
}

Tensor matrix_log(const Tensor& a) {
    Tensor ca = contiguous(a);
    int d = ca.shape().back();
    int bat = ca.size() / (d * d);
    auto res_impl = std::make_shared<TensorImpl>(ca.shape());

    float* pA = unwrap(ca).data_ptr.get();
    float* pR = res_impl->data_ptr.get();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bat; ++i) {
        Eigen::Map<const Mat> mapA(pA + i * d * d, d, d);
        Eigen::Map<Mat> mapRes(pR + i * d * d, d, d);
        mapRes.noalias() = mapA.log();
    }

    return Tensor(res_impl, DType::Float32);
}

std::tuple<Tensor, Tensor> qr(const Tensor& a) {
    Tensor ca = contiguous(a);
    int M = ca.shape()[ca.ndim()-2];
    int N = ca.shape().back();
    int K = std::min(M, N); // The rank/economy dimension
    int bat = ca.size() / (M * N);

    auto implQ = std::make_shared<TensorImpl>(ca.shape());
    auto shapeR = ca.shape(); shapeR[ca.ndim()-2] = K;
    auto implR = std::make_shared<TensorImpl>(shapeR);

    float* pA = unwrap(ca).data_ptr.get();
    float* pQ = implQ->data_ptr.get();
    float* pR = implR->data_ptr.get();

    #pragma omp parallel if(bat > 1)
    {
        // 2. Thread-local workspace hoisting
        // We instantiate the QR solver exactly ONCE per thread.
        Eigen::HouseholderQR<Mat> qr_workspace;

        // Pre-allocate the Identity matrix used to extract Q to avoid hidden temporaries
        Mat I = Mat::Identity(M, N);

        #pragma omp for schedule(static)
        for (int i = 0; i < bat; ++i) {
            Eigen::Map<const Mat> mapA(pA + i * M * N, M, N);
            Eigen::Map<Mat> mapQ(pQ + i * M * N, M, N);
            Eigen::Map<Mat> mapR(pR + i * K * N, K, N);

            // 3. Compute without reallocation
            // Because 'mapA' is the same size every iteration, qr_workspace reuses
            // its internal heap buffers instead of calling 'new' and 'delete'.
            qr_workspace.compute(mapA);

            // 4. Extract Q efficiently
            // Multiplying the Householder sequence by our preallocated identity
            // matrix extracts Q directly into the output pointer.
            mapQ.noalias() = qr_workspace.householderQ() * I;

            // 5. Extract R
            // Slice the top K rows and map only the upper triangular portion
            mapR = qr_workspace.matrixQR().topRows(K).template triangularView<Eigen::Upper>();
        }
    }

    return {Tensor(implQ, DType::Float32), Tensor(implR, DType::Float32)};
}

Tensor det(const Tensor& a) {
    Tensor ca = contiguous(a);
    int d = ca.shape().back();
    int bat = ca.size() / (d * d);
    auto out_shape = ca.shape(); out_shape.pop_back(); out_shape.pop_back();
    if (out_shape.empty()) out_shape = {1};

    auto res_impl = std::make_shared<TensorImpl>(out_shape);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bat; ++i) {
        Eigen::Map<const Mat> mapA(unwrap(ca).data_ptr.get() + i * d * d, d, d);
        res_impl->data_ptr[i] = mapA.partialPivLu().determinant();
    }
    return Tensor(res_impl, DType::Float32);
}

Tensor inv(const Tensor& a) {
    Tensor ca = contiguous(a);
    int d = ca.shape().back();
    int bat = ca.size() / (d * d);
    auto res_impl = std::make_shared<TensorImpl>(ca.shape());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bat; ++i) {
        Eigen::Map<const Mat> mapA(unwrap(ca).data_ptr.get() + i * d * d, d, d);
        Eigen::Map<Mat>(res_impl->data_ptr.get() + i * d * d, d, d) = mapA.inverse();
    }
    return Tensor(res_impl, DType::Float32);
}

Tensor diag_embed(const Tensor& v) {
    Tensor cv = contiguous(v);
    int k = cv.shape().back();
    int bat = cv.size() / k;
    auto res_impl = std::make_shared<TensorImpl>(std::vector<int>{bat, k, k});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bat; ++i) {
        for (int j = 0; j < k; ++j) {
            res_impl->data_ptr[i * k * k + j * k + j] = unwrap(cv).data_ptr[i * k + j];
        }
    }
    return Tensor(res_impl, DType::Float32);
}

Tensor diag_extract(const Tensor& a) {
    Tensor ca = contiguous(a);
    int k = ca.shape().back();
    int bat = ca.size() / (k * k);
    auto res_impl = std::make_shared<TensorImpl>(std::vector<int>{bat, k});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bat; ++i) {
        for (int j = 0; j < k; ++j) {
            res_impl->data_ptr[i * k + j] = unwrap(ca).data_ptr[i * k * k + j * k + j];
        }
    }
    return Tensor(res_impl, DType::Float32);
}

Tensor trace(const Tensor& a) {
    return sum(diag_extract(a), {-1}); // math.hpp
}

Tensor sign(const Tensor& a) {
    auto& impl = unwrap(a);
    Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>> map(impl.data_ptr.get(), impl.size());
    return wrap(map.sign(), a.shape());
}

    // ==============================================================================
    // 6. STOCHASTIC GENERATION
    // ==============================================================================

    /**
     * @brief Provides a thread-local Mersenne Twister generator.
     * Utilizing thread_local prevents race conditions and cache contention when
     * generating random numbers in parallel blocks.
     */
    static std::mt19937& get_rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

    Tensor random_normal(const std::vector<int> &shape, DType dtype) {
    // Allocate a new contiguous TensorImpl for the target shape
    auto impl = std::make_shared<TensorImpl>(shape);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* ptr = impl->data_ptr.get();
    int total = impl->size();

#pragma omp parallel
    {
        auto& gen = get_rng();
#pragma omp for schedule(static)
        for (int i = 0; i < total; ++i) {
            ptr[i] = dist(gen);
        }
    }
    return Tensor(impl, dtype);
}

    Tensor random_uniform(const std::vector<int> &shape, DType dtype) {
    // Allocate a new contiguous TensorImpl for the target shape
    auto impl = std::make_shared<TensorImpl>(shape);

    // Standard uniform distribution on the half-open interval [0.0, 1.0)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float* ptr = impl->data_ptr.get();
    int total = impl->size();

#pragma omp parallel
    {
        auto& gen = get_rng();
#pragma omp for schedule(static)
        for (int i = 0; i < total; ++i) {
            ptr[i] = dist(gen);
        }
    }
    return Tensor(impl, dtype);
}

// ==============================================================================
// 7. CPU-GPU BRIDGE
// ==============================================================================

double to_double(const Tensor &a) {
    auto& impl = unwrap(a);
    // Access the scalar at the current view's offset
    return static_cast<double>(impl.data_ptr[impl.offset]);
}

int to_int(const Tensor &a) {
    auto& impl = unwrap(a);
    return static_cast<int>(impl.data_ptr[impl.offset]);
}

std::vector<float> to_float_vector(const Tensor &a) {
    // Ensure the tensor is contiguous before performing a linear memory copy
    Tensor ca = contiguous(a);
    auto& impl = unwrap(ca);
    std::vector<float> v(impl.size());
    std::copy(impl.data_ptr.get(), impl.data_ptr.get() + impl.size(), v.begin());
    return v;
}

void eval(const Tensor &a) {
    // The Eigen backend is eager; all math is evaluated during the 'wrap' call
    // No asynchronous graph or pending computation exists to synchronize.
}

Tensor concatenate(const std::vector<Tensor> &tensors, int axis) {
    if (tensors.empty()) throw std::runtime_error("[isomorphism/eigen] concatenate: empty list");

    int ndim = tensors[0].ndim();
    if (axis < 0) axis += ndim;

    auto out_shape = tensors[0].shape();
    for (size_t i = 1; i < tensors.size(); ++i) {
        out_shape[axis] += tensors[i].shape()[axis];
    }

    // Allocate a new contiguous buffer for the result
    auto res_impl = std::make_shared<TensorImpl>(out_shape);
    float* out_data = res_impl->data_ptr.get();

    int outer_size = 1; for(int i = 0; i < axis; ++i) outer_size *= out_shape[i];
    int inner_size = 1; for(int i = axis + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int out_dim_size = out_shape[axis];

    // N-Dimensional strided concatenation
    for (int o = 0; o < outer_size; ++o) {
        int current_dim_offset = 0;
        for (const auto& t : tensors) {
            // Force contiguity for each source tensor to simplify the copy loop
            Tensor ct = contiguous(t);
            auto& t_impl = unwrap(ct);
            int t_dim_size = t.shape()[axis];

            for (int d = 0; d < t_dim_size; ++d) {
                const float* src = t_impl.data_ptr.get() + o * (t_dim_size * inner_size) + d * inner_size;
                float* dst = out_data + o * (out_dim_size * inner_size) + (current_dim_offset + d) * inner_size;
                std::memcpy(dst, src, inner_size * sizeof(float));
            }
            current_dim_offset += t_dim_size;
        }
    }
    return Tensor(res_impl, DType::Float32);
}

Tensor slice(const Tensor &a, int start, int end, int axis) {
    auto& impl = unwrap(a);
    int ndim = impl.shape.size();
    if (axis < 0) axis += ndim;

    // Zero-Copy Slice: Just update the shape and offset
    auto new_shape = impl.shape;
    new_shape[axis] = end - start;

    // The new offset is the old offset shifted by 'start' times the stride of the sliced axis
    int new_offset = impl.offset + (start * impl.strides[axis]);

    // Return a new view sharing the same underlying data_ptr
    return wrap_view(impl.data_ptr, new_shape, impl.strides, new_offset);
}

Tensor eigvalsh(const Tensor& a) {
    Tensor ca  = contiguous(a);
    int    d   = ca.shape().back();
    int    bat = ca.size() / (d * d);

    // Output shape: drop the last dimension → [..., d].
    auto shape_out = ca.shape();
    shape_out.pop_back();
    auto impl = std::make_shared<TensorImpl>(shape_out);

    const float* pA   = unwrap(ca).data_ptr.get();
    float*       pOut = impl->data_ptr.get();

    #pragma omp parallel if(bat > 1)
    {
        // Hoist solver workspace: one allocation per thread.
        Eigen::SelfAdjointEigenSolver<Mat> solver(d);

        #pragma omp for schedule(static)
        for (int i = 0; i < bat; ++i) {
            Eigen::Map<const Mat> mapA(pA   + i * d * d, d, d);
            Eigen::Map<Vec>       mapV(pOut + i * d,     d);
            solver.compute(mapA, Eigen::EigenvaluesOnly);
            mapV = solver.eigenvalues();
        }
    }
    return Tensor(impl, DType::Float32);
}

void set_default_device_cpu() {
    // Disable Eigen's internal multithreading to avoid contention with
    // external samplers already using OpenMP
    Eigen::setNbThreads(1);
}

void set_default_device_gpu() {
    // No-op for the Eigen CPU backend
}
} // namespace isomorphism::math
 // namespace isomorphism
