/**
 * @file interop/eigen.hpp
 * @brief Zero-copy interop between isomorphism::Tensor and Eigen tensors/matrices.
 *
 * Include this header when you are building against the Eigen CPU backend and
 * need to map a native Eigen Matrix or Tensor into (or out of) the isomorphism
 * API.
 *
 * Only available when compiled with USE_EIGEN.
 *
 * Usage:
 * @code
 *   #include <isomorphism/interop/eigen.hpp>
 *   namespace iso_eigen = isomorphism::interop::eigen;
 *
 *   // Eigen::MatrixXf → isomorphism (copies into pooled storage)
 *   Eigen::MatrixXf M = Eigen::MatrixXf::Random(4, 4);
 *   isomorphism::Tensor t = iso_eigen::wrap(M);
 *
 *   // isomorphism → Eigen::Map (zero-copy view, contiguous tensors only)
 *   auto view = iso_eigen::unwrap_map<2>(t);   // Eigen::TensorMap<...>
 *
 *   // isomorphism → Eigen::MatrixXf (copies out)
 *   Eigen::MatrixXf out = iso_eigen::to_matrix(t);
 * @endcode
 */

#pragma once

#ifndef USE_EIGEN
#  error "isomorphism/interop/eigen.hpp requires the Eigen backend (compile with USE_EIGEN defined)"
#endif

#include "isomorphism/tensor.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#if __has_include(<isomorphism/src/backends/eigen/tensor_impl_eigen.hpp>)
#  include <isomorphism/src/backends/eigen/tensor_impl_eigen.hpp>
#else
#  include "../../src/backends/eigen/tensor_impl_eigen.hpp"
#endif

namespace isomorphism::interop::eigen {

// ---------------------------------------------------------------------------
// wrap: Eigen::MatrixXf → isomorphism::Tensor
//
// Copies the matrix data into isomorphism's pooled storage (row-major).
// The shape is [rows, cols].
// ---------------------------------------------------------------------------

/**
 * @brief Wrap an Eigen::MatrixXf as an isomorphism::Tensor  [rows, cols].
 *
 * Data is copied into isomorphism's pooled buffer so the Tensor can outlive
 * the Eigen object.  For large matrices prefer building the Tensor directly
 * via isomorphism::math::array() to avoid the extra allocation.
 */
inline isomorphism::Tensor wrap(const Eigen::MatrixXf& M) {
    const std::vector<int> shape = {static_cast<int>(M.rows()),
                                    static_cast<int>(M.cols())};
    auto impl = std::make_shared<isomorphism::TensorImpl>(shape);

    // Eigen is column-major by default; map to row-major layout.
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        impl->data_ptr.get(), M.rows(), M.cols()) = M;

    return isomorphism::Tensor(impl, isomorphism::DType::Float32);
}

/**
 * @brief Wrap a raw float pointer + shape as an isomorphism::Tensor.
 *
 * The data is copied into pooled storage.  Useful when you already have a
 * contiguous row-major float buffer (e.g. from Eigen::Map).
 */
inline isomorphism::Tensor wrap(const float* data, const std::vector<int>& shape) {
    auto impl = std::make_shared<isomorphism::TensorImpl>(shape);
    size_t n = impl->size();
    std::copy(data, data + n, impl->data_ptr.get());
    return isomorphism::Tensor(impl, isomorphism::DType::Float32);
}

// ---------------------------------------------------------------------------
// unwrap helpers
// ---------------------------------------------------------------------------

/**
 * @brief Return a raw pointer to the contiguous float data inside a Tensor.
 *
 * @warning The Tensor must be contiguous (i.e. not a view created by slice/
 * transpose without a subsequent materialisation).  Call isomorphism::math::eval()
 * first if in doubt.
 *
 * @param t  A Tensor created by the Eigen backend.
 * @return   Pointer to the first float element; valid for t.size() floats.
 */
inline float* unwrap_ptr(const isomorphism::Tensor& t) {
    auto& impl = *t.get_impl();
    return impl.data_ptr.get() + impl.offset;
}

/**
 * @brief Return an Eigen::Map<MatrixXf> view (row-major) over a 2D Tensor.
 *
 * Zero-copy: the map points directly into the Tensor's pooled buffer.
 * The Tensor must be 2D and contiguous.
 *
 * @param t  A 2D Tensor created by the Eigen backend.
 */
inline Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
to_matrix_map(const isomorphism::Tensor& t) {
    const auto& s = t.get_impl()->shape;
    return {unwrap_ptr(t), s[0], s[1]};
}

/**
 * @brief Copy a 2D Tensor out as an Eigen::MatrixXf.
 */
inline Eigen::MatrixXf to_matrix(const isomorphism::Tensor& t) {
    return to_matrix_map(t);
}

} // namespace isomorphism::interop::eigen
