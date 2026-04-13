/**
 * @file tensor_torch.cpp
 * @brief PyTorch (LibTorch) backend implementation for the Tensor class.
 *
 * Defines the TensorImpl struct (forward-declared in tensor.hpp) backed by a
 * torch::Tensor. PyTorch is eager and manages its own memory — no custom
 * pooling or eval() triggers required.
 */

#include <iomanip>
#include <vector>
#include <ostream>

#include "tensor_impl_torch.hpp"
#include "isomorphism/tensor.hpp"

namespace isomorphism {

// ==============================================================================
// TENSOR CLASS IMPLEMENTATION
// ==============================================================================

Tensor::Tensor() : pimpl_(nullptr), dtype_(DType::Float32) {}

Tensor::Tensor(double scalar_value, DType dtype) : dtype_(dtype) {
    auto t = torch::full({}, static_cast<float>(scalar_value),
                         torch::TensorOptions().dtype(get_torch_dtype(dtype)));
    pimpl_ = std::make_shared<TensorImpl>(std::move(t));
}

Tensor::Tensor(std::shared_ptr<TensorImpl> pimpl, DType dtype)
    : pimpl_(std::move(pimpl)), dtype_(dtype) {}

Tensor::~Tensor() = default;

// ==============================================================================
// INTROSPECTION METHODS
// ==============================================================================

std::vector<int> Tensor::shape() const {
    if (!pimpl_) return {};
    auto sizes = pimpl_->data.sizes();
    return std::vector<int>(sizes.begin(), sizes.end());
}

int Tensor::ndim() const {
    if (!pimpl_) return 0;
    return static_cast<int>(pimpl_->data.dim());
}

int Tensor::size() const {
    if (!pimpl_) return 0;
    return static_cast<int>(pimpl_->data.numel());
}

// ==============================================================================
// PRINTING & FORMATTING
// ==============================================================================

static void print_recursive(std::ostream& os, const float* data, const std::vector<int>& shape,
                             int depth, size_t offset, const std::vector<size_t>& strides) {
    if (shape.empty()) { os << data[0]; return; }

    if (depth == static_cast<int>(shape.size()) - 1) {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            os << std::setprecision(6) << data[offset + i * strides[depth]];
            if (i < shape[depth] - 1) os << ", ";
        }
        os << "]";
    } else {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            if (i > 0) { os << ",\n" << std::string(depth + 1, ' '); }
            print_recursive(os, data, shape, depth + 1, offset + i * strides[depth], strides);
        }
        os << "]";
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    auto impl = tensor.get_impl();
    if (!impl) return os << "Tensor(Null)";

    // Bring to CPU float32 contiguous for printing
    auto t = impl->data.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto shape = tensor.shape();

    std::vector<size_t> strides(shape.size(), 1);
    if (!shape.empty()) {
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    const float* ptr = t.data_ptr<float>();

    os << "Tensor(shape={";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    os << "}, data=\n";
    print_recursive(os, ptr, shape, 0, 0, strides);
    os << "\n)";
    return os;
}

} // namespace isomorphism
