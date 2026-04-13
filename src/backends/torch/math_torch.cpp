// Path: isomorphism/src/backends/torch/math_torch.cpp

#include "isomorphism/math.hpp"
#include "tensor_impl_torch.hpp"
#include <torch/torch.h>

namespace isomorphism::math {

    // Active device for all tensor-creating operations (eye, full, random_*, array).
    // Defaults to CPU; updated by set_default_device_cpu/gpu.
    static torch::Device g_default_device(torch::kCPU);

    // ==============================================================================
    // INTERNAL HELPERS
    // ==============================================================================

    static inline torch::Tensor unwrap(const Tensor& t) {
        return t.get_impl()->data;
    }

    static inline Tensor wrap(torch::Tensor t) {
        return Tensor(std::make_shared<TensorImpl>(std::move(t)));
    }

    // ==============================================================================
    // 1. ELEMENT-WISE ARITHMETIC
    // ==============================================================================

    Tensor add(const Tensor& a, const Tensor& b)      { return wrap(torch::add(unwrap(a), unwrap(b))); }
    Tensor subtract(const Tensor& a, const Tensor& b) { return wrap(torch::sub(unwrap(a), unwrap(b))); }
    Tensor multiply(const Tensor& a, const Tensor& b) { return wrap(torch::mul(unwrap(a), unwrap(b))); }
    Tensor divide(const Tensor& a, const Tensor& b)   { return wrap(torch::div(unwrap(a), unwrap(b))); }
    Tensor floor(const Tensor& a)                     { return wrap(torch::floor(unwrap(a))); }
    Tensor mean(const Tensor& a)                      { return wrap(torch::mean(unwrap(a))); }
    Tensor minimum(const Tensor& a, const Tensor& b)  { return wrap(torch::minimum(unwrap(a), unwrap(b))); }
    Tensor ceil(const Tensor& a)                      { return wrap(torch::ceil(unwrap(a))); }
    Tensor round(const Tensor& a)                     { return wrap(torch::round(unwrap(a))); }

    Tensor clamp(const Tensor& a, float min_val, float max_val) {
        return wrap(torch::clamp(unwrap(a), min_val, max_val));
    }

    // ==============================================================================
    // 2. CORE MATRIX OPERATIONS
    // ==============================================================================

    Tensor matmul(const Tensor& a, const Tensor& b) {
        return wrap(torch::matmul(unwrap(a), unwrap(b)));
    }

    Tensor transpose(const Tensor& a, const std::vector<int>& axes) {
        std::vector<int64_t> perm(axes.begin(), axes.end());
        return wrap(unwrap(a).permute(perm));
    }

    Tensor eye(int d, DType dtype) {
        return wrap(torch::eye(d, torch::TensorOptions()
                                      .dtype(get_torch_dtype(dtype))
                                      .device(g_default_device)));
    }

    Tensor expand_dims(const Tensor& a, const std::vector<int>& axes) {
        auto t = unwrap(a);
        // Insert new dims in ascending order — each insertion preserves the
        // validity of subsequent indices in the sorted list.
        std::vector<int> sorted = axes;
        std::sort(sorted.begin(), sorted.end());
        for (int ax : sorted) {
            t = torch::unsqueeze(t, static_cast<int64_t>(ax));
        }
        return wrap(t);
    }

    Tensor reshape(const Tensor& a, const std::vector<int>& shape) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        return wrap(torch::reshape(unwrap(a), s));
    }

    Tensor broadcast_to(const Tensor& a, const std::vector<int>& shape) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        return wrap(unwrap(a).expand(s));
    }

    Tensor array(const std::vector<float>& data, const std::vector<int>& shape, DType dtype) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        // from_blob doesn't own the data, so clone() immediately to take ownership.
        auto t = torch::from_blob(const_cast<float*>(data.data()), s, torch::kFloat32).clone();
        if (dtype != DType::Float32) t = t.to(get_torch_dtype(dtype));
        return wrap(t.to(g_default_device));
    }

    Tensor astype_int32(const Tensor& a) {
        return wrap(unwrap(a).to(torch::kInt32));
    }

    // Semantics mirror MLX's take(): select elements along axis using flat indices.
    Tensor gather(const Tensor& a, const Tensor& indices, int axis) {
        auto idx = unwrap(indices).flatten().to(torch::kLong);
        return wrap(torch::index_select(unwrap(a), static_cast<int64_t>(axis), idx));
    }

    Tensor stack(const std::vector<Tensor>& tensors, int axis) {
        std::vector<torch::Tensor> ts;
        ts.reserve(tensors.size());
        for (const auto& t : tensors) ts.push_back(unwrap(t));
        return wrap(torch::stack(ts, static_cast<int64_t>(axis)));
    }

    Tensor squeeze(const Tensor& a, const std::vector<int>& axes) {
        auto t = unwrap(a);
        if (axes.empty()) return wrap(torch::squeeze(t));
        // Remove in descending order so earlier removals don't shift later indices.
        std::vector<int> sorted = axes;
        std::sort(sorted.rbegin(), sorted.rend());
        for (int ax : sorted) t = torch::squeeze(t, static_cast<int64_t>(ax));
        return wrap(t);
    }

    Tensor full(const std::vector<int>& shape, float value, DType dtype) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        return wrap(torch::full(s, value, torch::TensorOptions()
                                               .dtype(get_torch_dtype(dtype))
                                               .device(g_default_device)));
    }

    // ==============================================================================
    // 3. LOGICAL OPERATIONS
    // ==============================================================================

    Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
        return wrap(torch::where(unwrap(condition).to(torch::kBool), unwrap(x), unwrap(y)));
    }

    Tensor equal(const Tensor& a, const Tensor& b)         { return wrap(torch::eq(unwrap(a), unwrap(b))); }
    Tensor not_equal(const Tensor& a, const Tensor& b)     { return wrap(torch::ne(unwrap(a), unwrap(b))); }
    Tensor greater(const Tensor& a, const Tensor& b)       { return wrap(torch::gt(unwrap(a), unwrap(b))); }
    Tensor less(const Tensor& a, const Tensor& b)          { return wrap(torch::lt(unwrap(a), unwrap(b))); }
    Tensor greater_equal(const Tensor& a, const Tensor& b) { return wrap(torch::ge(unwrap(a), unwrap(b))); }
    Tensor less_equal(const Tensor& a, const Tensor& b)    { return wrap(torch::le(unwrap(a), unwrap(b))); }
    Tensor logical_and(const Tensor& a, const Tensor& b)   { return wrap(torch::logical_and(unwrap(a), unwrap(b))); }
    Tensor logical_or(const Tensor& a, const Tensor& b)    { return wrap(torch::logical_or(unwrap(a), unwrap(b))); }

    // ==============================================================================
    // 4. REDUCTIONS & NON-LINEARITIES
    // ==============================================================================

    Tensor sum(const Tensor& a, const std::vector<int>& axes) {
        if (axes.empty()) return wrap(torch::sum(unwrap(a)));
        std::vector<int64_t> dims(axes.begin(), axes.end());
        return wrap(torch::sum(unwrap(a), at::IntArrayRef(dims)));
    }

    Tensor cumsum(const Tensor& a, int axis) {
        return wrap(torch::cumsum(unwrap(a), static_cast<int64_t>(axis)));
    }

    Tensor min(const Tensor& a) { return wrap(torch::min(unwrap(a))); }
    Tensor max(const Tensor& a) { return wrap(torch::max(unwrap(a))); }

    Tensor exp(const Tensor& a)    { return wrap(torch::exp(unwrap(a))); }
    Tensor log(const Tensor& a)    { return wrap(torch::log(unwrap(a))); }
    Tensor square(const Tensor& a) { return wrap(torch::square(unwrap(a))); }
    Tensor sqrt(const Tensor& a)   { return wrap(torch::sqrt(unwrap(a))); }
    Tensor abs(const Tensor& a)    { return wrap(torch::abs(unwrap(a))); }
    Tensor sin(const Tensor& a)    { return wrap(torch::sin(unwrap(a))); }
    Tensor cos(const Tensor& a)    { return wrap(torch::cos(unwrap(a))); }
    Tensor asin(const Tensor& a)   { return wrap(torch::asin(unwrap(a))); }
    Tensor acos(const Tensor& a)   { return wrap(torch::acos(unwrap(a))); }
    Tensor atan(const Tensor& a)   { return wrap(torch::atan(unwrap(a))); }
    Tensor tan(const Tensor& a)    { return wrap(torch::tan(unwrap(a))); }
    Tensor sign(const Tensor& a)   { return wrap(torch::sign(unwrap(a))); }

    Tensor atan2(const Tensor& y, const Tensor& x) {
        return wrap(torch::atan2(unwrap(y), unwrap(x)));
    }

    Tensor argmax(const Tensor& a, int axis) {
        return wrap(torch::argmax(unwrap(a), static_cast<int64_t>(axis)));
    }

    Tensor pow(const Tensor& a, float exponent) {
        return wrap(torch::pow(unwrap(a), exponent));
    }

    Tensor prod(const Tensor& a, const std::vector<int>& axes) {
        if (axes.empty()) return wrap(torch::prod(unwrap(a)));
        // torch::prod only reduces one dim at a time; apply in descending order.
        auto t = unwrap(a);
        std::vector<int> sorted = axes;
        std::sort(sorted.rbegin(), sorted.rend());
        for (int ax : sorted) t = torch::prod(t, static_cast<int64_t>(ax));
        return wrap(t);
    }

    Tensor all(const Tensor& a, const std::vector<int>& axes) {
        if (axes.empty()) return wrap(torch::all(unwrap(a)));
        auto t = unwrap(a);
        std::vector<int> sorted = axes;
        std::sort(sorted.rbegin(), sorted.rend());
        for (int ax : sorted) t = torch::all(t, static_cast<int64_t>(ax));
        return wrap(t);
    }

    Tensor any(const Tensor& a, const std::vector<int>& axes) {
        if (axes.empty()) return wrap(torch::any(unwrap(a)));
        auto t = unwrap(a);
        std::vector<int> sorted = axes;
        std::sort(sorted.rbegin(), sorted.rend());
        for (int ax : sorted) t = torch::any(t, static_cast<int64_t>(ax));
        return wrap(t);
    }

    Tensor rfft(const Tensor& a, int n, int axis) {
        c10::optional<int64_t> opt_n = (n == -1) ? c10::nullopt : c10::optional<int64_t>(n);
        return wrap(torch::fft::rfft(unwrap(a), opt_n, static_cast<int64_t>(axis)));
    }

    Tensor irfft(const Tensor& a, int n, int axis) {
        c10::optional<int64_t> opt_n = (n == -1) ? c10::nullopt : c10::optional<int64_t>(n);
        return wrap(torch::fft::irfft(unwrap(a), opt_n, static_cast<int64_t>(axis)));
    }

    // ==============================================================================
    // 5. HEAVY LINEAR ALGEBRA
    // ==============================================================================

    // Richardson iteration solver (Mirroring MLX's custom GPU implementation)
    static torch::Tensor custom_gpu_solve_impl(const torch::Tensor& a, const torch::Tensor& b,
                                              int max_iters = 15, float tol = 1e-5f) {
        // 1. Dynamic Norm and Omega Calculation
        auto abs_a = torch::abs(a);
        auto row_sums = torch::sum(abs_a, /*dim=*/-1, /*keepdim=*/true);
        // torch::max returns a tuple of {values, indices}
        auto norm_a = std::get<0>(torch::max(row_sums, /*dim=*/-2, /*keepdim=*/true));

        auto epsilon = torch::full({}, 1e-7f, a.options());
        auto safe_norm = torch::maximum(norm_a, epsilon);
        auto omega = 1.0f / safe_norm;

        // 2. Initial State
        torch::Tensor x = omega * b;
        torch::Tensor residual = b - torch::matmul(a, x);

        const int check_interval = 5;
        for (int i = 0; i < max_iters; i += check_interval) {
            for (int j = 0; j < check_interval && (i + j) < max_iters; ++j) {
                // Update rule: X_{k+1} = X_k + omega * residual
                x = x + (omega * residual);
                residual = b - torch::matmul(a, x);
            }

            // item() triggers a synchronization point to check convergence
            if (torch::max(torch::abs(residual)).item<float>() < tol) {
                break;
            }
        }
        return x;
    }

    Tensor solve(const Tensor& a, const Tensor& b) {
        auto t_a = unwrap(a);
        auto t_b = unwrap(b);

        // If running on a GPU device (CUDA or Apple MPS), use the iterative solver
        if (t_a.is_cuda() || t_a.is_mps()) {
            return wrap(custom_gpu_solve_impl(t_a, t_b));
        }

        // Otherwise, use native CPU LAPACK
        return wrap(torch::linalg_solve(t_a, t_b));
    }

    std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& a) {
        auto t = unwrap(a);
        auto original_device = t.device();

        // Force the operation onto the CPU
        auto [U, S, Vh] = torch::linalg_svd(t.to(torch::kCPU), /*full_matrices=*/true);

        // Move the results back to the original active device (GPU/MPS/CPU)
        return {wrap(U.to(original_device)),
                wrap(S.to(original_device)),
                wrap(Vh.to(original_device))};
    }

    std::tuple<Tensor, Tensor> qr(const Tensor& a) {
        auto t = unwrap(a);
        auto original_device = t.device();

        // Force the operation onto the CPU
        auto [Q, R] = torch::linalg_qr(t.to(torch::kCPU), "reduced");

        // Move the results back to the original active device
        return {wrap(Q.to(original_device)),
                wrap(R.to(original_device))};
    }

    // PyTorch handles batched determinants natively — no custom Gaussian elimination needed.
    Tensor det(const Tensor& a) {
        return wrap(torch::linalg_det(unwrap(a)));
    }

    Tensor inv(const Tensor& a) {
        return wrap(torch::linalg_inv(unwrap(a)));
    }

    /*
    // Adaptive Padé matrix exponential — fully supported in PyTorch linalg.
    Tensor matrix_exp(const Tensor& a) {
        return wrap(torch::linalg_matrix_exp(unwrap(a)));
    }
    */

    // ==============================================================================
    // MATRIX EXPONENTIAL — Adaptive Diagonal Padé (Blanes, Kopylov, Seydaoglu 2024)
    // ==============================================================================

    Tensor matrix_exp(const Tensor& a) {
        torch::Tensor arr = unwrap(a);
        torch::ScalarType orig_dtype = arr.scalar_type();

        int ndim = arr.dim();
        int d = arr.size(-1); // Matrix dimension

        // Upcast low-precision inputs to float32 for stable Padé computation.
        bool needs_cast = (orig_dtype == torch::kFloat16 || orig_dtype == torch::kBFloat16);
        if (needs_cast) {
            arr = arr.to(torch::kFloat32);
        }

        // --- Batched 1-norm: max absolute column sum ---
        auto abs_a = torch::abs(arr);
        auto col_sums = torch::sum(abs_a, /*dim=*/-2);      // Sum over matrix rows (axis -2)

        // torch::max with a dimension returns a tuple (values, indices)
        auto max_col_sums = std::get<0>(torch::max(col_sums, /*dim=*/-1)); // Max over columns (axis -1)
        auto global_max_arr = torch::max(max_col_sums);      // Global max across entire batch

        double norm1 = static_cast<double>(global_max_arr.item<float>());

        // --- Single-precision backward-error thresholds ---
        constexpr double kTheta3 = 0.42;
        constexpr double kTheta5 = 1.90;
        constexpr double kTheta7 = 3.70;

        // --- Choose scaling s so that ||A/2^s||₁ <= kTheta7, then pick cheapest m ---
        int s = 0;
        double scaled_norm = norm1;
        while (scaled_norm > kTheta7) {
            scaled_norm /= 2.0;
            ++s;
        }
        const int m = (scaled_norm <= kTheta3) ? 3 : (scaled_norm <= kTheta5) ? 5 : 7;

        // --- B = A / 2^s ---
        torch::Tensor B = arr;
        if (s > 0) {
            float inv_s = 1.0f / std::pow(2.0f, static_cast<float>(s));
            B = arr * inv_s;
        }

        // --- Build Padé numerator p(B) = V + U and denominator q(B) = V - U ---
        // We use LibTorch's native operator overloading instead of explicit sc()/add() lambdas
        torch::Tensor I_d = torch::eye(d, torch::TensorOptions().dtype(torch::kFloat32).device(arr.device()));
        torch::Tensor B2 = torch::matmul(B, B);

        torch::Tensor U;
        torch::Tensor V;

        if (m == 3) {
            U = torch::matmul(B, (I_d * 60.f) + (B2 * 1.f));
            V = (I_d * 120.f) + (B2 * 12.f);
        } else if (m == 5) {
            torch::Tensor B4 = torch::matmul(B2, B2);
            U = torch::matmul(B, (I_d * 15120.f) + (B2 * 420.f) + (B4 * 1.f));
            V = (I_d * 30240.f) + (B2 * 3360.f) + (B4 * 30.f);
        } else {  // m == 7
            torch::Tensor B4 = torch::matmul(B2, B2);
            torch::Tensor B6 = torch::matmul(B4, B2);
            U = torch::matmul(B, (I_d * 8648640.f) + (B2 * 277200.f) + (B4 * 1512.f) + (B6 * 1.f));
            V = (I_d * 17297280.f) + (B2 * 1995840.f) + (B4 * 25200.f) + (B6 * 56.f);
        }

        // r_{m,m}(B) = (V - U)^{-1} (V + U) via solve
        torch::Tensor V_minus_U = V - U;
        torch::Tensor V_plus_U  = V + U;
        auto _a = wrap(V_minus_U);
        auto _b = wrap(V_plus_U);

        torch::Tensor result    = unwrap(solve(_a, _b));

        // --- Squaring phase: exp(A) = r_{m,m}(B)^{2^s} ---
        for (int i = 0; i < s; ++i) {
            result = torch::matmul(result, result);
        }

        if (needs_cast) {
            result = result.to(orig_dtype);
        }

        return wrap(result);
    }

    Tensor diag_embed(const Tensor& v) {
        // torch::diag on a 1D tensor creates a 2D diagonal matrix.
        return wrap(torch::diag(unwrap(v)));
    }

    Tensor diag_extract(const Tensor& a) {
        auto t = unwrap(a);
        if (t.dim() < 2) return wrap(t);
        // diagonal() returns a view; contiguous() ensures safe downstream use.
        return wrap(torch::diagonal(t, 0, -2, -1).contiguous());
    }

    Tensor trace(const Tensor& a) {
        auto t = unwrap(a);
        if (t.dim() < 2) return wrap(t);
        // Works for arbitrary batch dims: extract diagonal then sum last axis.
        return wrap(torch::diagonal(t, 0, -2, -1).sum(-1));
    }

    // ==============================================================================
    // 6. STOCHASTIC GENERATION
    // ==============================================================================

    Tensor random_normal(const std::vector<int>& shape, DType dtype) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        return wrap(torch::randn(s, torch::TensorOptions()
                                        .dtype(get_torch_dtype(dtype))
                                        .device(g_default_device)));
    }

    Tensor random_uniform(const std::vector<int>& shape, DType dtype) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        return wrap(torch::rand(s, torch::TensorOptions()
                                       .dtype(get_torch_dtype(dtype))
                                       .device(g_default_device)));
    }

    // ==============================================================================
    // 7. CPU-GPU BRIDGE
    // ==============================================================================

    double to_double(const Tensor& a) {
        return static_cast<double>(unwrap(a).to(torch::kCPU).to(torch::kFloat32).item<float>());
    }

    std::vector<float> to_float_vector(const Tensor& a) {
        auto t = unwrap(a).to(torch::kCPU).to(torch::kFloat32).contiguous();
        const float* ptr = t.data_ptr<float>();
        return std::vector<float>(ptr, ptr + t.numel());
    }

    int to_int(const Tensor& a) {
        return unwrap(a).to(torch::kCPU).item<int>();
    }

    void eval(const Tensor& /*a*/) {
        // PyTorch is eager — all operations are already evaluated.
    }

    Tensor concatenate(const std::vector<Tensor>& tensors, int axis) {
        std::vector<torch::Tensor> ts;
        ts.reserve(tensors.size());
        for (const auto& t : tensors) ts.push_back(unwrap(t));
        return wrap(torch::cat(ts, static_cast<int64_t>(axis)));
    }

    Tensor slice(const Tensor& a, int start, int end, int axis) {
        return wrap(torch::narrow(unwrap(a), static_cast<int64_t>(axis),
                                  static_cast<int64_t>(start),
                                  static_cast<int64_t>(end - start)));
    }

    void set_default_device_cpu() {
        g_default_device = torch::Device(torch::kCPU);
    }

    void set_default_device_gpu() {
        if (torch::cuda::is_available()) {
            g_default_device = torch::Device(torch::kCUDA);
        } else {
            // MPS (Apple Silicon) — attempt to allocate a probe tensor.
            // If MPS is unavailable the constructor throws and we fall back to CPU.
            try {
                torch::zeros({1}, torch::TensorOptions().device(torch::kMPS));
                g_default_device = torch::Device(torch::kMPS);
            } catch (...) {
                g_default_device = torch::Device(torch::kCPU);
            }
        }
    }

} // namespace isomorphism::math
