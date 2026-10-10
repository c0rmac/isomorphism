// Path: isomorphism/src/backends/torch/math_torch.cpp

#include "isomorphism/math.hpp"
#include "tensor_impl_torch.hpp"
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ISOMORPHISM_HAS_METAL_LINALG
#include <metal_linalg/c_api.h>
#include <metal_linalg/device.h>
#include <torch/mps.h>
#endif

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

    static inline DType get_isomorphism_dtype(torch::ScalarType type) {
        switch (type) {
            case torch::kFloat64: return DType::Float64;
            case torch::kFloat16: return DType::Float16;
            case torch::kBFloat16: return DType::BFloat16;
            case torch::kFloat32:
            default:              return DType::Float32;
        }
    }

    static inline Tensor wrap(torch::Tensor t) {
        DType dt = get_isomorphism_dtype(t.scalar_type());
        return Tensor(std::make_shared<TensorImpl>(std::move(t)), dt);
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

    Tensor array(const std::vector<double>& data, const std::vector<int>& shape, DType dtype) {
        std::vector<int64_t> s(shape.begin(), shape.end());
        auto t = torch::from_blob(const_cast<double*>(data.data()), s, torch::kFloat64).clone();
        if (dtype != DType::Float64) t = t.to(get_torch_dtype(dtype));
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

    // torch.linalg.solve on the tensors' device, or on the CPU (the result moved
    // back) where that device has no kernel for it.
    static torch::Tensor linalg_solve_anywhere(const torch::Tensor& a, const torch::Tensor& b) {
        try {
            return torch::linalg_solve(a, b);
        } catch (const c10::Error&) {
            if (a.device().is_cpu()) throw;
            return torch::linalg_solve(a.cpu(), b.cpu()).to(a.device());
        }
    }

    // b as matrices [..., n, k] against a [..., n, n], both expanded to one
    // batch shape (torch's MPS kernels get a broadcast batch wrong, 2.13). b
    // [..., n] is a batch of vectors when it is 1-D or has a's shape without
    // its last axis, as torch.linalg.solve has it; `vector` says so.
    static std::pair<torch::Tensor, torch::Tensor> batch_rhs(torch::Tensor a, torch::Tensor b, bool& vector) {
        vector = b.dim() == 1 || b.sizes() == a.sizes().slice(0, a.dim() - 1);
        if (vector) b = b.unsqueeze(-1);
        auto batch = at::infer_size(a.sizes().slice(0, a.dim() - 2), b.sizes().slice(0, b.dim() - 2));
        std::vector<int64_t> sa(batch), sb(batch);
        sa.insert(sa.end(), {a.size(-2), a.size(-1)});
        sb.insert(sb.end(), {b.size(-2), b.size(-1)});
        return {a.expand(sa).contiguous(), b.expand(sb).contiguous()};
    }


#ifdef ISOMORPHISM_HAS_METAL_LINALG
    // ==============================================================================
    // metal-linalg (Apple Silicon)
    // ==============================================================================
    // The decompositions and solves of CPU and MPS tensors go to metal-linalg,
    // in float32 (float16 and bfloat16 promoted, the results cast back): a CPU
    // tensor on its CPU paths alone (LAPACK through Accelerate, a batch over
    // every core, which outrun torch's), an MPS tensor wherever its measured
    // per-device policy sends that shape, the GPU or the CPU. float64, CUDA and
    // empty tensors stay with torch.
    static bool ml_takes(const torch::Tensor& t) {
        return (t.is_cpu() || t.is_mps()) && t.is_floating_point() && t.scalar_type() != torch::kFloat64 &&
               t.dim() >= 2 && t.numel() > 0;
    }

    // A contiguous MPS tensor's memory as a CPU address, or nullptr where
    // metal-linalg cannot use it in place (not a Metal buffer in shared
    // storage). An MPS storage's data pointer is its id<MTLBuffer>.
    static void* mps_address(const torch::Tensor& t) {
        return metal_linalg_buffer_contents(t.storage().data(), (uint64_t)t.storage_offset() * t.element_size(),
                                            (uint64_t)t.numel() * t.element_size());
    }

    // Whether MPS tensors are in memory metal-linalg can read and write in
    // place: a CPU read of what MPS wrote, and an MPS read of a CPU write.
    // Probed once; where not, MPS tensors are copied to the CPU and back.
    static bool mps_in_place() {
        static const bool ok = [] {
            try {
                auto t = torch::arange(1, 65, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS)) * 2;
                torch::mps::synchronize();
                auto* p = static_cast<float*>(mps_address(t));
                if (!p) return false;
                for (int i = 0; i < 64; ++i)
                    if (p[i] != 2.0f * (i + 1)) return false;
                p[0] = -1.0f;
                return t[0].item<float>() == -1.0f;
            } catch (const std::exception&) {
                return false;
            }
        }();
        return ok;
    }

    // [..., m, n] as (batch, m, n), each below 2^32.
    struct MlDims {
        std::vector<int64_t> lead;
        uint32_t batch, m, n;
    };
    static MlDims ml_dims(const torch::Tensor& t) {
        MlDims d{std::vector<int64_t>(t.sizes().begin(), t.sizes().end() - 2), 1, (uint32_t)t.size(-2),
                 (uint32_t)t.size(-1)};
        int64_t batch = 1;
        for (int64_t e : d.lead) batch *= e;
        if (batch > UINT32_MAX || t.size(-2) > UINT32_MAX || t.size(-1) > UINT32_MAX)
            throw std::invalid_argument("[isomorphism/torch] metal-linalg takes a batch, rows and columns below 2^32");
        d.batch = (uint32_t)batch;
        return d;
    }
    static std::vector<int64_t> with_tail(std::vector<int64_t> lead, std::initializer_list<int64_t> tail) {
        lead.insert(lead.end(), tail);
        return lead;
    }

    struct MlOut {
        std::vector<int64_t> shape;
        torch::ScalarType dtype = torch::kFloat32;
    };
    using MlCall = std::function<metal_linalg_status(const std::vector<const float*>&, const std::vector<void*>&)>;

    // metal-linalg's GPU workspaces are not thread-safe: one GPU-routed call at a time.
    static std::mutex& ml_gpu_mutex() {
        static std::mutex m;
        return m;
    }

    static void ml_check(metal_linalg_status s) {
        if (s != METAL_LINALG_OK)
            throw std::runtime_error(std::string("[isomorphism/torch] metal-linalg: ") + metal_linalg_last_error());
    }

    // Runs call(input addresses, output addresses) on float32 contiguous forms
    // of `in` (all on one device) into new outputs of `outs` on that device.
    static std::vector<torch::Tensor> ml_run(const std::vector<torch::Tensor>& in, const std::vector<MlOut>& outs,
                                             const MlCall& call) {
        const torch::Device device = in[0].device();
        std::vector<torch::Tensor> x, y;
        for (const auto& t : in) x.push_back(t.detach().to(torch::kFloat32).contiguous());
        auto alloc = [&](torch::Device d) {
            y.clear();
            for (const auto& o : outs) y.push_back(torch::empty(o.shape, torch::TensorOptions().dtype(o.dtype).device(d)));
        };
        if (device.is_mps() && mps_in_place()) {
            alloc(device);
            // MPS's queued work may still be writing the input; metal-linalg
            // reads and writes from its own queue and from the CPU.
            torch::mps::synchronize();
            std::vector<const float*> xa;
            std::vector<void*> ya;
            bool all = true;
            for (auto& t : x) {
                xa.push_back(static_cast<const float*>(mps_address(t)));
                all = all && xa.back() != nullptr;
            }
            for (auto& t : y) {
                ya.push_back(mps_address(t));
                all = all && ya.back() != nullptr;
            }
            if (all) {
                std::lock_guard<std::mutex> lock(ml_gpu_mutex());
                // The tensors' own Metal buffers, known to metal-linalg for the
                // call, so that its GPU backends use them rather than wrapping
                // the memory in new buffers
                std::vector<std::pair<const void*, const void*>> known;
                auto know = [&](const torch::Tensor& t, const void* p) {
                    if (p && t.storage_offset() == 0 && metal_linalg_know_buffer(p, t.storage().data()))
                        known.emplace_back(p, t.storage().data());
                };
                for (size_t i = 0; i < x.size(); ++i) know(x[i], xa[i]);
                for (size_t i = 0; i < y.size(); ++i) know(y[i], ya[i]);
                const metal_linalg_status st = call(xa, ya);
                for (const auto& k : known) metal_linalg_forget_buffer(k.first, k.second);
                ml_check(st);
                return y;
            }
        }
        // CPU tensors, on the CPU paths alone; MPS ones copied, routed as on MPS
        for (auto& t : x) t = t.cpu();
        alloc(torch::kCPU);
        std::vector<const float*> xa;
        std::vector<void*> ya;
        for (auto& t : x) xa.push_back(t.data_ptr<float>());
        for (auto& t : y) ya.push_back(t.data_ptr());
        if (device.is_cpu()) {
            metal_linalg::CpuOnly cpu;
            ml_check(call(xa, ya));
        } else {
            std::lock_guard<std::mutex> lock(ml_gpu_mutex());
            ml_check(call(xa, ya));
        }
        if (!device.is_cpu())
            for (auto& t : y) t = t.to(device);
        return y;
    }

    // The determinant from a packed LU [n, n] (row-major) and its 0-based
    // pivots: (-1)^(row swaps + negative pivots) * exp(sum log|u_ii|), in
    // double so that no product overflows on the way; zero for a singular
    // matrix (info > 0).
    static double det_from_lu(const float* lu, const uint32_t* pivots, uint32_t info, uint32_t n) {
        if (info) return 0.0;
        double log_abs = 0.0;
        unsigned flips = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const double u = lu[(size_t)i * n + i];
            log_abs += std::log(std::abs(u));
            flips += (u < 0) + (pivots[i] != i);
        }
        const double m = std::exp(log_abs);
        return flips % 2 ? -m : m;
    }

    static torch::Tensor as(const torch::Tensor& t, torch::ScalarType dtype) {
        return t.scalar_type() == dtype ? t : t.to(dtype);
    }
#endif

    // LU with partial pivoting on every device (metal-linalg's, else
    // torch.linalg.solve, MPS included since torch 2.x).
    Tensor solve(const Tensor& a, const Tensor& b) {
        bool vector = false;
        auto [t_a, t_b] = batch_rhs(unwrap(a), unwrap(b), vector);
        torch::Tensor x;
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        if (ml_takes(t_a) && ml_takes(t_b) && t_a.device() == t_b.device()) {
            const auto out = torch::promote_types(t_a.scalar_type(), t_b.scalar_type());
            const MlDims d = ml_dims(t_a);
            const uint32_t k = (uint32_t)t_b.size(-1);
            auto r = ml_run({t_a, t_b}, {{t_b.sizes().vec()}, {d.lead, torch::kInt32}},
                            [&](const auto& in, const auto& o) {
                                return metal_linalg_solve(in[0], d.batch, d.n, in[1], k, (float*)o[0], (uint32_t*)o[1]);
                            });
            x = as(r[0], out);
        } else
#endif
        x = linalg_solve_anywhere(t_a, t_b);
        return wrap(vector ? x.squeeze(-1) : x);
    }

    std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& a) {
        auto t = unwrap(a);
        auto original_device = t.device();
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        // metal-linalg's thin factors are the full ones for a square matrix;
        // other shapes keep torch's full factors.
        if (ml_takes(t) && t.size(-1) == t.size(-2)) {
            const MlDims d = ml_dims(t);
            const int64_t n = d.n;
            auto r = ml_run({t}, {{t.sizes().vec()}, {with_tail(d.lead, {n})}, {t.sizes().vec()}},
                            [&](const auto& in, const auto& o) {
                                return metal_linalg_svd(in[0], d.batch, d.m, d.n, (float*)o[0], (float*)o[1],
                                                        (float*)o[2], nullptr);
                            });
            return {wrap(as(r[0], t.scalar_type())), wrap(as(r[1], t.scalar_type())), wrap(as(r[2], t.scalar_type()))};
        }
#endif

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
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        if (ml_takes(t)) {
            const MlDims d = ml_dims(t);
            const int64_t m = d.m, n = d.n, k = std::min(m, n);
            auto r = ml_run({t}, {{with_tail(d.lead, {m, k})}, {with_tail(d.lead, {k, n})}},
                            [&](const auto& in, const auto& o) {
                                return metal_linalg_qr(in[0], d.batch, d.m, d.n, (float*)o[0], (float*)o[1]);
                            });
            return {wrap(as(r[0], t.scalar_type())), wrap(as(r[1], t.scalar_type()))};
        }
#endif

        // Force the operation onto the CPU
        auto [Q, R] = torch::linalg_qr(t.to(torch::kCPU), "reduced");

        // Move the results back to the original active device
        return {wrap(Q.to(original_device)),
                wrap(R.to(original_device))};
    }

    // PyTorch handles batched determinants natively — no custom Gaussian elimination needed.
    // On MPS, from its LU: torch's MPS det gives NaN for some sizes (2.13: n = 512
    // and 1536) where its LU of the same matrix is sound.
    Tensor det(const Tensor& a) {
        auto t = unwrap(a);
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        // From metal-linalg's P A = L U, in one pass over its diagonals and
        // pivots once it returns (det_from_lu).
        if (ml_takes(t) && t.size(-1) == t.size(-2)) {
            const MlDims d = ml_dims(t);
            const int64_t n = d.n;
            auto r = ml_run({t},
                            {{t.sizes().vec()}, {with_tail(d.lead, {n}), torch::kInt32}, {d.lead, torch::kInt32},
                             {d.lead}},
                            [&](const auto& in, const auto& o) {
                                const metal_linalg_status st = metal_linalg_lu_factor(
                                    in[0], d.batch, d.n, (float*)o[0], (uint32_t*)o[1], (uint32_t*)o[2]);
                                if (st == METAL_LINALG_OK)
                                    for (uint32_t b = 0; b < d.batch; ++b)
                                        ((float*)o[3])[b] = (float)det_from_lu((const float*)o[0] + (size_t)b * n * n,
                                                                               (const uint32_t*)o[1] + (size_t)b * n,
                                                                               ((const uint32_t*)o[2])[b], d.n);
                                return st;
                            });
            return wrap(as(r[3], t.scalar_type()));
        }
#endif
        if (!t.is_mps() || t.dim() < 2 || t.size(-1) == 0) return wrap(torch::linalg_det(t));
        // The product as exp(sum log|u_ii|) with the signs counted apart: a
        // float32 product can overflow and underflow on the way to a
        // determinant it can hold.
        auto [lu, pivots, info] = torch::linalg_lu_factor_ex(t);
        const int64_t n = t.size(-1);
        auto u = lu.diagonal(0, -2, -1);
        auto flips = pivots.ne(torch::arange(1, n + 1, pivots.options())).sum(-1) + u.lt(0).sum(-1);
        auto d = u.abs().log().sum(-1).exp();
        d = torch::where(flips.remainder(2).eq(1), -d, d);
        return wrap(torch::where(info.gt(0), torch::zeros_like(d), d));   // an exactly zero pivot: singular
    }

    Tensor inv(const Tensor& a) {
        auto t = unwrap(a);
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        // metal-linalg's gives NaN for a singular matrix; torch's own throws.
        if (ml_takes(t) && t.size(-1) == t.size(-2)) {
            const MlDims d = ml_dims(t);
            auto r = ml_run({t}, {{t.sizes().vec()}, {d.lead, torch::kInt32}}, [&](const auto& in, const auto& o) {
                return metal_linalg_inv(in[0], d.batch, d.n, (float*)o[0], (uint32_t*)o[1]);
            });
            return wrap(as(r[0], t.scalar_type()));
        }
#endif
        return wrap(torch::linalg_inv(t));
    }

    // A matrix that is not positive definite becomes NaN rather than throwing,
    // as on the other backends.
    Tensor cholesky(const Tensor& a, bool upper) {
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        if (auto t = unwrap(a); ml_takes(t) && t.size(-1) == t.size(-2)) {
            const MlDims d = ml_dims(t);
            auto r = ml_run({t}, {{t.sizes().vec()}}, [&](const auto& in, const auto& o) {
                return metal_linalg_cholesky(in[0], d.batch, d.n, upper ? 1 : 0, (float*)o[0], nullptr);
            });
            return wrap(as(r[0], t.scalar_type()));
        }
#endif
        auto [L, info] = torch::linalg_cholesky_ex(unwrap(a), upper);
        auto failed = info.ne(0).unsqueeze(-1).unsqueeze(-1);
        return wrap(torch::where(failed, torch::full({}, std::numeric_limits<double>::quiet_NaN(), L.options()), L));
    }

    // b [..., n, k] or [..., n], as for solve (torch's own takes matrices only).
    Tensor solve_triangular(const Tensor& a, const Tensor& b, bool upper, bool unit_diagonal) {
        bool vector = false;
        auto [t_a, t_b] = batch_rhs(unwrap(a), unwrap(b), vector);
        torch::Tensor x;
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        if (ml_takes(t_a) && ml_takes(t_b) && t_a.device() == t_b.device()) {
            const auto out = torch::promote_types(t_a.scalar_type(), t_b.scalar_type());
            const MlDims d = ml_dims(t_a);
            const uint32_t k = (uint32_t)t_b.size(-1);
            auto r = ml_run({t_a, t_b}, {{t_b.sizes().vec()}}, [&](const auto& in, const auto& o) {
                return metal_linalg_solve_triangular(in[0], d.batch, d.n, in[1], k, upper ? 1 : 0,
                                                     unit_diagonal ? 1 : 0, (float*)o[0]);
            });
            x = as(r[0], out);
        } else
#endif
        x = torch::linalg_solve_triangular(t_a, t_b, upper, /*left=*/true, unit_diagonal);
        return wrap(vector ? x.squeeze(-1) : x);
    }

    Tensor eigvalsh(const Tensor& a) {
        auto t = unwrap(a);
        auto original_device = t.device();
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        // Ascending, the lower triangle read, as torch's.
        if (ml_takes(t) && t.size(-1) == t.size(-2)) {
            const MlDims d = ml_dims(t);
            auto r = ml_run({t}, {{with_tail(d.lead, {(int64_t)d.n})}}, [&](const auto& in, const auto& o) {
                return metal_linalg_eigh(in[0], d.batch, d.n, 1, (float*)o[0], nullptr, nullptr);
            });
            return wrap(as(r[0], t.scalar_type()));
        }
#endif
        // torch::linalg_eigvalsh returns eigenvalues in ascending order.
        auto vals = torch::linalg_eigvalsh(t.to(torch::kCPU), "L");
        return wrap(vals.to(original_device));
    }

    std::tuple<Tensor, Tensor> eigh(const Tensor& a) {
        auto t = unwrap(a);
        auto original_device = t.device();
#ifdef ISOMORPHISM_HAS_METAL_LINALG
        if (ml_takes(t) && t.size(-1) == t.size(-2)) {
            const MlDims d = ml_dims(t);
            auto r = ml_run({t}, {{with_tail(d.lead, {(int64_t)d.n})}, {t.sizes().vec()}},
                            [&](const auto& in, const auto& o) {
                                return metal_linalg_eigh(in[0], d.batch, d.n, 1, (float*)o[0], (float*)o[1], nullptr);
                            });
            return {wrap(as(r[0], t.scalar_type())), wrap(as(r[1], t.scalar_type()))};
        }
#endif
        auto [vals, vecs] = torch::linalg_eigh(t.to(torch::kCPU), "L");
        return {wrap(vals.to(original_device)), wrap(vecs.to(original_device))};
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

    Tensor matrix_log(const Tensor& a) {
        torch::Tensor arr = unwrap(a);
        torch::ScalarType orig_dtype = arr.scalar_type();

        bool needs_cast = (orig_dtype == torch::kFloat16 || orig_dtype == torch::kBFloat16);
        if (needs_cast)
            arr = arr.to(torch::kFloat32);

        int ndim = arr.dim();
        int d = arr.size(-1);
        auto dev = arr.device();
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

        static constexpr float kNodes[8] = {
            0.019855072f, 0.101666761f, 0.237233795f, 0.408282679f,
            0.591717321f, 0.762766205f, 0.898333239f, 0.980144928f
        };
        static constexpr float kWeights[8] = {
            0.050614268f, 0.111190517f, 0.156853323f, 0.181341892f,
            0.181341892f, 0.156853323f, 0.111190517f, 0.050614268f
        };

        // Denman-Beavers matrix square root via coupled iteration.
        auto denman_beavers = [&](torch::Tensor Y) -> torch::Tensor {
            torch::Tensor Z = torch::eye(d, opts);  // broadcasts with batch
            for (int iter = 0; iter < 6; ++iter) {
                torch::Tensor Yinv = torch::linalg_inv(Y);
                torch::Tensor Zinv = torch::linalg_inv(Z);
                torch::Tensor Ynew = (Y + Zinv) * 0.5f;
                Z = (Z + Yinv) * 0.5f;
                Y = Ynew;
            }
            return Y;
        };

        // Phase 1: scale down via repeated square roots until ||Ak - I||₁ ≤ 0.75.
        const double kThreshold = 0.75;
        int k = 0;
        torch::Tensor Ak = arr;
        while (true) {
            torch::Tensor diff    = Ak - torch::eye(d, opts);
            torch::Tensor col_sum = torch::sum(torch::abs(diff), /*dim=*/-2);
            torch::Tensor g_max   = torch::max(col_sum);
            double norm1 = static_cast<double>(g_max.item<float>());

            if (norm1 <= kThreshold || k >= 16) break;

            Ak = denman_beavers(Ak);
            ++k;
        }

        // Phase 2: log(I+X) = ∫₀¹ X(I+tX)⁻¹ dt via 8-point Gauss-Legendre.
        torch::Tensor X      = Ak - torch::eye(d, opts);
        torch::Tensor result = torch::zeros_like(X);
        for (int j = 0; j < 8; ++j) {
            torch::Tensor denom   = torch::eye(d, opts) + X * kNodes[j];
            // solve(denom, X) = denom^{-1} X
            torch::Tensor contrib = unwrap(solve(wrap(denom), wrap(X)));
            result = result + contrib * kWeights[j];
        }

        // Phase 3: undo scaling.
        if (k > 0)
            result = result * std::pow(2.0f, static_cast<float>(k));

        if (needs_cast)
            result = result.to(orig_dtype);

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

    std::vector<double> to_double_vector(const Tensor& a) {
        auto t = unwrap(a).to(torch::kCPU).to(torch::kFloat64).contiguous();
        const double* ptr = t.data_ptr<double>();
        return std::vector<double>(ptr, ptr + t.numel());
    }

    int to_int(const Tensor& a) {
        return unwrap(a).to(torch::kCPU).item<int>();
    }

    void eval(const Tensor& /*a*/) {
        // PyTorch is eager — all operations are already evaluated.
    }

    void clear_backend_cache() {
        // Eager allocator managed by libtorch — nothing to release here.
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
