// Path: isomorphism/src/backends/mlx/math_mlx.cpp

#include <iostream>
#include <numeric>

#include "isomorphism/math.hpp"
#include "tensor_impl_mlx.hpp" // Our private header
#include <mlx/mlx.h>
#include "qr_accelerated/qr.h"
#include <thread>
#include <vector>

namespace isomorphism::math {
    // Internal state to track GPU FFT support
    static bool g_fft_gpu_supported = true;
    static std::once_flag g_fft_probe_flag;

    static void probe_fft_support() {
        std::call_once(g_fft_probe_flag, []() {
            try {
                // Perform a minimal 1D FFT on the default device (GPU)
                auto probe_arr = mlx::core::zeros({8}, mlx::core::float32);
                auto result = mlx::core::fft::rfft(probe_arr);

                // Force evaluation to trigger the Metal/CUDA kernel
                mlx::core::eval({result});
                g_fft_gpu_supported = true;
            } catch (const std::exception& e) {
                // If it fails (e.g., "no CUDA implementation"), mark as unsupported
                g_fft_gpu_supported = false;
            }
        });
    }

    // ==============================================================================
    // INTERNAL HELPERS (Pimpl Translation)
    // ==============================================================================

    // Extracts the MLX array from the opaque Tensor handle
    static inline mlx::core::array unwrap(const Tensor &t) {
        return t.get_impl()->data;
    }

    // Wraps an MLX array back into an opaque Tensor handle
    static inline Tensor wrap(mlx::core::array arr) {
        return Tensor(std::make_shared<TensorImpl>(std::move(arr)));
    }

    // ==============================================================================
    // 1. ELEMENT-WISE ARITHMETIC
    // ==============================================================================

    Tensor add(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::add(unwrap(a), unwrap(b)));
    }

    Tensor subtract(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::subtract(unwrap(a), unwrap(b)));
    }

    Tensor multiply(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::multiply(unwrap(a), unwrap(b)));
    }

    Tensor divide(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::divide(unwrap(a), unwrap(b)));
    }

    Tensor floor(const Tensor &a) {
        return wrap(mlx::core::floor(unwrap(a)));
    }

    Tensor mean(const Tensor &a) {
        return wrap(mlx::core::mean(unwrap(a)));
    }

    Tensor minimum(const Tensor &a, const Tensor &b) {
        // mlx::core::minimum handles the element-wise comparison and broadcasting natively
        return wrap(mlx::core::minimum(unwrap(a), unwrap(b)));
    }

    Tensor ceil(const Tensor &a) {
        return wrap(mlx::core::ceil(unwrap(a))); //
    }

    Tensor round(const Tensor &a) {
        return wrap(mlx::core::round(unwrap(a))); //
    }

    Tensor clamp(const Tensor &a, float min_val, float max_val) {
        // MLX uses 'clip' for clamping operations
        return wrap(mlx::core::clip(unwrap(a),
                                    mlx::core::array(min_val),
                                    mlx::core::array(max_val))); //
    }

    // ==============================================================================
    // 2. CORE MATRIX OPERATIONS
    // ==============================================================================

    Tensor matmul(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::matmul(unwrap(a), unwrap(b)));
    }

    Tensor transpose(const Tensor &a, const std::vector<int> &axes) {
        // In MLX, calling transpose without axes swaps the last two dimensions by default,
        // which is exactly what we need for batched [N, d, d] skew-symmetric calculations.
        return wrap(mlx::core::transpose(unwrap(a), axes));
    }

    Tensor eye(int d, DType dtype) {
        return wrap(mlx::core::eye(d, get_mlx_dtype(dtype)));
    }

    Tensor expand_dims(const Tensor &a, const std::vector<int> &axes) {
        return wrap(mlx::core::expand_dims(unwrap(a), axes));
    }

    Tensor reshape(const Tensor &a, const std::vector<int> &shape) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());
        return wrap(mlx::core::reshape(unwrap(a), mlx_shape));
    }

    Tensor broadcast_to(const Tensor &a, const std::vector<int> &shape) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());
        return wrap(mlx::core::broadcast_to(unwrap(a), mlx_shape));
    }

    Tensor array(const std::vector<float> &data, const std::vector<int> &shape, DType dtype) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());
        return wrap(mlx::core::array(data.data(), mlx_shape, get_mlx_dtype(dtype)));
    }

    Tensor array(const std::vector<double> &data, const std::vector<int> &shape, DType dtype) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());
        return wrap(mlx::core::array(data.data(), mlx_shape, get_mlx_dtype(dtype)));
    }

    Tensor astype_int32(const Tensor &a) {
        // Casts to int32 entirely on the GPU (No CPU sync!)
        return wrap(mlx::core::astype(unwrap(a), mlx::core::int32));
    }

    Tensor gather(const Tensor &a, const Tensor &indices, int axis) {
        // MLX natively uses 'take' for gather operations
        return wrap(mlx::core::take(unwrap(a), unwrap(indices), axis));
    }

    Tensor stack(const std::vector<Tensor> &tensors, int axis) {
        std::vector<mlx::core::array> arrs;
        arrs.reserve(tensors.size());
        for (const auto &t : tensors) {
            arrs.push_back(unwrap(t));
        }
        return wrap(mlx::core::stack(arrs, axis)); //
    }

    Tensor squeeze(const Tensor &a, const std::vector<int> &axes) {
        return wrap(mlx::core::squeeze(unwrap(a), axes)); //
    }

    Tensor full(const std::vector<int> &shape, float value, DType dtype) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());
        return wrap(mlx::core::full(mlx_shape, value, get_mlx_dtype(dtype))); //
    }

    // ==============================================================================
    // 3. LOGICAL OPERATIONS
    // ==============================================================================

    Tensor where(const Tensor &condition, const Tensor &x, const Tensor &y) {
        return wrap(mlx::core::where(unwrap(condition), unwrap(x), unwrap(y))); //
    }

    Tensor equal(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::equal(unwrap(a), unwrap(b))); //
    }

    Tensor not_equal(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::not_equal(unwrap(a), unwrap(b))); //
    }

    Tensor greater(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::greater(unwrap(a), unwrap(b))); //
    }

    Tensor less(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::less(unwrap(a), unwrap(b))); //
    }

    Tensor greater_equal(const Tensor &a, const Tensor &b) {
        // MLX handles broadcasting and GPU dispatch natively
        return wrap(mlx::core::greater_equal(unwrap(a), unwrap(b)));
    }

    Tensor less_equal(const Tensor &a, const Tensor &b) {
        // Maps directly to the Metal 'less_equal' kernel
        return wrap(mlx::core::less_equal(unwrap(a), unwrap(b)));
    }

    Tensor logical_and(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::logical_and(unwrap(a), unwrap(b))); //
    }

    Tensor logical_or(const Tensor &a, const Tensor &b) {
        return wrap(mlx::core::logical_or(unwrap(a), unwrap(b))); //
    }

    // ==============================================================================
    // 4. REDUCTIONS & NON-LINEARITIES
    // ==============================================================================

    Tensor sum(const Tensor &a, const std::vector<int> &axes) {
        if (axes.empty()) {
            return wrap(mlx::core::sum(unwrap(a)));
        }
        return wrap(mlx::core::sum(unwrap(a), axes));
    }

    Tensor cumsum(const Tensor &a, int axis) {
        // MLX handles the scan operation efficiently on the active device.
        return wrap(mlx::core::cumsum(unwrap(a), axis));
    }

    Tensor min(const Tensor &a) {
        return wrap(mlx::core::min(unwrap(a)));
    }

    Tensor exp(const Tensor &a) {
        return wrap(mlx::core::exp(unwrap(a)));
    }

    Tensor log(const Tensor &a) {
        return wrap(mlx::core::log(unwrap(a)));
    }

    Tensor square(const Tensor &a) {
        return wrap(mlx::core::square(unwrap(a)));
    }

    Tensor sqrt(const Tensor &a) {
        return wrap(mlx::core::sqrt(unwrap(a)));
    }

    Tensor abs(const Tensor &a) {
        return wrap(mlx::core::abs(unwrap(a)));
    }

    Tensor sin(const Tensor &a) {
        return wrap(mlx::core::sin(unwrap(a)));
    }

    Tensor cos(const Tensor &a) {
        return wrap(mlx::core::cos(unwrap(a)));
    }

    Tensor asin(const Tensor &a) {
        // MLX natively uses arcsin
        return wrap(mlx::core::arcsin(unwrap(a)));
    }

    Tensor acos(const Tensor &a) {
        // MLX natively uses arccos
        return wrap(mlx::core::arccos(unwrap(a)));
    }

    Tensor atan(const Tensor &a) {
        // MLX natively uses arctan
        return wrap(mlx::core::arctan(unwrap(a)));
    }

    Tensor argmax(const Tensor &a, int axis) {
        // MLX returns the index as a uint32/int32 array
        return wrap(mlx::core::argmax(unwrap(a), axis));
    }

    Tensor max(const Tensor &a) {
        return wrap(mlx::core::max(unwrap(a)));
    }

    Tensor prod(const Tensor &a, const std::vector<int> &axes) {
        return wrap(mlx::core::prod(unwrap(a), axes)); //
    }

    Tensor all(const Tensor &a, const std::vector<int> &axes) {
        return wrap(mlx::core::all(unwrap(a), axes)); //
    }

    Tensor any(const Tensor &a, const std::vector<int> &axes) {
        return wrap(mlx::core::any(unwrap(a), axes)); //
    }

    Tensor pow(const Tensor &a, float exponent) {
        return wrap(mlx::core::power(unwrap(a), mlx::core::array(exponent))); //
    }

    Tensor tan(const Tensor &a) {
        return wrap(mlx::core::tan(unwrap(a))); //
    }

    Tensor atan2(const Tensor &y, const Tensor &x) {
        return wrap(mlx::core::arctan2(unwrap(y), unwrap(x))); //
    }

    Tensor rfft(const Tensor &a, int n, int axis) {
        probe_fft_support();

        // Select CPU explicitly if GPU implementation is missing
        auto device = g_fft_gpu_supported ? mlx::core::default_device() : mlx::core::Device::cpu;
        mlx::core::array arr = unwrap(a);

        if (n == -1) {
            return wrap(mlx::core::fft::rfft(arr, axis, mlx::core::fft::FFTNorm::Backward, device));
        } else {
            return wrap(mlx::core::fft::rfft(arr, n, axis, mlx::core::fft::FFTNorm::Backward, device));
        }
    }

    Tensor irfft(const Tensor &a, int n, int axis) {
        probe_fft_support();

        auto device = g_fft_gpu_supported ? mlx::core::default_device() : mlx::core::Device::cpu;
        mlx::core::array arr = unwrap(a);

        if (n == -1) {
            return wrap(mlx::core::fft::irfft(arr, axis, mlx::core::fft::FFTNorm::Backward, device));
        } else {
            return wrap(mlx::core::fft::irfft(arr, n, axis, mlx::core::fft::FFTNorm::Backward, device));
        }
    }

    Tensor norm(const Tensor &a, const std::vector<int> &axes) {
        // mlx::core::linalg::norm defaults to the Frobenius norm for matrices
        // (or L2 for vectors) when no 'ord' is specified.
        if (axes.empty()) {
            return wrap(mlx::core::linalg::norm(unwrap(a)));
        }
        return wrap(mlx::core::linalg::norm(unwrap(a), axes));
    }

    // ==============================================================================
    // 5. HEAVY LINEAR ALGEBRA
    // ==============================================================================

    static mlx::core::array custom_gpu_solve_impl(const mlx::core::array& a, const mlx::core::array& b,
                                                 int max_iters, float tol) {
        // 1. Dynamic Norm and Omega Calculation
        mlx::core::array abs_a = mlx::core::abs(a);
        mlx::core::array row_sums = mlx::core::sum(abs_a, {-1}, true);
        mlx::core::array norm_a = mlx::core::max(row_sums, {-2}, true);

        mlx::core::array epsilon = mlx::core::array(1e-7f, a.dtype());
        mlx::core::array safe_norm = mlx::core::maximum(norm_a, epsilon);
        mlx::core::array omega = mlx::core::divide(mlx::core::array(1.0f, a.dtype()), safe_norm);

        // 2. Initial State: Compute initial residual before the loop
        mlx::core::array x = mlx::core::multiply(omega, b);
        mlx::core::array ax_init = mlx::core::matmul(a, x);
        mlx::core::array residual = mlx::core::subtract(b, ax_init);

        const int check_interval = 5;

        for (int i = 0; i < max_iters; i += check_interval) {
            // 3. Iterative Chunking: Prevents excessive CPU-GPU context switching
            for (int j = 0; j < check_interval && (i + j) < max_iters; ++j) {
                // Update rule: X_{k+1} = X_k + omega * residual
                x = mlx::core::add(x, mlx::core::multiply(omega, residual));

                // Refresh residual for the next step
                mlx::core::array ax = mlx::core::matmul(a, x);
                residual = mlx::core::subtract(b, ax);
            }

            // --- ASYNCHRONOUS CONVERGENCE CHECK ---
            // Only 'eval' every few steps to maintain GPU throughput
            mlx::core::array max_err_arr = mlx::core::max(mlx::core::abs(residual));
            mlx::core::eval({max_err_arr, x});

            if (max_err_arr.item<float>() < tol) {
                break;
            }
        }

        return x;
    }

    Tensor solve(const Tensor &a, const Tensor &b) {
        // Check the active execution stream/device.
        // If running on the GPU, route to the asynchronous, graph-native iterative solver.
        if (mlx::core::default_device() == mlx::core::Device::gpu) {
            // 15 iterations is typically sufficient for the strongly diagonally dominant
            // denominator matrix (V - U) generated in the Padé approximant.
            return wrap(custom_gpu_solve_impl(unwrap(a), unwrap(b), 15, 1e-5f));
        }
        // If explicitly running on the CPU, fall back to the native LAPACK implementation.
        else {
            return wrap(mlx::core::linalg::solve(unwrap(a), unwrap(b), mlx::core::Device::cpu));
        }
    }

    std::tuple<Tensor, Tensor, Tensor> svd(const Tensor &a) {
        mlx::core::array arr = unwrap(a);
        mlx::core::Dtype orig_dtype = arr.dtype();

        // 1. Upcast to Float32 if we are running in lower precision
        bool needs_cast = (orig_dtype == mlx::core::float16 || orig_dtype == mlx::core::bfloat16);
        if (needs_cast) {
            arr = mlx::core::astype(arr, mlx::core::float32);
        }

        // 2. Perform the SVD on the CPU (MLX requires this for linalg::svd)
        auto result = mlx::core::linalg::svd(arr, mlx::core::Device::cpu);

        mlx::core::array U = result[0];
        mlx::core::array S = result[1];
        mlx::core::array Vt = result[2];

        // 3. Safely cast the results back to the original precision
        if (needs_cast) {
            U = mlx::core::astype(U, orig_dtype);
            S = mlx::core::astype(S, orig_dtype);
            Vt = mlx::core::astype(Vt, orig_dtype);
        }

        //std::cout << mlx::core::matmul(U, Vt) << std::endl;

        return {wrap(U), wrap(S), wrap(Vt)};
    }

    std::tuple<Tensor, Tensor> eigh(const Tensor &a) {
        mlx::core::array arr = unwrap(a);
        mlx::core::Dtype orig_dtype = arr.dtype();

        // Upcast low-precision inputs — eigh requires at least float32.
        bool needs_cast = (orig_dtype == mlx::core::float16 ||
                           orig_dtype == mlx::core::bfloat16);
        if (needs_cast) {
            arr = mlx::core::astype(arr, mlx::core::float32);
        }

        // Native MLX batched eigh (CPU evaluated for linalg ops)
        // Using C++17 structured binding to unpack the std::pair directly
        auto [vals, vecs] = mlx::core::linalg::eigh(arr, "L", mlx::core::Device::cpu);

        if (needs_cast) {
            vals = mlx::core::astype(vals, orig_dtype);
            vecs = mlx::core::astype(vecs, orig_dtype);
        }

        return {wrap(vals), wrap(vecs)};
    }

    std::tuple<Tensor, Tensor> qr(const Tensor &a) {
        if (mlx::core::default_device() == mlx::core::Device::gpu) {
            auto [q, r] = custom_math::qr_accelerated(unwrap(a));
            return {wrap(q), wrap(r)};
        }
        auto [q, r] = mlx::core::linalg::qr(unwrap(a), mlx::core::Device::cpu);
        return {wrap(q), wrap(r)};
    }

    Tensor eigvalsh(const Tensor &a) {
        mlx::core::array arr = unwrap(a);
        mlx::core::Dtype orig_dtype = arr.dtype();

        // Upcast low-precision inputs — eigvalsh requires at least float32.
        bool needs_cast = (orig_dtype == mlx::core::float16 ||
                           orig_dtype == mlx::core::bfloat16);
        if (needs_cast)
            arr = mlx::core::astype(arr, mlx::core::float32);

        // LAPACK dsyevd via MLX (CPU only for linalg ops).
        mlx::core::array vals =
            mlx::core::linalg::eigvalsh(arr, "L", mlx::core::Device::cpu);

        if (needs_cast)
            vals = mlx::core::astype(vals, orig_dtype);

        return wrap(vals);
    }

#include <dispatch/dispatch.h> // Apple's native parallelization library

    // ==============================================================================
    // CUSTOM DETERMINANT (Arbitrary Dimensions via Pure C++ Parallel Gaussian Elimination)
    // ==============================================================================
    Tensor det(const Tensor &a) {
        mlx::core::array arr = unwrap(a);

        // Safety: Force contiguity before reading raw pointer
        arr = mlx::core::multiply(arr, mlx::core::array(1.0f, arr.dtype()));

        // 1. Force evaluation so the GPU finishes writing to unified memory
        mlx::core::eval({arr});

        auto shape = arr.shape();
        int ndim = shape.size();
        if (ndim < 2) {
            throw std::runtime_error("[isomorphism] det requires at least a 2D matrix.");
        }
        int d = shape.back();

        // Calculate total number of matrices based on arbitrary batch dimensions
        int num_matrices = 1;
        for (int i = 0; i < ndim - 2; ++i) {
            num_matrices *= shape[i];
        }

        // Upcast to float32 on the CPU side to safely read data
        mlx::core::array f32_arr = mlx::core::astype(arr, mlx::core::float32);
        mlx::core::eval({f32_arr});
        const float *data = f32_arr.data<float>();

        std::vector<float> dets(num_matrices);

        // 2. Pure C++ Parallelization
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 8; // Safe fallback
        std::vector<std::thread> threads;

        auto worker = [&](int start_idx, int end_idx) {
            for (int k = start_idx; k < end_idx; ++k) {
                std::vector<std::vector<float> > M(d, std::vector<float>(d));
                for (int i = 0; i < d; ++i) {
                    for (int j = 0; j < d; ++j) {
                        M[i][j] = data[k * d * d + i * d + j];
                    }
                }

                float det_val = 1.0f;
                for (int i = 0; i < d; ++i) {
                    int pivot = i;
                    for (int j = i + 1; j < d; ++j) {
                        if (std::abs(M[j][i]) > std::abs(M[pivot][i])) {
                            pivot = j;
                        }
                    }

                    if (pivot != i) {
                        std::swap(M[i], M[pivot]);
                        det_val = -det_val;
                    }

                    if (std::abs(M[i][i]) < 1e-6f) {
                        det_val = 0.0f;
                        break;
                    }

                    det_val *= M[i][i];

                    for (int j = i + 1; j < d; ++j) {
                        float factor = M[j][i] / M[i][i];
                        for (int l = i + 1; l < d; ++l) {
                            M[j][l] -= factor * M[i][l];
                        }
                    }
                }
                dets[k] = det_val;
            }
        };

        // Chunk the workload across threads
        int chunk_size = num_matrices / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? num_matrices : start + chunk_size;
            threads.emplace_back(worker, start, end);
        }

        // Wait for all threads to finish
        for (auto &th: threads) {
            th.join();
        }

        // 3. Reshape back to the original batch dimensions
        std::vector<int> out_shape(shape.begin(), shape.end() - 2);
        mlx::core::Shape mlx_out_shape(out_shape.begin(), out_shape.end());

        mlx::core::array res_f32 = mlx::core::array(0.0f);
        if (out_shape.empty()) {
            res_f32 = mlx::core::array(dets[0]);
        } else {
            res_f32 = mlx::core::array(dets.data(), mlx_out_shape, mlx::core::float32);
        }

        // Restore original precision
        return wrap(mlx::core::astype(res_f32, arr.dtype()));
    }

    Tensor inv(const Tensor &a) {
        // MLX performs matrix inversion on the CPU via its linalg extension
        return wrap(mlx::core::linalg::inv(unwrap(a), mlx::core::Device::cpu)); //
    }

    // ==============================================================================
    // MATRIX EXPONENTIAL — Adaptive Diagonal Padé (Blanes, Kopylov, Seydaoglu 2024)
    //
    // Uses diagonal Padé approximants r_{m,m}(A) only, which are the unique
    // rational approximants satisfying r_{m,m}(-x) = 1/r_{m,m}(x).  This mirrors
    // the Lie algebra / Lie group duality: if A ∈ so(n) then exp(A) ∈ SO(n), and
    // the same holds for the Padé approximant.  (Taylor series does not have this
    // property and can drift off the manifold.)
    //
    // For float32 (u ≈ 2^{-24}), the backward-error thresholds scale as
    //   θ_m(fp32) / θ_m(fp64) ≈ (2^29)^{1/(2m)}
    // giving the following single-precision thresholds for the Padé orders used:
    //   m=3: θ ≈ 0.42   (2 matmuls + 1 solve)
    //   m=5: θ ≈ 1.90   (3 matmuls + 1 solve)
    //   m=7: θ ≈ 3.70   (4 matmuls + 1 solve)
    // For ||A||₁ > θ_7, we scale A → A/2^s until ||A/2^s||₁ ≤ θ_7, then square
    // back.  Typical 2k×2k shape matrices (k≈100) need s ≤ 3 squarings, giving
    // ~6 total operations vs ~20 for the 16-term Taylor + squaring approach.
    // ==============================================================================

    Tensor matrix_exp(const Tensor &a) {
        mlx::core::array arr = unwrap(a);
        mlx::core::Dtype orig_dtype = arr.dtype();

        auto shape = arr.shape();
        int ndim = shape.size();
        int d = shape.back(); // Matrix dimension
        const mlx::core::Dtype f32 = mlx::core::float32;

        // Upcast low-precision inputs to float32 for stable Padé computation.
        bool needs_cast = (orig_dtype == mlx::core::float16 ||
                           orig_dtype == mlx::core::bfloat16);
        if (needs_cast)
            arr = mlx::core::astype(arr, f32);

        // --- Batched 1-norm: max absolute column sum ---
        // For a skew-symmetric A, ||A||₁ = ||A||_∞.
        // We compute the norm for every matrix in the batch, then find the global
        // maximum to determine a single scaling factor for the entire GPU workload.
        auto abs_a = mlx::core::abs(arr);
        auto col_sums = mlx::core::sum(abs_a, {ndim - 2});      // Sum over matrix rows (axis -2)
        auto max_col_sums = mlx::core::max(col_sums, {ndim - 2}); // Max over matrix columns (axis -1)
        auto global_max_arr = mlx::core::max(max_col_sums);      // Global max across entire batch

        // The "Compute Trigger": Sync once to decide Padé parameters for all N samples.
        mlx::core::eval({global_max_arr});
        double norm1 = static_cast<double>(global_max_arr.item<float>());

        // --- Single-precision backward-error thresholds (derived in paper §2) ---
        constexpr double kTheta3 = 0.42;
        constexpr double kTheta5 = 1.90;
        constexpr double kTheta7 = 3.70;

        // --- Choose scaling s so that ||A/2^s||₁ ≤ kTheta7, then pick cheapest m ---
        int s = 0;
        double scaled_norm = norm1;
        while (scaled_norm > kTheta7) { scaled_norm /= 2.0; ++s; }
        const int m = (scaled_norm <= kTheta3) ? 3 : (scaled_norm <= kTheta5) ? 5 : 7;

        // --- B = A / 2^s ---
        mlx::core::array B = arr;
        if (s > 0) {
            float inv_s = 1.0f / std::pow(2.0f, static_cast<float>(s));
            B = mlx::core::multiply(arr, mlx::core::array(inv_s, f32));
        }

        // --- Build Padé numerator p(B) = V + U and denominator q(B) = V - U ---
        // All operations below are automatically vectorized across the batch dimension.
        auto sc = [&](float c, mlx::core::array X) {
            return mlx::core::multiply(X, mlx::core::array(c, f32));
        };

        // eye(d) broadcasts across the [..., d, d] batch automatically.
        mlx::core::array I_d = mlx::core::eye(d, f32);
        mlx::core::array B2 = mlx::core::matmul(B, B);

        mlx::core::array U = mlx::core::array(0.0f);
        mlx::core::array V = mlx::core::array(0.0f);

        if (m == 3) {
            U = mlx::core::matmul(B, mlx::core::add(sc(60.f, I_d), sc(1.f, B2)));
            V = mlx::core::add(sc(120.f, I_d), sc(12.f, B2));
        } else if (m == 5) {
            mlx::core::array B4 = mlx::core::matmul(B2, B2);
            U = mlx::core::matmul(B,
                    mlx::core::add(sc(15120.f, I_d),
                    mlx::core::add(sc(420.f,   B2),
                                   sc(1.f,     B4))));
            V =     mlx::core::add(sc(30240.f, I_d),
                    mlx::core::add(sc(3360.f,  B2),
                                   sc(30.f,    B4)));
        } else {  // m == 7
            mlx::core::array B4 = mlx::core::matmul(B2, B2);
            mlx::core::array B6 = mlx::core::matmul(B4, B2);
            U = mlx::core::matmul(B,
                    mlx::core::add(sc(8648640.f, I_d),
                    mlx::core::add(sc(277200.f,  B2),
                    mlx::core::add(sc(1512.f,    B4),
                                   sc(1.f,       B6)))));
            V =     mlx::core::add(sc(17297280.f, I_d),
                    mlx::core::add(sc(1995840.f,  B2),
                    mlx::core::add(sc(25200.f,    B4),
                                   sc(56.f,       B6))));
        }

        // r_{m,m}(B) = (V - U)^{-1} (V + U) via solve [avoids explicit inversion]
        // We utilize the custom iterative GPU solver for batched numerical stability.
        Tensor T_V_minus_U = wrap(mlx::core::subtract(V, U));
        Tensor T_V_plus_U  = wrap(mlx::core::add(V, U));
        Tensor T_result    = solve(T_V_minus_U, T_V_plus_U);
        //Tensor T_result    = mlx::core::linalg::solve(unwrap(T_V_minus_U), unwrap(T_V_plus_U), mlx::core::Device::cpu);
        mlx::core::array result = unwrap(T_result);

        // --- Squaring phase: exp(A) = r_{m,m}(B)^{2^s} ---
        for (int i = 0; i < s; ++i)
            result = mlx::core::matmul(result, result);

        if (needs_cast)
            result = mlx::core::astype(result, orig_dtype);

        return wrap(result);
    }

    // ==============================================================================
    // MATRIX LOGARITHM — Inverse Scaling-and-Squaring via Min-Max Computation Graphs
    //
    // Based on "Polynomial approximations for the matrix logarithm with
    // computation graphs" (Jarlebring, Sastre, Javier, 2024).
    //
    // Phase 1: Scale A to near-identity using Denman-Beavers iteration until
    //          ||A - I|| <= 0.1 (Theta_5 threshold).
    // Phase 2: Compute degree-32 polynomial approximation of log(I+X) using the
    //          k=5 min-max optimized computation graph (5 matmuls).
    // Phase 3: Scale back log(A) = 2^s * log(A^{1/2^s}).
    // ==============================================================================

    Tensor matrix_log(const Tensor &a) {
        mlx::core::array arr = unwrap(a);
        mlx::core::Dtype orig_dtype = arr.dtype();

        auto shape = arr.shape();
        int ndim = static_cast<int>(shape.size());
        int d = shape.back();
        const mlx::core::Dtype f32 = mlx::core::float32;

        bool needs_cast = (orig_dtype == mlx::core::float16 ||
                           orig_dtype == mlx::core::bfloat16);
        if (needs_cast)
            arr = mlx::core::astype(arr, f32);

        // ----------------------------------------------------------------------
        // CASE: d = 2 (Analytical SO(2) Solution)
        // ----------------------------------------------------------------------
        if (d == 2) {
            // Helper to slice a specific element (r, c) out of the batched matrices
            auto get_elem = [&](int r, int c) {
                std::vector<int> st(ndim, 0);
                std::vector<int> sp(shape.begin(), shape.end());
                std::vector<int> strides_vec(ndim, 1);

                st[ndim - 2] = r; sp[ndim - 2] = r + 1;
                st[ndim - 1] = c; sp[ndim - 1] = c + 1;

                // Convert std::vector<int> to mlx::core::Shape as required by the C++ API
                mlx::core::Shape mlx_st(st.begin(), st.end());
                mlx::core::Shape mlx_sp(sp.begin(), sp.end());
                mlx::core::Shape mlx_strides(strides_vec.begin(), strides_vec.end());

                // Pass the Shape objects to slice
                return mlx::core::squeeze(
                    mlx::core::slice(arr, mlx_st, mlx_sp, mlx_strides),
                    {ndim - 2, ndim - 1}
                );
            };

            // R = [[A, -B], [B, A]]. Extract A and B.
            auto A = get_elem(0, 0); // cos(theta)
            auto B = get_elem(1, 0); // sin(theta)

            auto theta = mlx::core::arctan2(B, A);
            auto zero = mlx::core::zeros_like(theta);
            auto neg_theta = mlx::core::negative(theta);

            // Construct skew-symmetric matrix: [[0, -theta], [theta, 0]]
            auto row0 = mlx::core::stack({zero, neg_theta}, -1);
            auto row1 = mlx::core::stack({theta, zero}, -1);
            auto result = mlx::core::stack({row0, row1}, -2);

            // Restore Original Precision
            if (needs_cast) {
                result = mlx::core::astype(result, orig_dtype);
            }

            return wrap(result);
        }
        // ----------------------------------------------------------------------
        // CASE: d = 3 (Analytical SO(3) Solution via Inverse Rodrigues)
        // ----------------------------------------------------------------------
        else if (d == 3) {
            // 1. Calculate trace(R)
            auto trace_R = mlx::core::sum(mlx::core::diagonal(arr, 0, ndim - 2, ndim - 1), {-1});

            // 2. Extract angle theta = arccos((trace - 1) / 2)
            auto val = mlx::core::divide(
                mlx::core::subtract(trace_R, mlx::core::array(1.0f, f32)),
                mlx::core::array(2.0f, f32)
            );
            // Clamp to [-1, 1] to prevent NaN in arccos due to float precision
            val = mlx::core::clip(val, mlx::core::array(-1.0f, f32), mlx::core::array(1.0f, f32));
            auto theta = mlx::core::arccos(val);
            auto sin_theta = mlx::core::sin(theta);

            // 3. Compute R - R^T
            std::vector<int> transp_axes(ndim);
            std::iota(transp_axes.begin(), transp_axes.end(), 0);
            std::swap(transp_axes[ndim - 1], transp_axes[ndim - 2]);
            auto R_T = mlx::core::transpose(arr, transp_axes);
            auto diff = mlx::core::subtract(arr, R_T);

            // 4. Compute multiplier = theta / (2 * sin(theta))
            // To prevent 0/0 division when theta ~ 0, we use a threshold and Taylor expansion
            auto eps = mlx::core::array(1e-4f, f32);
            auto is_zero = mlx::core::less(mlx::core::abs(theta), eps);

            // Taylor expansion of x / (2*sin(x)) near x=0 is 0.5 + x^2 / 12
            auto theta_sq = mlx::core::square(theta);
            auto taylor_mult = mlx::core::add(
                mlx::core::array(0.5f, f32),
                mlx::core::divide(theta_sq, mlx::core::array(12.0f, f32))
            );

            // Standard exact multiplier for safe zones
            // Note: This implementation assumes theta < pi - eps. Handling exact pi
            // rotations robustly in a pure vectorized graph requires an eigenvector branch.
            auto safe_sin = mlx::core::where(is_zero, eps, sin_theta);
            auto exact_mult = mlx::core::divide(theta, mlx::core::multiply(mlx::core::array(2.0f, f32), safe_sin));

            auto mult = mlx::core::where(is_zero, taylor_mult, exact_mult);

            // Expand dimensions to broadcast multiplier across the [..., 3, 3] matrices
            auto mult_expanded = mlx::core::expand_dims(mlx::core::expand_dims(mult, -1), -1);
            auto result = mlx::core::multiply(mult_expanded, diff);

            // Restore Original Precision
            if (needs_cast) {
                result = mlx::core::astype(result, orig_dtype);
            }

            return wrap(result);
        }

        // --- Denman-Beavers matrix square root ---
        auto denman_beavers = [&](mlx::core::array Y) -> mlx::core::array {
            mlx::core::array Z = mlx::core::eye(d, f32);
            for (int iter = 0; iter < 6; ++iter) {
                mlx::core::array Yinv = mlx::core::linalg::inv(Y, mlx::core::Device::cpu);
                mlx::core::array Zinv = mlx::core::linalg::inv(Z, mlx::core::Device::cpu);
                mlx::core::array Ynew = mlx::core::multiply(
                    mlx::core::add(Y, Zinv), mlx::core::array(0.5f, f32));
                Z = mlx::core::multiply(
                    mlx::core::add(Z, Yinv), mlx::core::array(0.5f, f32));
                Y = Ynew;
                mlx::core::eval({Y, Z});
            }
            return Y;
        };

        // ----------------------------------------------------------------------
        // CASE: d > 3 (General Fallback - Min-Max Polynomial)
        // ----------------------------------------------------------------------

        // --- Phase 1: Scale down to near-identity (Theta_5 = 0.1 threshold) ---
        const double kTheta5 = 0.1;
        int s = 0;
        mlx::core::array As = arr;
        while (true) {
            mlx::core::array diff    = mlx::core::subtract(As, mlx::core::eye(d, f32));
            mlx::core::array abs_d   = mlx::core::abs(diff);
            mlx::core::array col_sum = mlx::core::sum(abs_d, {ndim - 2});
            mlx::core::array row_max = mlx::core::max(col_sum, {ndim - 2});
            mlx::core::array g_max   = mlx::core::max(row_max);
            mlx::core::eval({g_max});
            double norm1 = static_cast<double>(g_max.item<float>());

            if (norm1 <= kTheta5 || s >= 16) break;

            As = denman_beavers(As);
            ++s;
        }

        // --- Phase 2: k=5 Min-Max Computation Graph Evaluation ---
        // We approximate f(A) = -log(I-X).
        // Therefore, log(A_s) = log(I + (A_s - I)) approx -z(-(A_s - I)) = -z(I - A_s)
        mlx::core::array X = mlx::core::subtract(mlx::core::eye(d, f32), As);

        // Helper for scalar multiplication
        auto sc = [&](float c, const mlx::core::array& M) {
            return mlx::core::multiply(M, mlx::core::array(c, f32));
        };

        // Node P2 and P3
        mlx::core::array P2 = X;
        mlx::core::array P3 = mlx::core::matmul(X, X);

        // Node P4
        mlx::core::array P4_h = mlx::core::add(sc(7.363757032799957e-02f, P2), sc(-1.050281301619960e+00f, P3));
        mlx::core::array P4_g = mlx::core::add(sc(-9.666134174379001e-01f, P2), sc(-4.395519034717933e-01f, P3));
        mlx::core::array P4   = mlx::core::matmul(P4_h, P4_g);

        // Node P5
        mlx::core::array P5_h = mlx::core::add(sc(8.897468955192446e-02f, P2),
                                mlx::core::add(sc(-1.599651928992725e-01f, P3), sc(9.577281350989334e-01f, P4)));
        mlx::core::array P5_g = mlx::core::add(sc(1.048664069004776e-01f, P2),
                                mlx::core::add(sc(1.585606124033259e-01f, P3), sc(1.668066506920988e-01f, P4)));
        mlx::core::array P5   = mlx::core::matmul(P5_h, P5_g);

        // Node P6
        mlx::core::array P6_h = mlx::core::add(sc(5.394999133948797e-01f, P2),
                                mlx::core::add(sc(6.700731102561937e-02f, P3),
                                mlx::core::add(sc(-5.158769100223212e-02f, P4), sc(1.094308587350110e+00f, P5))));
        mlx::core::array P6_g = mlx::core::add(sc(-8.025600931705978e-02f, P2),
                                mlx::core::add(sc(-1.159854366397558e-01f, P3),
                                mlx::core::add(sc(1.066554944706011e-01f, P4), sc(1.127094008297975e+00f, P5))));
        mlx::core::array P6   = mlx::core::matmul(P6_h, P6_g);

        // Node P7
        mlx::core::array P7_h = mlx::core::add(sc(1.027072285939197e-01f, P2),
                                mlx::core::add(sc(-8.964023050065877e-03f, P3),
                                mlx::core::add(sc(-2.100705663612491e-01f, P4),
                                mlx::core::add(sc(1.949655359168707e-01f, P5), sc(1.117368056772713e+00f, P6)))));
        mlx::core::array P7_g = mlx::core::add(sc(2.702180425508705e-01f, P2),
                                mlx::core::add(sc(4.137541209720699e-02f, P3),
                                mlx::core::add(sc(4.857347452405025e-01f, P4),
                                mlx::core::add(sc(-6.000256005636980e-01f, P5), sc(1.063393233943084e+00f, P6)))));
        mlx::core::array P7   = mlx::core::matmul(P7_h, P7_g);

        // Final linear combination z(X)
        mlx::core::array z = mlx::core::add(sc(1.0f, P2),
                             mlx::core::add(sc(5.065546620208965e-01f, P3),
                             mlx::core::add(sc(3.832512052972577e-01f, P4),
                             mlx::core::add(sc(1.088307723749078e+00f, P5),
                             mlx::core::add(sc(2.787461897212877e-01f, P6), sc(8.157421998489228e-01f, P7))))));

        // Result L_s = -z(I - A_s)
        mlx::core::array result = sc(-1.0f, z);

        // --- Phase 3: Undo Scaling ---
        if (s > 0) {
            float scale = std::pow(2.0f, static_cast<float>(s));
            result = mlx::core::multiply(result, mlx::core::array(scale, f32));
        }

        if (needs_cast)
            result = mlx::core::astype(result, orig_dtype);

        return wrap(result);
    }

    Tensor diag_embed(const Tensor &v) {
        // Creates a square diagonal matrix from a 1D tensor
        return wrap(mlx::core::diag(unwrap(v), 0));
    }

    Tensor diag_extract(const Tensor &a) {
        // Extracts the main diagonal of a 2D matrix as a 1D tensor.
        // Uses the same mlx::core::diagonal call pattern as trace().
        auto arr = unwrap(a);
        int ndim = arr.ndim();
        if (ndim < 2) return a;
        // Explicitly extract diagonal from the last two axes (Rows, Cols)
        return wrap(mlx::core::diagonal(arr, 0, ndim - 2, ndim - 1));
    }

    Tensor sign(const Tensor &a) {
        return wrap(mlx::core::sign(unwrap(a)));
    }

    Tensor trace(const Tensor &a) {
        // Trace is the sum of the diagonal elements
        auto arr = unwrap(a);
        int ndim = arr.ndim();
        if (ndim < 2) return a;
        // Extract diagonal from last two axes, then sum the resulting vector
        return wrap(mlx::core::sum(mlx::core::diagonal(arr, 0, ndim - 2, ndim - 1), {-1}));
    }

    // ==============================================================================
    // 6. STOCHASTIC GENERATION
    // ==============================================================================

    Tensor random_normal(const std::vector<int> &shape, DType dtype) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());

        return wrap(mlx::core::random::normal(mlx_shape, get_mlx_dtype(dtype)));
    }

    Tensor random_uniform(const std::vector<int> &shape, DType dtype) {
        mlx::core::Shape mlx_shape(shape.begin(), shape.end());
        // MLX uniform defaults to [0, 1)
        mlx::core::array low = mlx::core::array(0.0f, get_mlx_dtype(dtype));
        mlx::core::array high = mlx::core::array(1.0f, get_mlx_dtype(dtype));
        return wrap(mlx::core::random::uniform(low, high, mlx_shape, get_mlx_dtype(dtype)));
    }

    // ==============================================================================
    // 7. CPU-GPU BRIDGE (The Compute Trigger)
    // ==============================================================================

    double to_double(const Tensor &a) {
        mlx::core::array arr = unwrap(a);

        // Force the Apple GPU to execute the graph
        mlx::core::eval({arr});

        // Safely read the native 4-byte float first, THEN cast to C++ double
        // This prevents out-of-bounds 8-byte reads from zeroed memory arenas
        if (arr.dtype() == mlx::core::float32) {
            return static_cast<double>(arr.item<float>());
        }

        // Fallback if you ever explicitly generate Float64 tensors
        return arr.item<double>();
    }

    std::vector<float> to_float_vector(const Tensor &a) {
        mlx::core::array arr = unwrap(a);

        // FIX: Force the array to be contiguous. Multiplying by 1 allocates a fresh,
        // densely packed memory buffer, eliminating any strides from previous slices.
        arr = mlx::core::multiply(arr, mlx::core::array(1.0f, arr.dtype()));

        if (arr.dtype() != mlx::core::float32) {
            arr = mlx::core::astype(arr, mlx::core::float32);
        }

        mlx::core::eval({arr});

        const float *data_ptr = arr.data<float>();
        size_t size = arr.size();

        return std::vector<float>(data_ptr, data_ptr + size);
    }

    std::vector<double> to_double_vector(const Tensor &a) {
        mlx::core::array arr = unwrap(a);

        // FIX: Force contiguity.
        arr = mlx::core::multiply(arr, mlx::core::array(1.0f, arr.dtype()));

        if (arr.dtype() != mlx::core::float64) {
            arr = mlx::core::astype(arr, mlx::core::float64);
        }

        mlx::core::eval({arr});

        const double *data_ptr = arr.data<double>();
        size_t size = arr.size();

        return std::vector<double>(data_ptr, data_ptr + size);
    }

    int to_int(const Tensor &a) {
        mlx::core::array arr = unwrap(a);
        // Force evaluation to ensure the result is computed on the GPU
        mlx::core::eval({arr});

        // Extract the item as an integer
        return arr.item<int>();
    }

    void eval(const Tensor &a) {
        mlx::core::eval({unwrap(a)});
    }

    Tensor concatenate(const std::vector<Tensor> &tensors, int axis) {
        std::vector<mlx::core::array> arrs;
        arrs.reserve(tensors.size());
        for (const auto &t: tensors) {
            arrs.push_back(unwrap(t));
        }
        return wrap(mlx::core::concatenate(arrs, axis));
    }

    Tensor slice(const Tensor &a, int start, int end, int axis) {
        mlx::core::array arr = unwrap(a);
        int ndim = arr.ndim();

        // 1. Initialize std::vectors
        std::vector<int> starts(ndim, 0);
        auto shape = arr.shape();
        std::vector<int> stops(shape.begin(), shape.end());
        std::vector<int> strides(ndim, 1);

        // 2. Override the specific axis boundaries
        starts[axis] = start;
        stops[axis] = end;

        // 3. FIX: Cast the standard vectors to MLX Shapes
        mlx::core::Shape mlx_starts(starts.begin(), starts.end());
        mlx::core::Shape mlx_stops(stops.begin(), stops.end());
        mlx::core::Shape mlx_strides(strides.begin(), strides.end());

        // 4. Call MLX slice with the correct types
        return wrap(mlx::core::slice(arr, mlx_starts, mlx_stops, mlx_strides));
    }

    void set_default_device_cpu() {
        mlx::core::set_default_device(mlx::core::Device::cpu);
    }

    void set_default_device_gpu() {
        mlx::core::set_default_device(mlx::core::Device::gpu);
    }
} // namespace isomorphism
