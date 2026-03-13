// Path: isomorphism/src/backends/mlx/math_mlx.cpp

#include <iostream>

#include "isomorphism/math.hpp"
#include "tensor_impl_mlx.hpp" // Our private header
#include <mlx/mlx.h>
#include <thread>
#include <vector>

namespace isomorphism::math {
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
        if (n == -1) {
            // Calls the version: rfft(const array& a, int axis = -1, ...)
            return wrap(mlx::core::fft::rfft(unwrap(a), axis));
        } else {
            // Calls the version: rfft(const array& a, int n, int axis, ...)
            return wrap(mlx::core::fft::rfft(unwrap(a), n, axis));
        }
    }

    Tensor irfft(const Tensor &a, int n, int axis) {
        if (n == -1) {
            // Calls the version: irfft(const array& a, int axis = -1, ...)
            return wrap(mlx::core::fft::irfft(unwrap(a), axis));
        } else {
            // Calls the version: irfft(const array& a, int n, int axis, ...)
            return wrap(mlx::core::fft::irfft(unwrap(a), n, axis));
        }
    }

    // ==============================================================================
    // 5. HEAVY LINEAR ALGEBRA
    // ==============================================================================

    Tensor solve(const Tensor &a, const Tensor &b) {
        // The engine behind your Cayley Transform
        return wrap(mlx::core::linalg::solve(unwrap(a), unwrap(b), mlx::core::Device::cpu));
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

    std::tuple<Tensor, Tensor> qr(const Tensor &a) {
        // mlx::core::linalg::qr returns a vector of arrays: {Q, R}
        auto [q, r] = mlx::core::linalg::qr(unwrap(a), mlx::core::Device::cpu);
        return {wrap(q), wrap(r)};
    }

#include <dispatch/dispatch.h> // Apple's native parallelization library

    // ==============================================================================
    // CUSTOM DETERMINANT (Arbitrary Dimensions via Pure C++ Parallel Gaussian Elimination)
    // ==============================================================================
    Tensor det(const Tensor &a) {
        mlx::core::array arr = unwrap(a);

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

    Tensor trace(const Tensor &a) {
        // Trace is the sum of the diagonal elements
        auto arr = unwrap(a);
        int ndim = arr.ndim();
        // In MLX, we can extract the diagonal and sum it
        return wrap(mlx::core::sum(mlx::core::diagonal(arr), {ndim - 1})); //
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

        // Ensure the array is evaluated and synchronized with the CPU
        mlx::core::eval({arr});

        // Check if the data is already double; if not, cast it
        if (arr.dtype() != mlx::core::float32) {
            arr = mlx::core::astype(arr, mlx::core::float32);
            mlx::core::eval({arr});
        }

        const float *data_ptr = arr.data<float>();
        size_t size = arr.size();

        // Construct the vector using the pointer range
        return std::vector<float>(data_ptr, data_ptr + size);
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
} // namespace isomorphism
