// matrix_log_so on the GPU device against the exact logarithm and against the CPU device.
//
// The logarithm is G·W with G = V g(Λ) Vᵀ built from eigh of the symmetric part, so it is
// only as good as that eigh.  On the GPU device the eigh goes through metal_linalg, whose
// Jacobi kernels are a different algorithm from LAPACK's.  The rotations here are built on
// the host in double as R = Q·blockdiag(rot θ_k)·Qᵀ, so log R = Q·blockdiag(θ_k J)·Qᵀ is
// known exactly and neither device is the other's reference.
//
// Run with EIGH_DEVICE=cpu to force metal_linalg's CPU route: the GPU column's timing
// then matches the CPU column's, which is how to tell the default route is the GPU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "isomorphism/math.hpp"
#include "isomorphism/tensor.hpp"

using namespace isomorphism;

namespace {
constexpr auto F32 = DType::Float32;

// n rotations with principal angles uniform in [0.05, theta_max], and their exact logs.
//
// A draw whose RMS angle exceeds 2.1 is redrawn.  matrix_log_so clamps ‖L‖_F at π·√(d/2),
// which a log with RMS principal angle above π/√2 ≈ 2.22 exceeds (‖L‖²_F = 2Σθ²), and it
// rescales those on either device alike.  That clamp is not what this test is about.
void exact_rotations(int n, int d, double theta_max, std::mt19937_64 &gen,
                     std::vector<double> &R, std::vector<double> &L) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.05, theta_max);
    const size_t dd = static_cast<size_t>(d) * d;
    R.assign(n * dd, 0.0);
    L.assign(n * dd, 0.0);
    std::vector<double> Q(dd), B(dd), T(dd), QB(dd), th(d / 2);

    for (int m = 0; m < n; ++m) {
        // Q: modified Gram-Schmidt on a Gaussian matrix (columns).
        for (double &x : Q) x = nd(gen);
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                double dot = 0.0;
                for (int i = 0; i < d; ++i) dot += Q[i * d + k] * Q[i * d + j];
                for (int i = 0; i < d; ++i) Q[i * d + j] -= dot * Q[i * d + k];
            }
            double nrm = 0.0;
            for (int i = 0; i < d; ++i) nrm += Q[i * d + j] * Q[i * d + j];
            nrm = std::sqrt(nrm);
            for (int i = 0; i < d; ++i) Q[i * d + j] /= nrm;
        }
        std::fill(B.begin(), B.end(), 0.0);
        std::fill(T.begin(), T.end(), 0.0);
        double ms = 0.0;
        do {
            ms = 0.0;
            for (double &t : th) { t = ud(gen); ms += t * t; }
        } while (ms > 2.1 * 2.1 * th.size());
        for (int k = 0; k + 1 < d; k += 2) {
            const double t = th[k / 2], c = std::cos(t), s = std::sin(t);
            B[k * d + k] = c;       B[k * d + k + 1] = -s;
            B[(k + 1) * d + k] = s; B[(k + 1) * d + k + 1] = c;
            T[k * d + k + 1] = -t;  T[(k + 1) * d + k] = t;
        }
        if (d % 2) B[dd - 1] = 1.0;

        auto sandwich = [&](const std::vector<double> &M, double *out) {   // Q M Qᵀ
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) {
                    double a = 0.0;
                    for (int k = 0; k < d; ++k) a += Q[i * d + k] * M[k * d + j];
                    QB[i * d + j] = a;
                }
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) {
                    double a = 0.0;
                    for (int k = 0; k < d; ++k) a += QB[i * d + k] * Q[j * d + k];
                    out[i * d + j] = a;
                }
        };
        sandwich(B, &R[m * dd]);
        sandwich(T, &L[m * dd]);
    }
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

template <class F>
double best_ms(F &&f, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}
}  // namespace

int main() {
    struct Case { int n, d; double theta_max; };
    const Case cases[] = {
        {650, 25, 1.5}, {650, 25, 2.8},   // a swarm of particles
        {150, 15, 1.5}, {150, 15, 2.8},
        {4096, 8, 2.8},                   // many small matrices
        {1, 25, 2.8},                     // a lone consensus matrix
    };

    std::printf("matrix_log_so against the exact logarithm, GPU device and CPU device\n\n");
    std::printf("%6s %4s %6s | %10s %10s %10s %9s | %9s %9s\n", "batch", "d", "th_max",
                "gpu err", "cpu err", "|gpu-cpu|", "skew", "gpu ms", "cpu ms");
    std::printf("------------------------------------------------------------------------------------\n");

    std::mt19937_64 gen(20260930u);
    bool ok = true;
    for (const Case &c : cases) {
        std::vector<double> R_host, L_true;
        exact_rotations(c.n, c.d, c.theta_max, gen, R_host, L_true);
        const std::vector<int> shape{c.n, c.d, c.d};

        math::set_default_device_gpu();
        Tensor R_gpu = math::array(R_host, shape, F32);
        Tensor L_gpu;
        const double gpu_ms = best_ms([&] {
            L_gpu = math::matrix_log_so(R_gpu);
            math::eval(L_gpu);
        }, 5);
        const std::vector<double> Lg = math::to_double_vector(L_gpu);

        math::set_default_device_cpu();
        Tensor R_cpu = math::array(R_host, shape, F32);
        Tensor L_cpu;
        const double cpu_ms = best_ms([&] {
            L_cpu = math::matrix_log_so(R_cpu);
            math::eval(L_cpu);
        }, 5);
        const std::vector<double> Lc = math::to_double_vector(L_cpu);
        math::set_default_device_gpu();

        const double gpu_err = max_abs_diff(Lg, L_true);
        const double cpu_err = max_abs_diff(Lc, L_true);
        const double dev     = max_abs_diff(Lg, Lc);
        double skew = 0.0;
        const size_t dd = static_cast<size_t>(c.d) * c.d;
        for (int m = 0; m < c.n; ++m)
            for (int i = 0; i < c.d; ++i)
                for (int j = 0; j < c.d; ++j)
                    skew = std::max(skew, std::abs(Lg[m * dd + i * c.d + j] +
                                                   Lg[m * dd + j * c.d + i]));

        // float32 input: R carries ~6e-8 of rounding, which g = θ/sin θ amplifies by
        // ~1/sin³θ (about 30 at θ = 2.8).  The GPU path must be no worse than the CPU
        // path beyond that same order.
        const bool pass = gpu_err < 1e-4 && gpu_err < 10.0 * cpu_err + 1e-6 && skew == 0.0;
        ok = ok && pass;
        std::printf("%6d %4d %6.2f | %10.2e %10.2e %10.2e %9.1e | %9.3f %9.3f  %s\n", c.n,
                    c.d, c.theta_max, gpu_err, cpu_err, dev, skew, gpu_ms, cpu_ms,
                    pass ? "" : "FAIL");
    }

    std::printf("\n%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
