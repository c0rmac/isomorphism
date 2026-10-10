// solve, inv, det, cholesky and solve_triangular on both devices: residuals,
// known values, shapes (vectors, broadcast batches, upper and unit triangles),
// and systems large enough for metal-linalg to put on the GPU (MLX backend).
// Inputs are made and results checked on the host, so that only the op under
// test runs on the backend; the large ones are checked against random vectors.
// Exits non-zero on a failure.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "isomorphism/tensor.hpp"
#include "isomorphism/math.hpp"

using namespace isomorphism;
using namespace isomorphism::math;

namespace {

using Vec = std::vector<float>;

int g_checks = 0, g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL: %s\n", what.c_str());
        std::fflush(stdout);
    }
}

void check_close(double got, double bound, const std::string& what) {
    check(std::isfinite(got) && got <= bound, what + " (" + std::to_string(got) + " > " + std::to_string(bound) + ")");
}

std::mt19937 g_rng(20261010);

Vec normal(size_t n, float scale = 1.0f) {
    std::normal_distribution<float> dist(0.0f, scale);
    Vec v(n);
    for (auto& x : v) x = dist(g_rng);
    return v;
}

// batch general matrices M / sqrt(n) + 2 I: well conditioned.
Vec general(int batch, int n) {
    Vec m = normal((size_t)batch * n * n, 1.0f / std::sqrt((float)n));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i) m[((size_t)b * n + i) * n + i] += 2.0f;
    return m;
}

// batch symmetric positive definite matrices: 2 I plus a symmetric random
// part of spectral radius about 1, so eigenvalues in about [1, 3].
Vec spd(int batch, int n) {
    Vec a((size_t)batch * n * n);
    std::normal_distribution<float> dist(0.0f, 0.5f / std::sqrt((float)n));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                const float x = i == j ? 2.0f : dist(g_rng);
                a[((size_t)b * n + i) * n + j] = a[((size_t)b * n + j) * n + i] = x;
            }
    return a;
}

// c = a b for a [m, k] and b [k, n] at the given offsets.
void matmul_into(const float* a, const float* b, float* c, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        float* ci = c + (size_t)i * n;
        std::fill(ci, ci + n, 0.0f);
        for (int t = 0; t < k; ++t) {
            const float ait = a[(size_t)i * k + t];
            const float* bt = b + (size_t)t * n;
            for (int j = 0; j < n; ++j) ci[j] += ait * bt[j];
        }
    }
}

// batch products a[b] x[b], a [n, n] (one shared when a_batch is 1), x [n, k].
Vec batch_matmul(const Vec& a, int a_batch, const Vec& x, int batch, int n, int k) {
    Vec c((size_t)batch * n * k);
    for (int b = 0; b < batch; ++b)
        matmul_into(a.data() + (size_t)(a_batch == 1 ? 0 : b) * n * n, x.data() + (size_t)b * n * k,
                    c.data() + (size_t)b * n * k, n, n, k);
    return c;
}

Vec transposed(const Vec& a, int batch, int rows, int cols) {
    Vec t(a.size());
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                t[((size_t)b * cols + j) * rows + i] = a[((size_t)b * rows + i) * cols + j];
    return t;
}

// max |a - b| / max(max |b|, 1), infinity for anything not finite or a size mismatch.
double rel_err(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) return INFINITY;
    double e = 0.0, m = 1.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) return INFINITY;
        e = std::max(e, (double)std::abs(a[i] - b[i]));
        m = std::max(m, (double)std::abs(b[i]));
    }
    return e / m;
}

bool all_nan(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isnan(p[i])) return false;
    return true;
}

Tensor T(const Vec& v, const std::vector<int>& shape) { return array(v, shape, DType::Float32); }

std::string shape_str(const std::vector<int>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
}

void test_solve(const std::string& dev) {
    {
        Vec a = general(4, 6), b = normal(4 * 6 * 3);
        Tensor x = solve(T(a, {4, 6, 6}), T(b, {4, 6, 3}));
        check(x.shape() == std::vector<int>{4, 6, 3}, dev + " solve shape " + shape_str(x.shape()));
        check_close(rel_err(batch_matmul(a, 4, to_float_vector(x), 4, 6, 3), b), 1e-5, dev + " solve residual");
    }
    {
        // A batch of vectors: b has a's shape without its last axis.
        Vec a = general(4, 6), b = normal(4 * 6);
        Tensor x = solve(T(a, {4, 6, 6}), T(b, {4, 6}));
        check(x.shape() == std::vector<int>{4, 6}, dev + " solve vector shape " + shape_str(x.shape()));
        check_close(rel_err(batch_matmul(a, 4, to_float_vector(x), 4, 6, 1), b), 1e-5, dev + " solve vector residual");
    }
    {
        // One matrix against a batch of right-hand sides, and a 1-D b.
        Vec a = general(1, 5), b = normal(3 * 5 * 2), v = normal(5);
        Tensor x = solve(T(a, {5, 5}), T(b, {3, 5, 2}));
        check(x.shape() == std::vector<int>{3, 5, 2}, dev + " solve broadcast shape " + shape_str(x.shape()));
        check_close(rel_err(batch_matmul(a, 1, to_float_vector(x), 3, 5, 2), b), 1e-5, dev + " solve broadcast residual");
        Tensor y = solve(T(a, {5, 5}), T(v, {5}));
        check(y.shape() == std::vector<int>{5}, dev + " solve 1-D shape " + shape_str(y.shape()));
        check_close(rel_err(batch_matmul(a, 1, to_float_vector(y), 1, 5, 1), v), 1e-5, dev + " solve 1-D residual");
    }
    {
        // Large enough for the GPU (metal-linalg's LU, from N = 1536 on the M5 Pro).
        const int n = 2048, k = 64;
        Vec a = general(1, n), b = normal((size_t)n * k);
        Tensor x = solve(T(a, {n, n}), T(b, {n, k}));
        check_close(rel_err(batch_matmul(a, 1, to_float_vector(x), 1, n, k), b), 1e-3, dev + " solve 2048 residual");
    }
}

void test_inv(const std::string& dev) {
    Vec a = general(3, 7);
    Tensor x = inv(T(a, {3, 7, 7}));
    check(x.shape() == std::vector<int>{3, 7, 7}, dev + " inv shape");
    Vec eye((size_t)3 * 49, 0.0f);
    for (int b = 0; b < 3; ++b)
        for (int i = 0; i < 7; ++i) eye[(size_t)b * 49 + i * 7 + i] = 1.0f;
    check_close(rel_err(batch_matmul(a, 3, to_float_vector(x), 3, 7, 7), eye), 1e-5, dev + " inv A A^-1 = I");
    // Large: A (A^-1 v) = v.
    const int n = 2048;
    Vec big = general(1, n), v = normal(n);
    Vec xv = batch_matmul(to_float_vector(inv(T(big, {n, n}))), 1, v, 1, n, 1);
    check_close(rel_err(batch_matmul(big, 1, xv, 1, n, 1), v), 1e-3, dev + " inv 2048: A (A^-1 v) = v");
}

void test_det(const std::string& dev) {
    // [[2,1],[1,3]]: 5. A swap: -1. Singular: 0. Upper triangular: the diagonal's product.
    Vec a = {2, 1, 1, 3, 0, 1, 1, 0, 1, 2, 2, 4, 2, 7, 0, -3};
    Vec d = to_float_vector(det(T(a, {4, 2, 2})));
    check(d.size() == 4, dev + " det shape");
    if (d.size() == 4) {
        check(std::abs(d[0] - 5.0f) < 1e-5f, dev + " det [[2,1],[1,3]] = 5, got " + std::to_string(d[0]));
        check(std::abs(d[1] + 1.0f) < 1e-6f, dev + " det of a swap = -1, got " + std::to_string(d[1]));
        check(std::abs(d[2]) < 1e-6f, dev + " det singular = 0, got " + std::to_string(d[2]));
        check(std::abs(d[3] + 6.0f) < 1e-5f, dev + " det triangular = -6, got " + std::to_string(d[3]));
    }
    // det(A B) = det(A) det(B), and a 3x3 by hand.
    Vec x = general(5, 6), y = general(5, 6);
    Vec xy = batch_matmul(x, 5, y, 5, 6, 6);
    Vec dxy = to_float_vector(det(T(xy, {5, 6, 6}))), dx = to_float_vector(det(T(x, {5, 6, 6}))),
        dy = to_float_vector(det(T(y, {5, 6, 6})));
    Vec prod(5);
    for (int i = 0; i < 5; ++i) prod[i] = dx[i] * dy[i];
    check_close(rel_err(dxy, prod), 1e-4, dev + " det(AB) = det(A) det(B)");
    const double d3 = to_double(det(T({0, 2, 1, 3, -1, 4, 1, 1, 5}, {3, 3})));
    check(std::abs(d3 + 18.0) < 1e-4, dev + " det 3x3 = -18, got " + std::to_string(d3));
    // Large enough for the GPU: A = L U with L unit lower bidiagonal and U's
    // diagonal alternately 2 and 1/2, so det A = 1, then two rows swapped: -1.
    const int n = 1536;
    Vec u = normal((size_t)n * n, 1.0f / n), l = normal(n, 0.5f), m((size_t)n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float& e = u[(size_t)i * n + j];
            e = j < i ? 0.0f : (j == i ? (i % 2 ? 0.5f : 2.0f) : e);
        }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m[(size_t)i * n + j] = u[(size_t)i * n + j] + (i ? l[i] * u[(size_t)(i - 1) * n + j] : 0.0f);
    std::swap_ranges(m.begin(), m.begin() + n, m.begin() + n);
    const double got = to_double(det(T(m, {n, n})));
    check(std::abs(got + 1.0) < 2e-2, dev + " det 1536 = -1, got " + std::to_string(got));
}

void test_cholesky(const std::string& dev) {
    for (int n : {1, 5, 33, 200}) {
        const std::string at = " n=" + std::to_string(n);
        Vec a = spd(3, n);
        Vec l = to_float_vector(cholesky(T(a, {3, n, n})));
        check_close(rel_err(batch_matmul(l, 3, transposed(l, 3, n, n), 3, n, n), a), 1e-5, dev + " cholesky L L^T" + at);
        // The upper triangle is zero, the diagonal positive.
        bool lower = l.size() == a.size(), positive = lower;
        for (int b = 0; lower && b < 3; ++b)
            for (int i = 0; i < n; ++i) {
                positive &= l[((size_t)b * n + i) * n + i] > 0;
                for (int j = i + 1; j < n; ++j) lower &= l[((size_t)b * n + i) * n + j] == 0.0f;
            }
        check(lower && positive, dev + " cholesky triangle" + at);
        Vec u = to_float_vector(cholesky(T(a, {3, n, n}), true));
        check_close(rel_err(batch_matmul(transposed(u, 3, n, n), 3, u, 3, n, n), a), 1e-5, dev + " cholesky upper U^T U" + at);
        check_close(rel_err(u, transposed(l, 3, n, n)), 1e-5, dev + " cholesky upper = L^T" + at);
    }
    {
        // Only the lower triangle is read: junk above it changes nothing.
        Vec a = spd(1, 8), junk = a;
        for (int i = 0; i < 8; ++i)
            for (int j = i + 1; j < 8; ++j) junk[i * 8 + j] = 1e6f;
        check_close(rel_err(to_float_vector(cholesky(T(junk, {8, 8}))), to_float_vector(cholesky(T(a, {8, 8})))), 1e-6,
                    dev + " cholesky reads the lower triangle only");
    }
    {
        // A matrix that is not positive definite is NaN; its neighbours are not.
        Vec good = spd(1, 4), a = good;
        const Vec bad = {1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        a.insert(a.end(), bad.begin(), bad.end());
        a.insert(a.end(), good.begin(), good.end());
        Vec l = to_float_vector(cholesky(T(a, {3, 4, 4})));
        check(l.size() == 48 && !all_nan(l.data(), 16) && all_nan(l.data() + 16, 16) && !all_nan(l.data() + 32, 16),
              dev + " cholesky: a matrix that is not positive definite is NaN, alone");
        // Failing midway (a negative pivot at 60 of 100), either triangle.
        Vec mid = spd(1, 100);
        mid[60 * 100 + 60] = -1.0f;
        Vec lm = to_float_vector(cholesky(T(mid, {100, 100}))), um = to_float_vector(cholesky(T(mid, {100, 100}), true));
        check(all_nan(lm.data(), lm.size()) && all_nan(um.data(), um.size()), dev + " cholesky: failing at pivot 60 is NaN");
    }
    {
        // Large enough for the GPU (in batches from 1536 on the M5 Pro): L (L^T v) = A v.
        const int n = 1536;
        Vec a = spd(2, n), v = normal((size_t)2 * n);
        Vec l = to_float_vector(cholesky(T(a, {2, n, n})));
        Vec llv = batch_matmul(l, 2, batch_matmul(transposed(l, 2, n, n), 2, v, 2, n, 1), 2, n, 1);
        check_close(rel_err(llv, batch_matmul(a, 2, v, 2, n, 1)), 1e-4, dev + " cholesky [2,1536,1536]");
    }
}

void test_solve_triangular(const std::string& dev) {
    for (int n : {1, 7, 130}) {
        Vec a = general(2, n), b = normal((size_t)2 * n * 3);
        for (bool upper : {false, true})
            for (bool unit : {false, true}) {
                const std::string what = dev + " solve_triangular n=" + std::to_string(n) + (upper ? " upper" : " lower") +
                                         (unit ? " unit" : "");
                // The other triangle (and with unit, the diagonal) is junk that must not be read.
                Tensor x = solve_triangular(T(a, {2, n, n}), T(b, {2, n, 3}), upper, unit);
                Vec tri = a;
                for (int k = 0; k < 2; ++k)
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n; ++j) {
                            float& e = tri[((size_t)k * n + i) * n + j];
                            if (upper ? j < i : j > i) e = 0.0f;
                            if (unit && i == j) e = 1.0f;
                        }
                check(x.shape() == std::vector<int>{2, n, 3}, what + " shape");
                check_close(rel_err(batch_matmul(tri, 2, to_float_vector(x), 2, n, 3), b), 1e-4, what + " residual");
            }
    }
    {
        // A batch of vectors, and one matrix against a batch of right-hand sides.
        Vec l = to_float_vector(cholesky(T(spd(3, 9), {3, 9, 9}))), v = normal(3 * 9);
        Tensor x = solve_triangular(T(l, {3, 9, 9}), T(v, {3, 9}));
        check(x.shape() == std::vector<int>{3, 9}, dev + " solve_triangular vector shape " + shape_str(x.shape()));
        check_close(rel_err(batch_matmul(l, 3, to_float_vector(x), 3, 9, 1), v), 1e-5, dev + " solve_triangular vector residual");
        Vec one(l.begin(), l.begin() + 81), b = normal(4 * 9 * 2);
        Tensor y = solve_triangular(T(one, {9, 9}), T(b, {4, 9, 2}));
        check(y.shape() == std::vector<int>{4, 9, 2}, dev + " solve_triangular broadcast shape " + shape_str(y.shape()));
        check_close(rel_err(batch_matmul(one, 1, to_float_vector(y), 4, 9, 2), b), 1e-5, dev + " solve_triangular broadcast residual");
    }
    {
        // With cholesky: A x = b by two triangular solves, against solve.
        Vec a = spd(2, 40), b = normal(2 * 40 * 5);
        Vec l = to_float_vector(cholesky(T(a, {2, 40, 40})));
        Tensor y = solve_triangular(T(l, {2, 40, 40}), T(b, {2, 40, 5}));
        Tensor x = solve_triangular(T(transposed(l, 2, 40, 40), {2, 40, 40}), y, true);
        check_close(rel_err(to_float_vector(x), to_float_vector(solve(T(a, {2, 40, 40}), T(b, {2, 40, 5})))), 1e-4,
                    dev + " cholesky + solve_triangular = solve");
    }
    {
        // Large enough for the GPU (N 2048 with 1024 right-hand sides on the M5
        // Pro): L (X w) = B w.
        const int n = 2048, k = 1024;
        Vec l = normal((size_t)n * n, 1.0f / n), b = normal((size_t)n * k), w = normal(k);
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j) l[(size_t)i * n + j] = i == j ? 2.0f : 0.0f;
        Vec x = to_float_vector(solve_triangular(T(l, {n, n}), T(b, {n, k})));
        Vec xw((size_t)n), bw((size_t)n);
        matmul_into(x.data(), w.data(), xw.data(), n, k, 1);
        matmul_into(b.data(), w.data(), bw.data(), n, k, 1);
        check_close(rel_err(batch_matmul(l, 1, xw, 1, n, 1), bw), 1e-3, dev + " solve_triangular 2048 x 1024");
    }
}

void run(const std::string& dev) {
    std::printf("%s device\n", dev.c_str());
    std::fflush(stdout);
    test_solve(dev);
    test_inv(dev);
    test_det(dev);
    test_cholesky(dev);
    test_solve_triangular(dev);
}

} // namespace

int main() {
    set_default_device_gpu();
    run("gpu");
    set_default_device_cpu();
    run("cpu");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
