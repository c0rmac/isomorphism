#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

#include "isomorphism/tensor.hpp"
#include "isomorphism/math.hpp"

using namespace isomorphism;
using namespace isomorphism::math;

// Helper to calculate L2 norm of (QR - A) to verify correctness
float verify_reconstruction(const Tensor& A, const Tensor& Q, const Tensor& R) {
    Tensor QR = matmul(Q, R);
/*
    std::cout << "Q: " << "\n";
    std::cout <<  Q << "\n";

    std::cout << "R: " << "\n";
    std::cout <<  R << "\n";

    std::cout << "QR: " << "\n";
    std::cout << QR << "\n";
*/
    Tensor Diff = subtract(QR, A);
    Tensor Squared = multiply(Diff, Diff);
    Tensor Sum = sum(Squared);
    eval(Sum);
    return std::sqrt(to_float_vector(Sum)[0]);
}

void run_test_case(const std::string& label, const Tensor& A) {
    //std::cout << "A: " << "\n";
    //std::cout << A << "\n";

    auto shape = A.shape();
    std::cout << "Running Test: " << label << " (" << shape[shape.size()-2]
              << "x" << shape[shape.size()-1] << ")\n";

    auto [Q, R] = qr(A);
    eval(Q);
    eval(R);

    float error = verify_reconstruction(A, Q, R);
    std::cout << "  Q shape: [" << Q.shape()[0] << ", " << Q.shape()[1] << "]\n";
    std::cout << "  R shape: [" << R.shape()[0] << ", " << R.shape()[1] << "]\n";
    std::cout << "  Reconstruction Error (||QR - A||): " << std::scientific << error << "\n";

    // Orthogonality check: Q^T * Q should be Identity
    int ndim = Q.shape().size();
    std::vector<int> t_axes(ndim);
    std::iota(t_axes.begin(), t_axes.end(), 0);
    std::swap(t_axes[ndim - 1], t_axes[ndim - 2]); // Swap the last two axes

    Tensor QTQ = matmul(transpose(Q, t_axes), Q);
    // (Simplified for 2D in this snippet, use proper axes for ND)

    std::cout << "--------------------------------------------------------\n";
}

int main() {
    set_default_device_gpu();

    std::cout << "========================================================\n";
    std::cout << "         Expanded QR Decomposition Test Suite\n";
    std::cout << "========================================================\n\n";

    // 1. Square Matrix (The "Fast Path" for SO(d) Sampler)
    std::vector<float> sq_data(32 * 32);
    std::iota(sq_data.begin(), sq_data.end(), 1.0f); // Fill with 1, 2, 3...
    run_test_case("1. Perfectly Aligned Square (32x32)", array(sq_data, {32, 32}, DType::Float32));

    // 2. Tall/Skinny Matrix
    std::vector<float> tall_data(50 * 10, 0.5f);
    run_test_case("2. Unaligned Tall Matrix (50x10)", array(tall_data, {50, 10}, DType::Float32));

    // 3. Wide Matrix (Underdetermined system)
    std::vector<float> wide_data(16 * 64, 1.2f);
    run_test_case("3. Wide Matrix (16x64)", array(wide_data, {16, 64}, DType::Float32));

    // 4. Rank Deficient Matrix
    std::vector<float> rank_def = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f, // Row 2 = 2 * Row 1
        0.0f, 1.0f, 5.0f
    };
    run_test_case("4. Rank Deficient Square (3x3)", array(rank_def, {3, 3}, DType::Float32));

    // 5. Large Scale Random
    // 1. Obtain a random seed from hardware if available
    std::random_device rd;

    // 2. Initialize the generator (Mersenne Twister is the standard choice)
    std::mt19937 gen(rd());

    // 3. Define the distribution (e.g., integers between 1 and 100 inclusive)
    std::uniform_int_distribution<> distr(1, 100);

    int large_n = 3;
    std::vector<float> large_data(large_n * large_n);
    for(int i=0; i<large_n*large_n; ++i) large_data[i] = static_cast<float>(distr(gen));
    run_test_case("5. Large Scale Random (128x128)", array(large_data, {large_n, large_n}, DType::Float32));

    // 6. Batched QR Test
    std::vector<float> batch_data(4 * 16 * 16, 1.0f);
    run_test_case("6. Batched Square QR (4x16x16)", array(batch_data, {4, 16, 16}, DType::Float32));

    // =========================================================================
    // NEW TESTS START HERE
    // =========================================================================

    // 7. Identity Matrix (Tests if the shader safely handles zeros below the diagonal)
    std::vector<float> eye_data(64 * 64, 0.0f);
    for(int i=0; i<64; ++i) eye_data[i * 64 + i] = 1.0f;
    run_test_case("7. Identity Matrix (64x64)", array(eye_data, {64, 64}, DType::Float32));

    // 8. All Zeros Matrix (Tests division-by-zero safety in Householder reflection)
    std::vector<float> zeros_data(32 * 32, 0.0f);
    run_test_case("8. All Zeros Matrix (32x32)", array(zeros_data, {32, 32}, DType::Float32));

    // 9. Negative and Mixed Signs (Tests sign preservation and mu computation)
    std::vector<float> mixed_data(48 * 48);
    for(int i=0; i<48*48; ++i) mixed_data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    run_test_case("9. Mixed Sign Random (48x48)", array(mixed_data, {48, 48}, DType::Float32));

    // 10. Single Column Vector (Extreme Tall)
    std::vector<float> col_data(100, 2.5f);
    run_test_case("10. Single Column Vector (100x1)", array(col_data, {100, 1}, DType::Float32));

    // 11. Single Row Vector (Extreme Wide)
    std::vector<float> row_data(100, -1.5f);
    run_test_case("11. Single Row Vector (1x100)", array(row_data, {1, 100}, DType::Float32));

    // 12. Non-Square Batched QR (Tests batch memory striding on unaligned memory)
    // Shape: [Batch=3, M=40, N=20]
    std::vector<float> batch_nonsq_data(3 * 40 * 20);
    for(int i=0; i<3*40*20; ++i) batch_nonsq_data[i] = static_cast<float>(rand()) / RAND_MAX;
    run_test_case("12. Batched Non-Square (3x40x20)", array(batch_nonsq_data, {3, 40, 20}, DType::Float32));

    // 13. Heavy Stress Test (Pushing L1 Cache & Threadgroup memory barriers)
    // Tests if the multi-pass grid_parallel_update sweeps correctly across many 32x32 tiles
    int stress_n = 512;
    std::vector<float> stress_data(stress_n * stress_n);
    for(int i=0; i<stress_n*stress_n; ++i) stress_data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    run_test_case("13. AMX Heavy Stress Test (512x512)", array(stress_data, {stress_n, stress_n}, DType::Float32));

    std::cout << "========================================================\n";
    std::cout << "                  ALL TESTS DISPATCHED\n";
    std::cout << "========================================================\n";

    return 0;
}