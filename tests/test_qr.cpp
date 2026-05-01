#include <iostream>
#include <iomanip>
#include <vector>

#include "isomorphism/tensor.hpp"
#include "isomorphism/math.hpp"

// Print a matrix stored row-major in a flat vector
static void print_matrix(const std::string& label,
                         const std::vector<float>& data,
                         int rows, int cols) {
    std::cout << label << " (" << rows << "x" << cols << "):\n";
    for (int r = 0; r < rows; ++r) {
        std::cout << "  [";
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << data[r * cols + c];
            if (c + 1 < cols) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
    std::cout << "\n";
}

int main() {
    using namespace isomorphism;
    using namespace isomorphism::math;

    // Use the Metal GPU path for QR
    set_default_device_gpu();

    std::cout << "========================================\n";
    std::cout << "         QR Decomposition Test\n";
    std::cout << "========================================\n\n";

    // 4x3 full-rank matrix (tall, so Q is 4x3, R is 3x3 in economy form)

    std::vector<float> data = {
        4.0f,  3.0f, -2.0f,
        2.0f,  1.0f,  0.0f,
        1.0f,  5.0f,  4.0f,
        3.0f,  2.0f,  1.0f,
    };
    const int rows = 4, cols = 3;

    Tensor A = array(data, {rows, cols}, DType::Float32);
    /*
    std::vector<float> data = {
        // Row 1 - 4
        1.2f,  0.5f, -2.1f,  3.3f,  0.0f,  1.1f, -0.5f,  4.0f,  2.2f,  0.8f,
        2.0f,  1.1f,  0.0f,  1.5f, -1.2f,  0.7f,  3.1f,  0.2f, -0.9f,  1.4f,
        1.0f,  5.2f,  4.4f, -0.3f,  2.1f,  0.9f,  1.8f,  0.5f, -2.2f,  3.0f,
        3.4f,  2.0f,  1.1f,  0.6f, -1.5f,  2.2f,  0.4f,  1.9f,  0.1f, -0.7f,

        // Row 5 - 8
        0.5f, -1.1f,  2.3f,  4.1f,  1.2f,  0.0f,  3.5f, -2.1f,  0.8f,  1.3f,
        2.2f,  3.1f, -0.4f,  1.7f,  0.9f,  2.5f,  0.1f,  1.1f, -1.8f,  0.6f,
        1.8f,  0.2f,  3.7f, -0.9f,  4.0f,  1.5f,  2.2f,  0.3f,  0.7f, -1.1f,
        0.9f,  4.4f, -1.2f,  2.8f,  0.6f,  3.1f, -0.5f,  1.2f,  2.0f,  0.4f,

        // Row 9 - 12
        4.1f,  1.3f,  0.2f, -1.5f,  3.3f,  0.7f,  1.9f,  0.5f, -0.8f,  2.2f,
        0.7f, -0.9f,  1.1f,  2.2f,  0.4f,  4.1f, -1.3f,  3.0f,  0.6f,  1.5f,
        2.5f,  3.7f, -0.6f,  0.8f,  1.1f,  2.0f,  0.4f, -2.5f,  1.3f,  0.9f,
        1.1f,  0.2f,  3.5f, -1.1f,  0.9f,  0.5f,  2.8f,  4.0f, -0.3f,  1.7f
    };
    // 0. Define the number of clones
    const int K = 5; // Replace with your desired batch size
    const int rows = 12, cols = 10;

    // 1. Create the single matrix
    Tensor A_single = array(data, {rows, cols}, DType::Float32);

    // 2. Create a vector containing K references to that matrix
    std::vector<Tensor> clones(K, A_single);

    // 3. Stack them along a new axis (0) to create a materialized [K, 12, 10] tensor
    // This ensures A_batched has a physical size of K * 12 * 10 in memory.
    Tensor A_batched = stack(clones, 0);*/

    //std::cout << "A_batched: " << A_batched << "\n";

    // 4. Perform QR
    auto [Q, R] = qr(A);
    eval(Q);
    eval(R);

    // Verification of shapes
    auto q_shape = Q.shape(); // Should be {K, 12, 10} (Economy form)
    auto r_shape = R.shape(); // Should be {K, 10, 10}

    print_matrix("Q", to_float_vector(Q), q_shape[0], q_shape[1]);
    print_matrix("R", to_float_vector(R), r_shape[0], r_shape[1]);

    // Verify: Q*R should reconstruct A
    Tensor QR = matmul(Q, R);
    eval(QR);

    std::cout << "Q*R (should match A):\n";
    std::cout << QR << "\n";
    //print_matrix("Q*R", to_float_vector(QR), rows, cols);

    return 0;
}
