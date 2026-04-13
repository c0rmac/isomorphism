#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include "isomorphism/tensor.hpp"
#include "isomorphism/math.hpp"

// Helper to compare floats with a small tolerance
inline bool is_close(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) < tol;
}

int main() {
    using namespace isomorphism;
    using namespace isomorphism::math;

    std::cout << "========================================\n";
    std::cout << "  Starting Isomorphism Backend Tests\n";
    std::cout << "========================================\n\n";

    try {
        // ---------------------------------------------------------
        // 1. INITIALIZATION & SHAPES
        // ---------------------------------------------------------
        std::cout << "[TEST] Initialization & Shapes... ";

        // Create a Batch of two 2x2 matrices: shape [2, 2, 2]
        // Matrix 0: [[1, 2], [3, 4]]
        // Matrix 1: [[5, 6], [7, 8]]
        std::vector<float> data_A = {1, 2, 3, 4, 5, 6, 7, 8};
        Tensor A = array(data_A, {2, 2, 2}, DType::Float32);

        assert(A.ndim() == 3);
        assert(A.size() == 8);
        assert(A.shape()[0] == 2 && A.shape()[1] == 2 && A.shape()[2] == 2);

        std::cout << "PASSED\n";

        // ---------------------------------------------------------
        // 2. ELEMENT-WISE MATH & BROADCASTING
        // ---------------------------------------------------------
        std::cout << "[TEST] Element-wise Math... ";

        // Add a scalar (broadcasting)
        Tensor B = add(A, Tensor(10.0, DType::Float32));
        auto vec_B = to_float_vector(B);

        // Matrix 0 should now be [[11, 12], [13, 14]]
        assert(is_close(vec_B[0], 11.0f));
        assert(is_close(vec_B[7], 18.0f)); // Last element: 8 + 10

        std::cout << "PASSED\n";

        // ---------------------------------------------------------
        // 3. AXIS REDUCTIONS
        // ---------------------------------------------------------
        std::cout << "[TEST] Axis Reductions (Sum)... ";

        // Sum across columns (axis = -1)
        // Matrix 0: [[1, 2], [3, 4]] -> row sums: [3, 7]
        // Matrix 1: [[5, 6], [7, 8]] -> row sums: [11, 15]
        Tensor S = sum(A, {-1});
        auto vec_S = to_float_vector(S);

        assert(S.shape() == std::vector<int>({2, 2})); // Reduced [2, 2, 2] -> [2, 2]
        assert(is_close(vec_S[0], 3.0f));
        assert(is_close(vec_S[1], 7.0f));
        assert(is_close(vec_S[2], 11.0f));
        assert(is_close(vec_S[3], 15.0f));

        std::cout << "PASSED\n";

        // ---------------------------------------------------------
        // 4. CORE MATRIX (BATCHED MATMUL)
        // ---------------------------------------------------------
        std::cout << "[TEST] Batched Matmul... ";

        // Create Identity Matrix Batch [2, 2, 2]
        // Matrix 0: [[2, 0], [0, 2]] (Scaled Identity)
        // Matrix 1: [[1, 0], [0, 1]] (Standard Identity)
        std::vector<float> data_I = {2, 0, 0, 2, 1, 0, 0, 1};
        Tensor I = array(data_I, {2, 2, 2}, DType::Float32);

        Tensor M = matmul(A, I);
        auto vec_M = to_float_vector(M);

        assert(M.shape() == std::vector<int>({2, 2, 2}));

        // Batch 0 Check: [[1, 2], [3, 4]] * 2 = [[2, 4], [6, 8]]
        assert(is_close(vec_M[0], 2.0f));
        assert(is_close(vec_M[3], 8.0f));

        // Batch 1 Check: [[5, 6], [7, 8]] * 1 = [[5, 6], [7, 8]]
        assert(is_close(vec_M[4], 5.0f));
        assert(is_close(vec_M[7], 8.0f));

        std::cout << "PASSED\n";

        // ---------------------------------------------------------
        // 5. HEAVY LINEAR ALGEBRA (DETERMINANT)
        // ---------------------------------------------------------
        std::cout << "[TEST] Batched Determinant... ";

        // Det of Matrix 0: (1*4) - (2*3) = 4 - 6 = -2
        // Det of Matrix 1: (5*8) - (6*7) = 40 - 42 = -2
        Tensor D = det(A);
        auto vec_D = to_float_vector(D);

        assert(D.shape() == std::vector<int>({2})); // Determinant drops the last two dimensions
        assert(is_close(vec_D[0], -2.0f));
        assert(is_close(vec_D[1], -2.0f));

        std::cout << "PASSED\n";

        // ---------------------------------------------------------
        // SUCCESS
        // ---------------------------------------------------------
        std::cout << "\n========================================\n";
        std::cout << "  ALL TESTS COMPLETED SUCCESSFULLY! \n";
        std::cout << "========================================\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}