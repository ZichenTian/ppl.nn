#include <iostream>
#include <chrono>

#include "ppl/kernel/arm_server/gemm/neon/gemm.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

void sgemm_ref(
    const float* A, 
    const float* B, 
    const int64_t M, 
    const int64_t N, 
    const int64_t K, 
    const int64_t lda, 
    const int64_t ldb, 
    const int64_t ldc, 
    float* C) {

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0;
            for (int64_t k = 0; k < K; k++) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

void test_sgemm(
    const int64_t M, 
    const int64_t N, 
    const int64_t K, 
    const bool diff = true) {

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* C_ref = nullptr;

    const int64_t A_len = M * K;
    const int64_t B_len = K * N;
    const int64_t C_len = M * N;

    A = (float*)aligned_alloc(64, A_len * sizeof(float));
    B = (float*)aligned_alloc(64, B_len * sizeof(float));
    C = (float*)aligned_alloc(64, C_len * sizeof(float));
    C_ref = (float*)aligned_alloc(64, C_len * sizeof(float));

    // const int64_t sgemm_m1 = 80;
    // const int64_t sgemm_n1 = 32;
    // const int64_t sgemm_k1 = 128;
    // const int64_t sgemm_m3 = 2560;
    // const int64_t sgemm_k3 = 5120;
    // const int64_t temp_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp32_gemm_get_buffer_size(sgemm_m1, sgemm_n1);

    // const int64_t temp_size = sgemm_algo1_buffer_bytes(M, N, K);

    const int64_t temp_size = ppl::kernel::arm_server::neon::gemm_ndarray_calc_buffer_size(ppl::common::DATATYPE_FLOAT32, M, N, K);

    float* temp = (float*)aligned_alloc(64, temp_size);

    if (!A || !B || !C || !C_ref || !temp) {
        fprintf(stderr, "malloc failed.\n");
        return;
    }

    for (int64_t i = 0; i < A_len; i++) {
        A[i] = i & 0x3;
    }
    for (int64_t i = 0; i < B_len; i++) {
        B[i] = i & 0x7;
    }
    for (int64_t i = 0; i < temp_size / sizeof(float); i++) {
        temp[i] = 0;
    }
    for (int64_t i = 0; i < C_len; i++) {
        C[i] = 0;
        C_ref[i] = 0;
    }

    if (diff) {
        sgemm_ref(A, B, M, N, K, K, N, N, C_ref);
    }

    // ppl::kernel::arm_server::neon::gemm_fp32(A, B, nullptr, C, temp, M, N, K, K, N, 0, N, 1, 0, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
    // sgemm_algo1(A, B, K, N, M, N, K, N, temp, C);
    ppl::kernel::arm_server::neon::gemm_ndarray(A, B, nullptr, ppl::common::DATATYPE_FLOAT32, M, N, K, K, N, 0, 0, 0, 1.0f, 0, N, ppl::kernel::arm_server::neon::gemm_C_type::EMPTY, C);

    if (diff) {
        int64_t err_cnt = 0;
        const int64_t err_verbose_num = 100;
        for (int64_t i = 0; i < C_len; i++) {
            if (C[i] != C_ref[i]) {
                err_cnt++;
                if (err_cnt <= err_verbose_num) {
                    fprintf(stderr, "error at %d: %f vs %f\n", i, C[i], C_ref[i]);
                }
            }
        }

        if (err_cnt <= 0) {
            fprintf(stderr, "diff pass\n");
        } else {
            fprintf(stderr, "diff failed, err_cnt = %d\n", err_cnt);
        }
    }

    const int64_t loops = 1;
    auto start = std::chrono::system_clock::now();
    for (int64_t i = 0; i < loops; i++) {
        // sgemm_algo1(A, B, K, N, M, N, K, N, temp, C);
        // ppl::kernel::arm_server::neon::gemm_fp32(A, B, nullptr, C, temp, M, N, K, K, N, 0, N, 1, 0, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
        ppl::kernel::arm_server::neon::gemm_ndarray(A, B, nullptr, ppl::common::DATATYPE_FLOAT32, M, N, K, K, N, 0, 0, 0, 1.0f, 0, N, ppl::kernel::arm_server::neon::gemm_C_type::EMPTY, C);
    }
    auto end = std::chrono::system_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time = time_diff.count() / 1000.f / loops;

    const double gops = M * N * K * 2 / 1e9;
    const double speed = gops / (time / 1000);

    fprintf(stderr, "time = %f ms, speed = %f gops/s\n", time, speed);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(temp);
}

int main(int argc, char* argv[]) {
    test_sgemm(1024, 1024, 1024, true);
    return 0;
}