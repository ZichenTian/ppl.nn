// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

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
    float* C)
{
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

void hgemm_ref(
    const __fp16* A,
    const __fp16* B,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    __fp16* C)
{
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            __fp16 sum = 0;
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
    const bool diff = true)
{
    float* A     = nullptr;
    float* B     = nullptr;
    float* C     = nullptr;
    float* C_ref = nullptr;

    const int64_t A_len = M * K;
    const int64_t B_len = K * N;
    const int64_t C_len = M * N;

    A     = (float*)aligned_alloc(64, A_len * sizeof(float));
    B     = (float*)aligned_alloc(64, B_len * sizeof(float));
    C     = (float*)aligned_alloc(64, C_len * sizeof(float));
    C_ref = (float*)aligned_alloc(64, C_len * sizeof(float));

    // const int64_t sgemm_m1 = 80;
    // const int64_t sgemm_n1 = 32;
    // const int64_t sgemm_k1 = 128;
    // const int64_t sgemm_m3 = 2560;
    // const int64_t sgemm_k3 = 5120;
    // const int64_t temp_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp32_gemm_get_buffer_size(sgemm_m1, sgemm_n1);

    // const int64_t temp_size = sgemm_algo1_buffer_bytes(M, N, K);

    const int64_t temp_size = 1;

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
        C[i]     = 0;
        C_ref[i] = 0;
    }

    if (diff) {
        sgemm_ref(A, B, M, N, K, K, N, N, C_ref);
    }

    auto start = std::chrono::system_clock::now();

    // ppl::kernel::arm_server::neon::gemm_fp32(A, B, nullptr, C, temp, M, N, K, K, N, 0, N, 1, 0, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
    // sgemm_algo1(A, B, K, N, M, N, K, N, temp, C);
    ppl::kernel::arm_server::neon::gemm_ndarray(A, B, nullptr, ppl::common::DATATYPE_FLOAT32, M, N, K, K, N, 0, 0, 0, 1.0f, 0, N, ppl::kernel::arm_server::neon::gemm_C_type::EMPTY, C);

    auto end       = std::chrono::system_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time    = time_diff.count() / 1e6;

    if (diff) {
        int64_t err_cnt               = 0;
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

    const int64_t loops = std::max(int32_t(1.0f / time), 1);
    start               = std::chrono::system_clock::now();
    for (int64_t i = 0; i < loops; i++) {
        // sgemm_algo1(A, B, K, N, M, N, K, N, temp, C);
        // ppl::kernel::arm_server::neon::gemm_fp32(A, B, nullptr, C, temp, M, N, K, K, N, 0, N, 1, 0, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
        ppl::kernel::arm_server::neon::gemm_ndarray(A, B, nullptr, ppl::common::DATATYPE_FLOAT32, M, N, K, K, N, 0, 0, 0, 1.0f, 0, N, ppl::kernel::arm_server::neon::gemm_C_type::EMPTY, C);
    }
    end       = std::chrono::system_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    time      = time_diff.count() / 1000.f / loops;

    const double gops  = M * N * K * 2 / 1e9;
    const double speed = gops / (time / 1000);
    const double peak_gops = 41.6 * PPL_OMP_MAX_THREADS();
    const double ratio = speed / peak_gops;

    fprintf(stderr, "time = %f ms, speed = %f gops/s, ratio = %f\n", time, speed, ratio);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(temp);
}

void test_hgemm(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const bool diff = true)
{
    __fp16* A     = nullptr;
    __fp16* B     = nullptr;
    __fp16* C     = nullptr;
    __fp16* C_ref = nullptr;

    const int64_t A_len = M * K;
    const int64_t B_len = K * N;
    const int64_t C_len = M * N;

    A     = (__fp16*)aligned_alloc(64, A_len * sizeof(__fp16));
    B     = (__fp16*)aligned_alloc(64, B_len * sizeof(__fp16));
    C     = (__fp16*)aligned_alloc(64, C_len * sizeof(__fp16));
    C_ref = (__fp16*)aligned_alloc(64, C_len * sizeof(__fp16));

    // const int64_t sgemm_m1 = 80;
    // const int64_t sgemm_n1 = 32;
    // const int64_t sgemm_k1 = 128;
    // const int64_t sgemm_m3 = 2560;
    // const int64_t sgemm_k3 = 5120;
    // const int64_t temp_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp32_gemm_get_buffer_size(sgemm_m1, sgemm_n1);

    // const int64_t temp_size = sgemm_algo1_buffer_bytes(M, N, K);

    const int64_t temp_size = 1;

    __fp16* temp = (__fp16*)aligned_alloc(64, temp_size);

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
    for (int64_t i = 0; i < temp_size / sizeof(__fp16); i++) {
        temp[i] = 0;
    }
    for (int64_t i = 0; i < C_len; i++) {
        C[i]     = 0;
        C_ref[i] = 0;
    }

    if (diff) {
        hgemm_ref(A, B, M, N, K, K, N, N, C_ref);
    }

    auto start = std::chrono::system_clock::now();

    // ppl::kernel::arm_server::neon::gemm_fp32(A, B, nullptr, C, temp, M, N, K, K, N, 0, N, 1, 0, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
    // sgemm_algo1(A, B, K, N, M, N, K, N, temp, C);
    ppl::kernel::arm_server::neon::gemm_ndarray(A, B, nullptr, ppl::common::DATATYPE_FLOAT16, M, N, K, K, N, 0, 0, 0, 1.0f, 0, N, ppl::kernel::arm_server::neon::gemm_C_type::EMPTY, C);

    auto end       = std::chrono::system_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time    = time_diff.count() / 1e6;

    if (diff) {
        int64_t err_cnt               = 0;
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

    const int64_t loops = std::max(int32_t(1.0f / time), 1);
    start               = std::chrono::system_clock::now();
    for (int64_t i = 0; i < loops; i++) {
        // sgemm_algo1(A, B, K, N, M, N, K, N, temp, C);
        // ppl::kernel::arm_server::neon::gemm_fp32(A, B, nullptr, C, temp, M, N, K, K, N, 0, N, 1, 0, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
        ppl::kernel::arm_server::neon::gemm_ndarray(A, B, nullptr, ppl::common::DATATYPE_FLOAT16, M, N, K, K, N, 0, 0, 0, 1.0f, 0, N, ppl::kernel::arm_server::neon::gemm_C_type::EMPTY, C);
    }
    end       = std::chrono::system_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    time      = time_diff.count() / 1000.f / loops;

    const double gops  = M * N * K * 2 / 1e9;
    const double speed = gops / (time / 1000);
    const double peak_gops = 83.2 * PPL_OMP_MAX_THREADS();
    const double ratio = speed / peak_gops;

    fprintf(stderr, "time = %f ms, speed = %f gops/s, ratio = %f\n", time, speed, ratio);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(temp);
}

int main(int argc, char* argv[])
{
    const bool diff = false;
    test_sgemm(1024, 1152, 1024, diff);
    test_sgemm(1024, 1024, 1024, diff);
    test_sgemm(255, 127, 33, diff);
    test_sgemm(5, 257, 193, diff);
    test_hgemm(1024, 1152, 1024, diff);
    test_hgemm(1024, 1024, 1024, diff);
    test_hgemm(255, 127, 33, diff);
    test_hgemm(5, 257, 193, diff);
    return 0;
}