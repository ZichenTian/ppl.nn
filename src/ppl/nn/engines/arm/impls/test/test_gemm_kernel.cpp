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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/gemm/neon/kernel/fp32/sgemm_ndarray_kernel.h"

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
                sum += A[k * lda + m] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

void test_sgemm_kernel(
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

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "malloc failed.\n");
        return;
    }

    for (int64_t i = 0; i < A_len; i++) {
        A[i] = i & 0x3;
    }
    for (int64_t i = 0; i < B_len; i++) {
        B[i] = i & 0x7;
    }
    for (int64_t i = 0; i < C_len; i++) {
        C[i] = 0;
        C_ref[i] = 0;
    }

    if (diff) {
        sgemm_ref(A, B, M, N, K, M, N, N, C_ref);
    }

    auto gemm_kernel_func = ppl::kernel::arm_server::neon::sgemm_ndarray_kernel_tn_max8x12_func_table[1][7][2];
    gemm_kernel_func(A, B, K, M, N, N, C);

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

    const int64_t loops = 1000000;
    auto start = std::chrono::system_clock::now();
    for (int64_t i = 0; i < loops; i++) {
        gemm_kernel_func(A, B, K, M, N, N, C);
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
}

int main(int argc, char* argv[]) {
    test_sgemm_kernel(8, 12, 256, true);
    return 0;
}
