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

#include <arm_neon.h>
#include <string.h>

#include "ppl/kernel/arm_server/gemm/neon/gemm.h"
#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

uint64_t gemm_ndarray_common_outer_calc_buffer_elemsize(
    const int64_t M,
    const int64_t N,
    const int64_t K)
{
    return 0;
}

template <typename eT>
ppl::common::RetCode gemm_ndarray_common_outer(
    const eT* A,
    const eT* B,
    const eT* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    eT* Y)
{
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0;
            for (int64_t k = 0; k < K; k++) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            Y[m * ldy + n] = sum;
        }
    }

    return ppl::common::RC_SUCCESS;
}

template ppl::common::RetCode gemm_ndarray_common_outer<float>(
    const float* A,
    const float* B,
    const float* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    float* Y);

}}}} // namespace ppl::kernel::arm_server::neon
