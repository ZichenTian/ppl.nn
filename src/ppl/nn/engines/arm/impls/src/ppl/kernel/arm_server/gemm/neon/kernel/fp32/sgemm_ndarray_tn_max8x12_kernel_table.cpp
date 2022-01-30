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

#include "ppl/kernel/arm_server/gemm/neon/kernel/fp32/sgemm_ndarray_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <int32_t init_t, int32_t m_block, int32_t n_block>
void sgemm_ndarray_tn_max8x12_kernel_func(
    const float* A, 
    const float* B, 
    const int32_t K, 
    const int32_t lda, 
    const int32_t ldb, 
    const int32_t ldc, 
    float* C);

#define INIT_T()    0   // init C as 0
    #define M_BLOCK()   1
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   2
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   3
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   4
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   5
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   6
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   7
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   8
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
#undef  INIT_T
#define INIT_T()    1   // init C by load
    #define M_BLOCK()   1
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   2
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   3
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   4
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   5
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   6
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   7
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
    #define M_BLOCK()   8
        #define N_BLOCK()   1
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   2
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
        #define N_BLOCK()   3
            #include "sgemm_ndarray_tn_max8x12_kernel.inc"
        #undef  N_BLOCK
    #undef  M_BLOCK
#undef  INIT_T

const sgemm_ndarray_kernel_func_t sgemm_ndarray_kernel_tn_max8x12_func_table[2][8][3] = {
    {
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 1, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 1, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 1, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 2, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 2, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 2, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 3, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 3, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 3, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 4, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 4, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 4, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 5, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 5, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 5, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 6, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 6, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 6, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 7, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 7, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 7, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<0, 8, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 8, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<0, 8, 3>, 
        }, 
    }, 
    {
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 1, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 1, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 1, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 2, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 2, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 2, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 3, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 3, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 3, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 4, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 4, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 4, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 5, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 5, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 5, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 6, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 6, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 6, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 7, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 7, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 7, 3>, 
        }, 
        {
            sgemm_ndarray_tn_max8x12_kernel_func<1, 8, 1>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 8, 2>, 
            sgemm_ndarray_tn_max8x12_kernel_func<1, 8, 3>, 
        }, 
    }, 
};

}}}}
