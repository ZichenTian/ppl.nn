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
#include "ppl/kernel/arm_server/common/simd_tools.h"
#include "ppl/kernel/arm_server/gemm/neon/kernel/fp32/sgemm_ndarray_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

// outer blk for multi-thread
#define M_OUTER_BLK() 256
#define N_OUTER_BLK() 192

#define K_INNER_BLK() 128
#define N_INNER_BLK() 96

#define M_KERNEL() 8
#define N_KERNEL() 12

inline uint64_t temp_buffer_a_elemsize(const int64_t K)
{
    return M_OUTER_BLK() * round_up(K, (int64_t)K_INNER_BLK());
}

inline uint64_t temp_buffer_b_elemsize(const int64_t K)
{
    return K_INNER_BLK() * N_INNER_BLK();
}

inline uint64_t temp_buffer_dst_elemsize(void)
{
    return M_OUTER_BLK() * N_OUTER_BLK();
}

inline uint64_t temp_buffer_per_thread(const int64_t K)
{
    return temp_buffer_a_elemsize(K) + temp_buffer_b_elemsize(K) + temp_buffer_dst_elemsize() + 128;
}

inline void* align_ptr(const void* ptr, uint64_t align)
{
    return (void*)(((uint64_t)ptr + align - 1) / align * align);
}

uint64_t gemm_ndarray_common_outer_calc_buffer_elemsize(
    const int64_t M,
    const int64_t N,
    const int64_t K)
{
    return temp_buffer_per_thread(K) * PPL_OMP_MAX_THREADS();
}

inline void transpose_4x4_32bit(
    float32x4_t& v0,
    float32x4_t& v1,
    float32x4_t& v2,
    float32x4_t& v3)
{
    const float32x4_t v4 = vtrn1q_f32(v0, v1);
    const float32x4_t v5 = vtrn2q_f32(v0, v1);
    const float32x4_t v6 = vtrn1q_f32(v2, v3);
    const float32x4_t v7 = vtrn2q_f32(v2, v3);

    v0 = (float32x4_t)vtrn1q_f64((float64x2_t)v4, (float64x2_t)v6);
    v1 = (float32x4_t)vtrn1q_f64((float64x2_t)v5, (float64x2_t)v7);
    v2 = (float32x4_t)vtrn2q_f64((float64x2_t)v4, (float64x2_t)v6);
    v3 = (float32x4_t)vtrn2q_f64((float64x2_t)v5, (float64x2_t)v7);
}

inline void prefetch_l1(const void* addr)
{
    asm volatile(
        "prfm   pldl1keep,  [%[addr],      #0]         \n"
        :
        : [addr] "r"(addr)
        :);
}

template <typename eT>
inline void gemm_ndarray_common_outer_pack_at(
    const eT* A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const int64_t transA,
    eT* dst);

template <>
inline void gemm_ndarray_common_outer_pack_at<float>(
    const float* A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const int64_t transA,
    float* dst)
{
    const int64_t simd_w = 4;
    const int64_t ld_dst = M_KERNEL();

    if (!transA) {
        for (int64_t k_base = 0; k_base < K; k_base += K_INNER_BLK()) {
            const int64_t k_eff = min(K - k_base, (int64_t)K_INNER_BLK());
            for (int64_t m_base = 0; m_base < M; m_base += M_KERNEL()) {
                const int64_t m_eff = min(M - m_base, (int64_t)M_KERNEL());
                const float* p_src  = A + m_base * lda + k_base;
                float* p_dst        = dst + k_base * M_OUTER_BLK() + m_base * K_INNER_BLK();

                int64_t m = 0;
                for (; m + simd_w < m_eff; m += simd_w) {
                    const float* p_src_0 = p_src + (m + 0) * lda;
                    const float* p_src_1 = p_src + (m + 1) * lda;
                    const float* p_src_2 = p_src + (m + 2) * lda;
                    const float* p_src_3 = p_src + (m + 3) * lda;

                    const float* p_src_4 = p_src + (m + 4) * lda;
                    const float* p_src_5 = p_src + (m + 5) * lda;
                    const float* p_src_6 = p_src + (m + 6) * lda;
                    const float* p_src_7 = p_src + (m + 7) * lda;

                    int64_t k = 0;
                    for (; k + simd_w <= k_eff; k += simd_w) {
                        prefetch_l1(p_src_4 + k);
                        prefetch_l1(p_src_5 + k);
                        prefetch_l1(p_src_6 + k);
                        prefetch_l1(p_src_7 + k);

                        float32x4_t v0 = vld1q_f32(p_src_0 + k);
                        float32x4_t v1 = vld1q_f32(p_src_1 + k);
                        float32x4_t v2 = vld1q_f32(p_src_2 + k);
                        float32x4_t v3 = vld1q_f32(p_src_3 + k);

                        transpose_4x4_32bit(v0, v1, v2, v3);

                        vst1q_f32(p_dst + (k + 0) * ld_dst + m, v0);
                        vst1q_f32(p_dst + (k + 1) * ld_dst + m, v1);
                        vst1q_f32(p_dst + (k + 2) * ld_dst + m, v2);
                        vst1q_f32(p_dst + (k + 3) * ld_dst + m, v3);
                    }
                    for (; k < k_eff; k++) {
                        p_dst[k * ld_dst + m + 0] = p_src_0[k];
                        p_dst[k * ld_dst + m + 1] = p_src_1[k];
                        p_dst[k * ld_dst + m + 2] = p_src_2[k];
                        p_dst[k * ld_dst + m + 3] = p_src_3[k];
                    }
                }
                for (; m < m_eff; m++) {
                    for (int64_t k = 0; k < k_eff; k++) {
                        p_dst[k * ld_dst + m] = p_src[m * lda + k];
                    }
                }
            }
        }
    }
}

template <typename eT>
inline void gemm_ndarray_common_outer_pack_bn(
    const eT* B,
    const int64_t K,
    const int64_t N,
    const int64_t ldb,
    const int64_t transB,
    eT* dst);

template <>
inline void gemm_ndarray_common_outer_pack_bn<float>(
    const float* B,
    const int64_t K,
    const int64_t N,
    const int64_t ldb,
    const int64_t transB,
    float* dst)
{
    const int64_t simd_w = 4;
    const int64_t ld_dst = N_KERNEL();

    if (!transB) {
        for (int64_t n_base = 0; n_base < N; n_base += N_KERNEL()) {
            const int64_t n_eff = min(N - n_base, (int64_t)N_KERNEL());
            const float* p_src  = B + n_base;
            float* p_dst        = dst + n_base * K_INNER_BLK();

            const int64_t prefetch_line = 1;
            const float* p_src_next     = p_src + prefetch_line * ldb;

            if (n_eff == N_KERNEL()) {
                for (int64_t k = 0; k < K; k++) {
                    prefetch_l1(p_src_next + k * ld_dst);
                    vst1q_f32(p_dst + k * ld_dst + simd_w * 0, vld1q_f32(p_src + k * ldb + simd_w * 0));
                    vst1q_f32(p_dst + k * ld_dst + simd_w * 1, vld1q_f32(p_src + k * ldb + simd_w * 1));
                    vst1q_f32(p_dst + k * ld_dst + simd_w * 2, vld1q_f32(p_src + k * ldb + simd_w * 2));
                }
            } else {
                for (int64_t k = 0; k < K; k++) {
                    prefetch_l1(p_src_next + k * ld_dst);
                    int64_t n = 0;
                    for (; n + simd_w <= n_eff; n += simd_w) {
                        vst1q_f32(p_dst + k * ld_dst + n, vld1q_f32(p_src + k * ldb + n));
                    }
                    for (; n < n_eff; n++) {
                        p_dst[k * ld_dst + n] = p_src[k * ldb + n];
                    }
                }
            }
        }
    }
}

template <typename eT>
inline void gemm_ndarray_common_outer_store_dst(
    const eT* src,
    const eT* C,
    const int64_t M,
    const int64_t N,
    const int64_t m_offset,
    const int64_t n_offset,
    const float alpha,
    const float beta,
    const int64_t ldc,
    const int64_t ld_dst,
    const gemm_C_type_t c_type,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w = sizeof(vecType) / sizeof(eT);
    const int64_t ld_src = N_INNER_BLK();

    const vecType v_alpha = vdup_n<eT, eN>((eT)alpha);

    if (c_type == gemm_C_type::EMPTY || c_type == gemm_C_type::SCALAR) {
        const eT bias        = (c_type == gemm_C_type::SCALAR ? C[0] : 0) * beta;
        const vecType v_bias = vdup_n<eT, eN>(bias);
        for (int64_t m = 0; m < M; m++) {
            int64_t n = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_bias);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + bias;
            }
        }
    } else if (c_type == gemm_C_type::VECTOR_H) {
        for (int64_t m = 0; m < M; m++) {
            const eT bias        = C[m + m_offset] * beta;
            const vecType v_bias = vdup_n<eT, eN>(bias);
            int64_t n            = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_bias);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + bias;
            }
        }
    } else if (c_type == gemm_C_type::VECTOR_W) {
        const vecType v_beta = vdup_n<eT, eN>((eT)beta);
        for (int64_t m = 0; m < M; m++) {
            int64_t n = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vecType v_c   = vld<eT, eN>(C + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_c * v_beta);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + C[n] * beta;
            }
        }
    } else if (c_type == gemm_C_type::MATRIX) {
        const vecType v_beta = vdup_n<eT, eN>((eT)beta);
        for (int64_t m = 0; m < M; m++) {
            int64_t n = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vecType v_c   = vld<eT, eN>(C + m * ldc + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_c * v_beta);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + C[m * ldc + n] * beta;
            }
        }
    }
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
    void* temp,
    eT* Y)
{
    const int64_t simd_w = 4;
    const int64_t num_threads = PPL_OMP_MAX_THREADS();
    std::vector<const float*> last_pack_a_ptr(num_threads, nullptr);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t m = 0; m < M; m += M_OUTER_BLK()) {
        for (int64_t n = 0; n < N; n += N_OUTER_BLK()) {
            const int64_t thread_id = PPL_OMP_THREAD_ID();
            eT* temp_buffer     = (eT*)align_ptr((eT*)temp + temp_buffer_per_thread(K) * thread_id, 64);
            eT* temp_buffer_a   = temp_buffer;
            eT* temp_buffer_b   = temp_buffer_a + temp_buffer_a_elemsize(K);
            eT* temp_buffer_dst = temp_buffer_b + temp_buffer_b_elemsize(K);

            const int64_t m_eff = min(M - m, (int64_t)M_OUTER_BLK());
            const int64_t n_eff = min(N - n, (int64_t)N_OUTER_BLK());
            const eT* p_src_a   = transA ? A + m : A + m * lda;
            if (last_pack_a_ptr[thread_id] != p_src_a) {
                gemm_ndarray_common_outer_pack_at<eT>(p_src_a, m_eff, K, lda, transA, temp_buffer_a);
                last_pack_a_ptr[thread_id] = p_src_a;
            }

            for (int64_t nn = 0; nn < n_eff; nn += N_INNER_BLK()) {
                const int64_t nn_eff = min(n_eff - nn, (int64_t)N_INNER_BLK());

                for (int64_t kk = 0; kk < K; kk += K_INNER_BLK()) {
                    const int64_t kk_eff = min(K - kk, (int64_t)K_INNER_BLK());
                    const eT* p_src_b    = transB ? B + (n + nn) * ldb + kk : B + kk * ldb + n + nn;
                    gemm_ndarray_common_outer_pack_bn<eT>(p_src_b, kk_eff, nn_eff, ldb, transB, temp_buffer_b);

                    const int64_t init_t = kk == 0 ? 0 : 1;

                    for (int64_t m_kernel = 0; m_kernel < m_eff; m_kernel += M_KERNEL()) {
                        for (int64_t n_kernel = 0; n_kernel < nn_eff; n_kernel += N_KERNEL()) {
                            const int64_t m_kernel_len = min(m_eff - m_kernel, (int64_t)M_KERNEL());
                            const int64_t n_kernel_len = min(nn_eff - n_kernel, (int64_t)N_KERNEL());
                            const int64_t n_kernel_blk = div_up(n_kernel_len, simd_w);

                            const int64_t prefetch_a = 1;
                            const int64_t prefetch_b = 1;

                            const int64_t m_kernel_idx = m_kernel_len - 1;
                            const int64_t n_kernel_idx = n_kernel_blk - 1;
                            if (std::is_same<eT, float>::value) {
                                auto gemm_kernel_func = sgemm_ndarray_kernel_tn_max8x12_func_table[prefetch_a][prefetch_b][init_t][m_kernel_idx][n_kernel_idx];
                                gemm_kernel_func(
                                    temp_buffer_a + kk * M_OUTER_BLK() + m_kernel * K_INNER_BLK(),
                                    temp_buffer_b + n_kernel * K_INNER_BLK(),
                                    kk_eff,
                                    M_KERNEL(),
                                    N_KERNEL(),
                                    N_INNER_BLK(),
                                    temp_buffer_dst + m_kernel * N_INNER_BLK() + n_kernel);
                            }
                        }
                    }
                }

                gemm_ndarray_common_outer_store_dst<eT>(temp_buffer_dst, C, m_eff, nn_eff, m, nn, alpha, beta, ldc, ldy, c_type, Y + m * ldy + n + nn);
            }
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
    void* temp,
    float* Y);

}}}} // namespace ppl::kernel::arm_server::neon
