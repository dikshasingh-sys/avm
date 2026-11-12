/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <immintrin.h>
#include <stdio.h>
#include "avm/avm_integer.h"
#include "av2/common/common.h"

int64_t av2_highbd_block_error_avx2(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff,
                                    intptr_t block_size, int64_t *ssz,
                                    int bps) {
  intptr_t i;
  const int shift = 2 * (bps - 8);
  const int rounding = shift > 0 ? 1 << (shift - 1) : 0;
  __m256i acc_err = _mm256_setzero_si256();
  __m256i acc_ssz = _mm256_setzero_si256();

  for (i = 0; i < block_size; i += 16) {
    const __m256i mm256_coeff =
        _mm256_loadu_si256((const __m256i *)(coeff + i));
    const __m256i mm256_coeff2 =
        _mm256_loadu_si256((const __m256i *)(coeff + i + 8));
    const __m256i mm256_dqcoeff =
        _mm256_loadu_si256((const __m256i *)(dqcoeff + i));
    const __m256i mm256_dqcoeff2 =
        _mm256_loadu_si256((const __m256i *)(dqcoeff + i + 8));

    const __m256i diff1 = _mm256_sub_epi32(mm256_coeff, mm256_dqcoeff);
    const __m256i diff2 = _mm256_sub_epi32(mm256_coeff2, mm256_dqcoeff2);
    const __m256i diff1h = _mm256_srli_epi64(diff1, 32);
    const __m256i diff2h = _mm256_srli_epi64(diff2, 32);
    __m256i res = _mm256_mul_epi32(diff1, diff1);
    __m256i res1 = _mm256_mul_epi32(diff1h, diff1h);
    __m256i res2 = _mm256_mul_epi32(diff2, diff2);
    __m256i res3 = _mm256_mul_epi32(diff2h, diff2h);
    const __m256i res_diff = _mm256_add_epi64(_mm256_add_epi64(res, res1),
                                              _mm256_add_epi64(res2, res3));
    acc_err = _mm256_add_epi64(acc_err, res_diff);

    const __m256i mm256_coeffh = _mm256_srli_epi64(mm256_coeff, 32);
    const __m256i mm256_coeffh2 = _mm256_srli_epi64(mm256_coeff2, 32);
    res = _mm256_mul_epi32(mm256_coeff, mm256_coeff);
    res1 = _mm256_mul_epi32(mm256_coeffh, mm256_coeffh);
    res2 = _mm256_mul_epi32(mm256_coeff2, mm256_coeff2);
    res3 = _mm256_mul_epi32(mm256_coeffh2, mm256_coeffh2);
    const __m256i res_sqcoeff = _mm256_add_epi64(_mm256_add_epi64(res, res1),
                                                 _mm256_add_epi64(res2, res3));
    acc_ssz = _mm256_add_epi64(acc_ssz, res_sqcoeff);
  }
  const __m128i err128 = _mm_add_epi64(_mm256_castsi256_si128(acc_err),
                                       _mm256_extracti128_si256(acc_err, 1));
  const __m128i ssz128 = _mm_add_epi64(_mm256_castsi256_si128(acc_ssz),
                                       _mm256_extracti128_si256(acc_ssz, 1));
  int64_t error =
      _mm_cvtsi128_si64(err128) + _mm_cvtsi128_si64(_mm_srli_si128(err128, 8));
  int64_t sqcoeff =
      _mm_cvtsi128_si64(ssz128) + _mm_cvtsi128_si64(_mm_srli_si128(ssz128, 8));
  assert(error >= 0 && sqcoeff >= 0);
  error = (error + rounding) >> shift;
  sqcoeff = (sqcoeff + rounding) >> shift;

  *ssz = sqcoeff;
  return error;
}
