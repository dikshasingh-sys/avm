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

#include "config/av2_rtcd.h"

#include "avm/avm_integer.h"
#include "avm_dsp/blend.h"
#include "avm_dsp/x86/synonyms.h"
#include "avm_dsp/x86/synonyms_avx2.h"
#include "av2/common/blockd.h"

static INLINE __m256i calc_mask_d16_avx2(const __m256i *data_src0,
                                         const __m256i *data_src1,
                                         const __m256i *round_const,
                                         const __m256i *mask_base_16,
                                         const __m256i *clip_diff, int round) {
  const __m256i diffa = _mm256_subs_epu16(*data_src0, *data_src1);
  const __m256i diffb = _mm256_subs_epu16(*data_src1, *data_src0);
  const __m256i diff = _mm256_max_epu16(diffa, diffb);
  const __m256i diff_round =
      _mm256_srli_epi16(_mm256_adds_epu16(diff, *round_const), round);
  const __m256i diff_factor = _mm256_srli_epi16(diff_round, DIFF_FACTOR_LOG2);
  const __m256i diff_mask = _mm256_adds_epi16(diff_factor, *mask_base_16);
  const __m256i diff_clamp = _mm256_min_epi16(diff_mask, *clip_diff);
  return diff_clamp;
}

static INLINE __m256i calc_mask_d16_inv_avx2(const __m256i *data_src0,
                                             const __m256i *data_src1,
                                             const __m256i *round_const,
                                             const __m256i *mask_base_16,
                                             const __m256i *clip_diff,
                                             int round) {
  const __m256i diffa = _mm256_subs_epu16(*data_src0, *data_src1);
  const __m256i diffb = _mm256_subs_epu16(*data_src1, *data_src0);
  const __m256i diff = _mm256_max_epu16(diffa, diffb);
  const __m256i diff_round =
      _mm256_srli_epi16(_mm256_adds_epu16(diff, *round_const), round);
  const __m256i diff_factor = _mm256_srli_epi16(diff_round, DIFF_FACTOR_LOG2);
  const __m256i diff_mask = _mm256_adds_epi16(diff_factor, *mask_base_16);
  const __m256i diff_clamp = _mm256_min_epi16(diff_mask, *clip_diff);
  const __m256i diff_const_16 = _mm256_sub_epi16(*clip_diff, diff_clamp);
  return diff_const_16;
}

static INLINE void build_compound_diffwtd_mask_d16_avx2(
    uint8_t *mask, const CONV_BUF_TYPE *src0, int src0_stride,
    const CONV_BUF_TYPE *src1, int src1_stride, int h, int w, int shift) {
  const int mask_base = 38;
  const __m256i _r = _mm256_set1_epi16((1 << shift) >> 1);
  const __m256i y38 = _mm256_set1_epi16(mask_base);
  const __m256i y64 = _mm256_set1_epi16(AVM_BLEND_A64_MAX_ALPHA);
  int i = 0;
  if (w == 4) {
    do {
      const __m128i s0A = xx_loadl_64(src0);
      const __m128i s0B = xx_loadl_64(src0 + src0_stride);
      const __m128i s0C = xx_loadl_64(src0 + src0_stride * 2);
      const __m128i s0D = xx_loadl_64(src0 + src0_stride * 3);
      const __m128i s1A = xx_loadl_64(src1);
      const __m128i s1B = xx_loadl_64(src1 + src1_stride);
      const __m128i s1C = xx_loadl_64(src1 + src1_stride * 2);
      const __m128i s1D = xx_loadl_64(src1 + src1_stride * 3);
      const __m256i s0 = yy_set_m128i(_mm_unpacklo_epi64(s0C, s0D),
                                      _mm_unpacklo_epi64(s0A, s0B));
      const __m256i s1 = yy_set_m128i(_mm_unpacklo_epi64(s1C, s1D),
                                      _mm_unpacklo_epi64(s1A, s1B));
      const __m256i m16 = calc_mask_d16_avx2(&s0, &s1, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16, _mm256_setzero_si256());
      xx_storeu_128(mask,
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(m8, 0xd8)));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 16;
      i += 4;
    } while (i < h);
  } else if (w == 8) {
    do {
      const __m256i s0AB = yy_loadu2_128(src0 + src0_stride, src0);
      const __m256i s0CD =
          yy_loadu2_128(src0 + src0_stride * 3, src0 + src0_stride * 2);
      const __m256i s1AB = yy_loadu2_128(src1 + src1_stride, src1);
      const __m256i s1CD =
          yy_loadu2_128(src1 + src1_stride * 3, src1 + src1_stride * 2);
      const __m256i m16AB =
          calc_mask_d16_avx2(&s0AB, &s1AB, &_r, &y38, &y64, shift);
      const __m256i m16CD =
          calc_mask_d16_avx2(&s0CD, &s1CD, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16AB, m16CD);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 32;
      i += 4;
    } while (i < h);
  } else if (w == 16) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + src0_stride);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + src1_stride);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 1;
      src1 += src1_stride << 1;
      mask += 32;
      i += 2;
    } while (i < h);
  } else if (w == 32) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 32;
      i += 1;
    } while (i < h);
  } else if (w == 64) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 64;
      i += 1;
    } while (i < h);
  } else if (w == 128) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s0E = yy_loadu_256(src0 + 64);
      const __m256i s0F = yy_loadu_256(src0 + 80);
      const __m256i s0G = yy_loadu_256(src0 + 96);
      const __m256i s0H = yy_loadu_256(src0 + 112);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i s1E = yy_loadu_256(src1 + 64);
      const __m256i s1F = yy_loadu_256(src1 + 80);
      const __m256i s1G = yy_loadu_256(src1 + 96);
      const __m256i s1H = yy_loadu_256(src1 + 112);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m16E =
          calc_mask_d16_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
      const __m256i m16F =
          calc_mask_d16_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
      const __m256i m16G =
          calc_mask_d16_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
      const __m256i m16H =
          calc_mask_d16_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
      const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
      yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 128;
      i += 1;
    } while (i < h);
  } else {
    assert(w == 256);
    do {
      const CONV_BUF_TYPE *src0_ptr = src0;
      const CONV_BUF_TYPE *src1_ptr = src1;
      for (int loop = 0; loop < 2; loop++) {
        const __m256i s0A = yy_loadu_256(src0_ptr);
        const __m256i s0B = yy_loadu_256(src0_ptr + 16);
        const __m256i s0C = yy_loadu_256(src0_ptr + 32);
        const __m256i s0D = yy_loadu_256(src0_ptr + 48);
        const __m256i s0E = yy_loadu_256(src0_ptr + 64);
        const __m256i s0F = yy_loadu_256(src0_ptr + 80);
        const __m256i s0G = yy_loadu_256(src0_ptr + 96);
        const __m256i s0H = yy_loadu_256(src0_ptr + 112);
        const __m256i s1A = yy_loadu_256(src1_ptr);
        const __m256i s1B = yy_loadu_256(src1_ptr + 16);
        const __m256i s1C = yy_loadu_256(src1_ptr + 32);
        const __m256i s1D = yy_loadu_256(src1_ptr + 48);
        const __m256i s1E = yy_loadu_256(src1_ptr + 64);
        const __m256i s1F = yy_loadu_256(src1_ptr + 80);
        const __m256i s1G = yy_loadu_256(src1_ptr + 96);
        const __m256i s1H = yy_loadu_256(src1_ptr + 112);
        const __m256i m16A =
            calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
        const __m256i m16B =
            calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
        const __m256i m16C =
            calc_mask_d16_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
        const __m256i m16D =
            calc_mask_d16_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
        const __m256i m16E =
            calc_mask_d16_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
        const __m256i m16F =
            calc_mask_d16_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
        const __m256i m16G =
            calc_mask_d16_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
        const __m256i m16H =
            calc_mask_d16_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
        const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
        const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
        const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
        const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
        yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
        yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
        yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
        yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
        src0_ptr += 128;
        src1_ptr += 128;
        mask += 128;
      }
      src0 += src0_stride;
      src1 += src1_stride;
      i += 1;
    } while (i < h);
  }
}

static INLINE void build_compound_diffwtd_mask_d16_inv_avx2(
    uint8_t *mask, const CONV_BUF_TYPE *src0, int src0_stride,
    const CONV_BUF_TYPE *src1, int src1_stride, int h, int w, int shift) {
  const int mask_base = 38;
  const __m256i _r = _mm256_set1_epi16((1 << shift) >> 1);
  const __m256i y38 = _mm256_set1_epi16(mask_base);
  const __m256i y64 = _mm256_set1_epi16(AVM_BLEND_A64_MAX_ALPHA);
  int i = 0;
  if (w == 4) {
    do {
      const __m128i s0A = xx_loadl_64(src0);
      const __m128i s0B = xx_loadl_64(src0 + src0_stride);
      const __m128i s0C = xx_loadl_64(src0 + src0_stride * 2);
      const __m128i s0D = xx_loadl_64(src0 + src0_stride * 3);
      const __m128i s1A = xx_loadl_64(src1);
      const __m128i s1B = xx_loadl_64(src1 + src1_stride);
      const __m128i s1C = xx_loadl_64(src1 + src1_stride * 2);
      const __m128i s1D = xx_loadl_64(src1 + src1_stride * 3);
      const __m256i s0 = yy_set_m128i(_mm_unpacklo_epi64(s0C, s0D),
                                      _mm_unpacklo_epi64(s0A, s0B));
      const __m256i s1 = yy_set_m128i(_mm_unpacklo_epi64(s1C, s1D),
                                      _mm_unpacklo_epi64(s1A, s1B));
      const __m256i m16 =
          calc_mask_d16_inv_avx2(&s0, &s1, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16, _mm256_setzero_si256());
      xx_storeu_128(mask,
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(m8, 0xd8)));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 16;
      i += 4;
    } while (i < h);
  } else if (w == 8) {
    do {
      const __m256i s0AB = yy_loadu2_128(src0 + src0_stride, src0);
      const __m256i s0CD =
          yy_loadu2_128(src0 + src0_stride * 3, src0 + src0_stride * 2);
      const __m256i s1AB = yy_loadu2_128(src1 + src1_stride, src1);
      const __m256i s1CD =
          yy_loadu2_128(src1 + src1_stride * 3, src1 + src1_stride * 2);
      const __m256i m16AB =
          calc_mask_d16_inv_avx2(&s0AB, &s1AB, &_r, &y38, &y64, shift);
      const __m256i m16CD =
          calc_mask_d16_inv_avx2(&s0CD, &s1CD, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16AB, m16CD);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 32;
      i += 4;
    } while (i < h);
  } else if (w == 16) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + src0_stride);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + src1_stride);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 1;
      src1 += src1_stride << 1;
      mask += 32;
      i += 2;
    } while (i < h);
  } else if (w == 32) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 32;
      i += 1;
    } while (i < h);
  } else if (w == 64) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_inv_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_inv_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 64;
      i += 1;
    } while (i < h);
  } else if (w == 128) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s0E = yy_loadu_256(src0 + 64);
      const __m256i s0F = yy_loadu_256(src0 + 80);
      const __m256i s0G = yy_loadu_256(src0 + 96);
      const __m256i s0H = yy_loadu_256(src0 + 112);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i s1E = yy_loadu_256(src1 + 64);
      const __m256i s1F = yy_loadu_256(src1 + 80);
      const __m256i s1G = yy_loadu_256(src1 + 96);
      const __m256i s1H = yy_loadu_256(src1 + 112);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_inv_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_inv_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m16E =
          calc_mask_d16_inv_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
      const __m256i m16F =
          calc_mask_d16_inv_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
      const __m256i m16G =
          calc_mask_d16_inv_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
      const __m256i m16H =
          calc_mask_d16_inv_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
      const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
      yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 128;
      i += 1;
    } while (i < h);
  } else {
    assert(w == 256);
    do {
      const CONV_BUF_TYPE *src0_ptr = src0;
      const CONV_BUF_TYPE *src1_ptr = src1;
      for (int loop = 0; loop < 2; loop++) {
        const __m256i s0A = yy_loadu_256(src0_ptr);
        const __m256i s0B = yy_loadu_256(src0_ptr + 16);
        const __m256i s0C = yy_loadu_256(src0_ptr + 32);
        const __m256i s0D = yy_loadu_256(src0_ptr + 48);
        const __m256i s0E = yy_loadu_256(src0_ptr + 64);
        const __m256i s0F = yy_loadu_256(src0_ptr + 80);
        const __m256i s0G = yy_loadu_256(src0_ptr + 96);
        const __m256i s0H = yy_loadu_256(src0_ptr + 112);
        const __m256i s1A = yy_loadu_256(src1_ptr);
        const __m256i s1B = yy_loadu_256(src1_ptr + 16);
        const __m256i s1C = yy_loadu_256(src1_ptr + 32);
        const __m256i s1D = yy_loadu_256(src1_ptr + 48);
        const __m256i s1E = yy_loadu_256(src1_ptr + 64);
        const __m256i s1F = yy_loadu_256(src1_ptr + 80);
        const __m256i s1G = yy_loadu_256(src1_ptr + 96);
        const __m256i s1H = yy_loadu_256(src1_ptr + 112);
        const __m256i m16A =
            calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
        const __m256i m16B =
            calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
        const __m256i m16C =
            calc_mask_d16_inv_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
        const __m256i m16D =
            calc_mask_d16_inv_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
        const __m256i m16E =
            calc_mask_d16_inv_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
        const __m256i m16F =
            calc_mask_d16_inv_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
        const __m256i m16G =
            calc_mask_d16_inv_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
        const __m256i m16H =
            calc_mask_d16_inv_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
        const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
        const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
        const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
        const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
        yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
        yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
        yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
        yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
        src0_ptr += 128;
        src1_ptr += 128;
        mask += 128;
      }
      src0 += src0_stride;
      src1 += src1_stride;
      i += 1;
    } while (i < h);
  }
}

void av2_build_compound_diffwtd_mask_d16_avx2(
    uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const CONV_BUF_TYPE *src0,
    int src0_stride, const CONV_BUF_TYPE *src1, int src1_stride, int h, int w,
    ConvolveParams *conv_params, int bd) {
  const int shift =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1 + (bd - 8);
  // When rounding constant is added, there is a possibility of overflow.
  // However that much precision is not required. Code should very well work for
  // other values of DIFF_FACTOR_LOG2 and AVM_BLEND_A64_MAX_ALPHA as well. But
  // there is a possibility of corner case bugs.
  assert(DIFF_FACTOR_LOG2 == 4);
  assert(AVM_BLEND_A64_MAX_ALPHA == 64);

  if (mask_type == DIFFWTD_38) {
    build_compound_diffwtd_mask_d16_avx2(mask, src0, src0_stride, src1,
                                         src1_stride, h, w, shift);
  } else {
    build_compound_diffwtd_mask_d16_inv_avx2(mask, src0, src0_stride, src1,
                                             src1_stride, h, w, shift);
  }
}

void av2_build_compound_diffwtd_mask_highbd_avx2(
    uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const uint16_t *ssrc0,
    int src0_stride, const uint16_t *ssrc1, int src1_stride, int h, int w,
    int bd) {
  if (w < 16) {
    av2_build_compound_diffwtd_mask_highbd_ssse3(
        mask, mask_type, ssrc0, src0_stride, ssrc1, src1_stride, h, w, bd);
  } else {
    assert(mask_type == DIFFWTD_38 || mask_type == DIFFWTD_38_INV);
    assert(bd >= 8);
    assert((w % 16) == 0);
    const __m256i y0 = _mm256_setzero_si256();
    const __m256i yAVM_BLEND_A64_MAX_ALPHA =
        _mm256_set1_epi16(AVM_BLEND_A64_MAX_ALPHA);
    const int mask_base = 38;
    const __m256i ymask_base = _mm256_set1_epi16(mask_base);
    if (bd == 8) {
      if (mask_type == DIFFWTD_38_INV) {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_srai_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), DIFF_FACTOR_LOG2);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAVM_BLEND_A64_MAX_ALPHA);
            m = _mm256_sub_epi16(yAVM_BLEND_A64_MAX_ALPHA, m);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      } else {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_srai_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), DIFF_FACTOR_LOG2);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAVM_BLEND_A64_MAX_ALPHA);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      }
    } else {
      const __m128i xshift = xx_set1_64_from_32i(bd - 8 + DIFF_FACTOR_LOG2);
      if (mask_type == DIFFWTD_38_INV) {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_sra_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), xshift);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAVM_BLEND_A64_MAX_ALPHA);
            m = _mm256_sub_epi16(yAVM_BLEND_A64_MAX_ALPHA, m);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      } else {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_sra_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), xshift);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAVM_BLEND_A64_MAX_ALPHA);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      }
    }
  }
}

// Doubled pixel indices (byte offsets) for a 16-pixel chunk of uint16_t.
DECLARE_ALIGNED(32, static const int8_t, pixel_byte_offsets[32]) = {
  0,  0,  2,  2,  4,  4,  6,  6,  8,  8,  10, 10, 12, 12, 14, 14,
  16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30
};

typedef enum {
  PAD_ROW_WIDE_16PX,           // Widths 17..32: two overlapping 16-px chunks.
  PAD_ROW_NARROW_ALIGN_LEFT,   // Widths 8..16: 16-px load from row start.
  PAD_ROW_NARROW_ALIGN_RIGHT,  // Widths 8..16: 16-px load from row end.
  PAD_ROW_BROADCAST_PIXEL,     // Entire row repeats one border pixel.
  PAD_ROW_GATHER_LANES,        // >7 px padding: four independent 8-px lanes.
  PAD_ROW_FALLBACK             // Out-of-line path (>7 px padding or <8, >32).
} PadRowMode;

typedef struct {
  __m256i shuffle_mask[2];
  int right_chunk_col;     // Start column of the right 16-px or 8-px chunk.
  int is_wide_chunk;       // 1 for 16-px chunks, 0 for 8-px chunks.
  int lane_src_offset[4];  // Source column offset per 8-px lane.
} PadRowConfig;

// Builds a pshufb mask clamping pixel indices to [min_col, max_col].
static INLINE __m256i build_clamp_shuffle_mask(__m256i byte_offsets,
                                               int min_col, int max_col) {
  byte_offsets =
      _mm256_max_epi8(byte_offsets, _mm256_set1_epi8((char)(2 * min_col)));
  byte_offsets =
      _mm256_min_epi8(byte_offsets, _mm256_set1_epi8((char)(2 * max_col)));
  return _mm256_add_epi8(byte_offsets, _mm256_set1_epi16(0x0100));
}

// Clamps an 8-px lane's source column so it overlaps valid reference columns.
static INLINE int clamp_lane_src_offset(int lane_dst_col, int valid_start_col,
                                        int valid_end_col) {
  return clamp(lane_dst_col, valid_start_col - 7, valid_end_col);
}

// Builds a 128-bit pshufb mask for an 8-px lane at lane_dst_col.
static INLINE __m128i build_lane_shuffle_mask(int lane_dst_col,
                                              int lane_src_col,
                                              int valid_start_col,
                                              int valid_end_col) {
  const int col_shift = clamp(lane_dst_col - lane_src_col, -8, 8);
  const int valid_lane_start = AVMMAX(valid_start_col - lane_src_col, 0);
  const int valid_lane_end = AVMMIN(valid_end_col - lane_src_col, 7);
  __m128i byte_offsets = _mm_load_si128((const __m128i *)pixel_byte_offsets);
  byte_offsets =
      _mm_add_epi8(byte_offsets, _mm_set1_epi8((char)(2 * col_shift)));
  byte_offsets =
      _mm_max_epi8(byte_offsets, _mm_set1_epi8((char)(2 * valid_lane_start)));
  byte_offsets =
      _mm_min_epi8(byte_offsets, _mm_set1_epi8((char)(2 * valid_lane_end)));
  return _mm_add_epi8(byte_offsets, _mm_set1_epi16(0x0100));
}

static AVM_FORCE_INLINE void load_padded_row(PadRowMode mode,
                                             const uint16_t *src_row,
                                             const PadRowConfig *cfg,
                                             __m256i *left_chunk,
                                             __m256i *right_chunk) {
  if (mode == PAD_ROW_WIDE_16PX) {
    *left_chunk =
        _mm256_shuffle_epi8(yy_loadu_256(src_row), cfg->shuffle_mask[0]);
    *right_chunk = _mm256_shuffle_epi8(
        yy_loadu_256(src_row + cfg->right_chunk_col), cfg->shuffle_mask[1]);
  } else if (mode == PAD_ROW_NARROW_ALIGN_LEFT) {
    *left_chunk = *right_chunk =
        _mm256_shuffle_epi8(yy_loadu_256(src_row), cfg->shuffle_mask[0]);
  } else if (mode == PAD_ROW_NARROW_ALIGN_RIGHT) {
    *left_chunk = *right_chunk = _mm256_shuffle_epi8(
        yy_loadu_256(src_row + cfg->right_chunk_col - 8), cfg->shuffle_mask[0]);
  } else if (mode == PAD_ROW_BROADCAST_PIXEL) {
    *left_chunk = *right_chunk = _mm256_set1_epi16(*src_row);
  } else {
    *left_chunk =
        _mm256_shuffle_epi8(yy_loadu2_128(src_row + cfg->lane_src_offset[1],
                                          src_row + cfg->lane_src_offset[0]),
                            cfg->shuffle_mask[0]);
    *right_chunk =
        _mm256_shuffle_epi8(yy_loadu2_128(src_row + cfg->lane_src_offset[3],
                                          src_row + cfg->lane_src_offset[2]),
                            cfg->shuffle_mask[1]);
  }
}

static AVM_FORCE_INLINE void store_padded_row(PadRowMode mode,
                                              uint16_t *dst_row,
                                              const PadRowConfig *cfg,
                                              __m256i left_chunk,
                                              __m256i right_chunk) {
  // Store the padded lane first so the valid lane overwrites shared columns.
  if (mode == PAD_ROW_NARROW_ALIGN_LEFT) {
    xx_storeu_128(dst_row + cfg->right_chunk_col,
                  _mm256_extracti128_si256(left_chunk, 1));
    xx_storeu_128(dst_row, _mm256_castsi256_si128(left_chunk));
  } else if (mode == PAD_ROW_NARROW_ALIGN_RIGHT) {
    xx_storeu_128(dst_row, _mm256_castsi256_si128(left_chunk));
    xx_storeu_128(dst_row + cfg->right_chunk_col,
                  _mm256_extracti128_si256(left_chunk, 1));
  } else if (mode == PAD_ROW_WIDE_16PX || cfg->is_wide_chunk) {
    yy_storeu_256(dst_row, left_chunk);
    yy_storeu_256(dst_row + cfg->right_chunk_col, right_chunk);
  } else {
    xx_storeu_128(dst_row, _mm256_castsi256_si128(left_chunk));
    xx_storeu_128(dst_row + cfg->right_chunk_col,
                  _mm256_castsi256_si128(right_chunk));
  }
}

static AVM_FORCE_INLINE void pad_block_rows(
    PadRowMode mode, const uint16_t *src_row, ptrdiff_t src_stride,
    uint16_t *dst_row, ptrdiff_t dst_stride, const PadRowConfig *cfg,
    int block_h, int cur_ref_row, int last_ref_row) {
  __m256i left_chunk, right_chunk;
  if (cur_ref_row >= last_ref_row || cur_ref_row + block_h <= 1) {
    // Entire block clamps to a single top or bottom boundary row.
    load_padded_row(mode, src_row, cfg, &left_chunk, &right_chunk);
    do {
      store_padded_row(mode, dst_row, cfg, left_chunk, right_chunk);
      dst_row += dst_stride;
    } while (--block_h);
    return;
  }
  do {
    load_padded_row(mode, src_row, cfg, &left_chunk, &right_chunk);
    store_padded_row(mode, dst_row, cfg, left_chunk, right_chunk);
    // Advance src_row only inside [0, last_ref_row].
    src_row +=
        (unsigned int)cur_ref_row < (unsigned int)last_ref_row ? src_stride : 0;
    ++cur_ref_row;
    dst_row += dst_stride;
  } while (--block_h);
}

#if defined(_MSC_VER)
#define PAD_MC_BORDER_NOINLINE __declspec(noinline)
#else
#define PAD_MC_BORDER_NOINLINE __attribute__((noinline))
#endif

// Handles >7 px horizontal padding or widths outside [8, 32].
static PAD_MC_BORDER_NOINLINE void pad_mc_border_fallback(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int x0,
    int y0, int b_w, int b_h, const ReferenceArea *ref_area) {
  if (b_w < 8 || b_w > 32) {
    refinemv_highbd_pad_mc_border_c(src, src_stride, dst, dst_stride, x0, y0,
                                    b_w, b_h, ref_area);
    return;
  }
  const int valid_start_col = ref_area->pad_block.x0 - x0;
  const int valid_end_col = ref_area->pad_block.x1 - 1 - x0;
  const int cur_ref_row = y0 - ref_area->pad_block.y0;
  const int last_ref_row = ref_area->pad_block.y1 - ref_area->pad_block.y0 - 1;
  const int clamped_start_row = clamp(cur_ref_row, 0, last_ref_row);
  const uint16_t *src_row =
      src + (ptrdiff_t)(clamped_start_row - cur_ref_row) * src_stride;
  PadRowConfig cfg;
  cfg.is_wide_chunk = b_w >= 16;
  cfg.right_chunk_col = b_w - (cfg.is_wide_chunk ? 16 : 8);
  if (valid_start_col >= b_w - 1 || valid_end_col <= 0) {
    const int border_col = valid_end_col <= 0 ? valid_end_col : valid_start_col;
    pad_block_rows(PAD_ROW_BROADCAST_PIXEL, src_row + border_col, src_stride,
                   dst, dst_stride, &cfg, b_h, cur_ref_row, last_ref_row);
    return;
  }
  const int lane1_col = cfg.is_wide_chunk ? 8 : 0;
  const int lane2_col = cfg.right_chunk_col;
  const int lane3_col = cfg.right_chunk_col + lane1_col;
  cfg.lane_src_offset[0] =
      clamp_lane_src_offset(0, valid_start_col, valid_end_col);
  cfg.lane_src_offset[1] =
      clamp_lane_src_offset(lane1_col, valid_start_col, valid_end_col);
  cfg.lane_src_offset[2] =
      clamp_lane_src_offset(lane2_col, valid_start_col, valid_end_col);
  cfg.lane_src_offset[3] =
      clamp_lane_src_offset(lane3_col, valid_start_col, valid_end_col);
  cfg.shuffle_mask[0] =
      yy_set_m128i(build_lane_shuffle_mask(lane1_col, cfg.lane_src_offset[1],
                                           valid_start_col, valid_end_col),
                   build_lane_shuffle_mask(0, cfg.lane_src_offset[0],
                                           valid_start_col, valid_end_col));
  cfg.shuffle_mask[1] =
      yy_set_m128i(build_lane_shuffle_mask(lane3_col, cfg.lane_src_offset[3],
                                           valid_start_col, valid_end_col),
                   build_lane_shuffle_mask(lane2_col, cfg.lane_src_offset[2],
                                           valid_start_col, valid_end_col));
  pad_block_rows(PAD_ROW_GATHER_LANES, src_row, src_stride, dst, dst_stride,
                 &cfg, b_h, cur_ref_row, last_ref_row);
}

// Picks a single-pshufb fast path when each 8-px lane has >= 1 valid pixel.
static INLINE PadRowMode select_pad_row_mode(int b_w, int valid_start_col,
                                             int valid_end_col) {
  if (b_w > 16) {
    if (b_w <= 32 && valid_start_col <= 7 && valid_end_col >= b_w - 8)
      return PAD_ROW_WIDE_16PX;
  } else if (b_w >= 8) {
    if (valid_start_col <= 7 && valid_end_col >= 8)
      return PAD_ROW_NARROW_ALIGN_LEFT;
    if (valid_start_col <= b_w - 9 && valid_end_col >= b_w - 8)
      return PAD_ROW_NARROW_ALIGN_RIGHT;
  }
  return PAD_ROW_FALLBACK;
}

void refinemv_highbd_pad_mc_border_avx2(const uint16_t *src, int src_stride,
                                        uint16_t *dst, int dst_stride, int x0,
                                        int y0, int b_w, int b_h,
                                        const ReferenceArea *ref_area) {
  const int valid_start_col = ref_area->pad_block.x0 - x0;
  const int valid_end_col = ref_area->pad_block.x1 - 1 - x0;
  const PadRowMode mode =
      select_pad_row_mode(b_w, valid_start_col, valid_end_col);
  if (mode == PAD_ROW_FALLBACK) {
    pad_mc_border_fallback(src, src_stride, dst, dst_stride, x0, y0, b_w, b_h,
                           ref_area);
    return;
  }
  const int cur_ref_row = y0 - ref_area->pad_block.y0;
  const int last_ref_row = ref_area->pad_block.y1 - ref_area->pad_block.y0 - 1;
  const int clamped_start_row = clamp(cur_ref_row, 0, last_ref_row);
  const uint16_t *src_row =
      src + (ptrdiff_t)(clamped_start_row - cur_ref_row) * src_stride;
  const int copy_start_col = AVMMAX(valid_start_col, 0);
  const int copy_end_col = AVMMIN(valid_end_col, b_w - 1);
  PadRowConfig cfg;
  cfg.is_wide_chunk = mode == PAD_ROW_WIDE_16PX;
  if (mode == PAD_ROW_WIDE_16PX) {
    const __m256i byte_offsets =
        _mm256_load_si256((const __m256i *)pixel_byte_offsets);
    cfg.right_chunk_col = b_w - 16;
    cfg.shuffle_mask[0] =
        build_clamp_shuffle_mask(byte_offsets, copy_start_col, copy_end_col);
    cfg.shuffle_mask[1] = build_clamp_shuffle_mask(
        byte_offsets, copy_start_col - cfg.right_chunk_col,
        copy_end_col - cfg.right_chunk_col);
    pad_block_rows(PAD_ROW_WIDE_16PX, src_row, src_stride, dst, dst_stride,
                   &cfg, b_h, cur_ref_row, last_ref_row);
    return;
  }
  cfg.right_chunk_col = b_w - 8;
  cfg.shuffle_mask[0] = build_clamp_shuffle_mask(
      yy_loadu2_128(pixel_byte_offsets + 2 * cfg.right_chunk_col,
                    pixel_byte_offsets),
      copy_start_col, copy_end_col);
  if (mode == PAD_ROW_NARROW_ALIGN_LEFT) {
    pad_block_rows(PAD_ROW_NARROW_ALIGN_LEFT, src_row, src_stride, dst,
                   dst_stride, &cfg, b_h, cur_ref_row, last_ref_row);
  } else {
    // Offset mask because the load starts (8 - right_chunk_col) px before x0.
    cfg.shuffle_mask[0] = _mm256_add_epi8(
        cfg.shuffle_mask[0],
        _mm256_set1_epi8((char)(2 * (8 - cfg.right_chunk_col))));
    pad_block_rows(PAD_ROW_NARROW_ALIGN_RIGHT, src_row, src_stride, dst,
                   dst_stride, &cfg, b_h, cur_ref_row, last_ref_row);
  }
}
