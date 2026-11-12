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
#include <string.h>

#include "config/av2_rtcd.h"
#include "config/avm_dsp_rtcd.h"
#include "avm_dsp/x86/intrapred_x86.h"
#include "avm_dsp/x86/lpf_common_sse2.h"
#include "avm_dsp/x86/synonyms.h"
#include "avm_ports/bitops.h"
#include "av2/common/reconintra.h"
#include "av2/common/warped_motion.h"

static DECLARE_ALIGNED(16, uint8_t, HighbdLoadMaskx[8][16]) = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 },
  { 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 },
  { 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
  { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7 },
  { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5 },
  { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3 },
  { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 },
};

static DECLARE_ALIGNED(32, uint16_t, HighbdBaseMask[17][16]) = {
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0, 0, 0, 0, 0, 0, 0,
    0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0, 0, 0, 0,
    0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0,
    0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0 },
  { 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff }
};

static INLINE void highbd_transpose4x16_avx2(__m256i *x0, __m256i *x1,
                                             __m256i *x2, __m256i *x3,
                                             __m256i *d0, __m256i *d1,
                                             __m256i *d2, __m256i *d3) {
  __m256i w0, w1, w2, w3, ww0, ww1;

  w0 = _mm256_unpacklo_epi16(*x0, *x1);  // 00 10 01 11 02 12 03 13
  w1 = _mm256_unpacklo_epi16(*x2, *x3);  // 20 30 21 31 22 32 23 33
  w2 = _mm256_unpackhi_epi16(*x0, *x1);  // 40 50 41 51 42 52 43 53
  w3 = _mm256_unpackhi_epi16(*x2, *x3);  // 60 70 61 71 62 72 63 73

  ww0 = _mm256_unpacklo_epi32(w0, w1);  // 00 10 20 30 01 11 21 31
  ww1 = _mm256_unpacklo_epi32(w2, w3);  // 40 50 60 70 41 51 61 71

  *d0 = _mm256_unpacklo_epi64(ww0, ww1);  // 00 10 20 30 40 50 60 70
  *d1 = _mm256_unpackhi_epi64(ww0, ww1);  // 01 11 21 31 41 51 61 71

  ww0 = _mm256_unpackhi_epi32(w0, w1);  // 02 12 22 32 03 13 23 33
  ww1 = _mm256_unpackhi_epi32(w2, w3);  // 42 52 62 72 43 53 63 73

  *d2 = _mm256_unpacklo_epi64(ww0, ww1);  // 02 12 22 32 42 52 62 72
  *d3 = _mm256_unpackhi_epi64(ww0, ww1);  // 03 13 23 33 43 53 63 73
}

static INLINE void highbd_transpose8x16_16x8_avx2(
    __m256i *x0, __m256i *x1, __m256i *x2, __m256i *x3, __m256i *x4,
    __m256i *x5, __m256i *x6, __m256i *x7, __m256i *d0, __m256i *d1,
    __m256i *d2, __m256i *d3, __m256i *d4, __m256i *d5, __m256i *d6,
    __m256i *d7) {
  __m256i w0, w1, w2, w3, ww0, ww1;

  w0 = _mm256_unpacklo_epi16(*x0, *x1);  // 00 10 01 11 02 12 03 13
  w1 = _mm256_unpacklo_epi16(*x2, *x3);  // 20 30 21 31 22 32 23 33
  w2 = _mm256_unpacklo_epi16(*x4, *x5);  // 40 50 41 51 42 52 43 53
  w3 = _mm256_unpacklo_epi16(*x6, *x7);  // 60 70 61 71 62 72 63 73

  ww0 = _mm256_unpacklo_epi32(w0, w1);  // 00 10 20 30 01 11 21 31
  ww1 = _mm256_unpacklo_epi32(w2, w3);  // 40 50 60 70 41 51 61 71

  *d0 = _mm256_unpacklo_epi64(ww0, ww1);  // 00 10 20 30 40 50 60 70
  *d1 = _mm256_unpackhi_epi64(ww0, ww1);  // 01 11 21 31 41 51 61 71

  ww0 = _mm256_unpackhi_epi32(w0, w1);  // 02 12 22 32 03 13 23 33
  ww1 = _mm256_unpackhi_epi32(w2, w3);  // 42 52 62 72 43 53 63 73

  *d2 = _mm256_unpacklo_epi64(ww0, ww1);  // 02 12 22 32 42 52 62 72
  *d3 = _mm256_unpackhi_epi64(ww0, ww1);  // 03 13 23 33 43 53 63 73

  w0 = _mm256_unpackhi_epi16(*x0, *x1);  // 04 14 05 15 06 16 07 17
  w1 = _mm256_unpackhi_epi16(*x2, *x3);  // 24 34 25 35 26 36 27 37
  w2 = _mm256_unpackhi_epi16(*x4, *x5);  // 44 54 45 55 46 56 47 57
  w3 = _mm256_unpackhi_epi16(*x6, *x7);  // 64 74 65 75 66 76 67 77

  ww0 = _mm256_unpacklo_epi32(w0, w1);  // 04 14 24 34 05 15 25 35
  ww1 = _mm256_unpacklo_epi32(w2, w3);  // 44 54 64 74 45 55 65 75

  *d4 = _mm256_unpacklo_epi64(ww0, ww1);  // 04 14 24 34 44 54 64 74
  *d5 = _mm256_unpackhi_epi64(ww0, ww1);  // 05 15 25 35 45 55 65 75

  ww0 = _mm256_unpackhi_epi32(w0, w1);  // 06 16 26 36 07 17 27 37
  ww1 = _mm256_unpackhi_epi32(w2, w3);  // 46 56 66 76 47 57 67 77

  *d6 = _mm256_unpacklo_epi64(ww0, ww1);  // 06 16 26 36 46 56 66 76
  *d7 = _mm256_unpackhi_epi64(ww0, ww1);  // 07 17 27 37 47 57 67 77
}

static INLINE void highbd_transpose16x16_avx2(__m256i *x, __m256i *d) {
  __m256i w0, w1, w2, w3, ww0, ww1;
  __m256i dd[16];
  w0 = _mm256_unpacklo_epi16(x[0], x[1]);
  w1 = _mm256_unpacklo_epi16(x[2], x[3]);
  w2 = _mm256_unpacklo_epi16(x[4], x[5]);
  w3 = _mm256_unpacklo_epi16(x[6], x[7]);

  ww0 = _mm256_unpacklo_epi32(w0, w1);  //
  ww1 = _mm256_unpacklo_epi32(w2, w3);  //

  dd[0] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[1] = _mm256_unpackhi_epi64(ww0, ww1);

  ww0 = _mm256_unpackhi_epi32(w0, w1);  //
  ww1 = _mm256_unpackhi_epi32(w2, w3);  //

  dd[2] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[3] = _mm256_unpackhi_epi64(ww0, ww1);

  w0 = _mm256_unpackhi_epi16(x[0], x[1]);
  w1 = _mm256_unpackhi_epi16(x[2], x[3]);
  w2 = _mm256_unpackhi_epi16(x[4], x[5]);
  w3 = _mm256_unpackhi_epi16(x[6], x[7]);

  ww0 = _mm256_unpacklo_epi32(w0, w1);  //
  ww1 = _mm256_unpacklo_epi32(w2, w3);  //

  dd[4] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[5] = _mm256_unpackhi_epi64(ww0, ww1);

  ww0 = _mm256_unpackhi_epi32(w0, w1);  //
  ww1 = _mm256_unpackhi_epi32(w2, w3);  //

  dd[6] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[7] = _mm256_unpackhi_epi64(ww0, ww1);

  w0 = _mm256_unpacklo_epi16(x[8], x[9]);
  w1 = _mm256_unpacklo_epi16(x[10], x[11]);
  w2 = _mm256_unpacklo_epi16(x[12], x[13]);
  w3 = _mm256_unpacklo_epi16(x[14], x[15]);

  ww0 = _mm256_unpacklo_epi32(w0, w1);
  ww1 = _mm256_unpacklo_epi32(w2, w3);

  dd[8] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[9] = _mm256_unpackhi_epi64(ww0, ww1);

  ww0 = _mm256_unpackhi_epi32(w0, w1);
  ww1 = _mm256_unpackhi_epi32(w2, w3);

  dd[10] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[11] = _mm256_unpackhi_epi64(ww0, ww1);

  w0 = _mm256_unpackhi_epi16(x[8], x[9]);
  w1 = _mm256_unpackhi_epi16(x[10], x[11]);
  w2 = _mm256_unpackhi_epi16(x[12], x[13]);
  w3 = _mm256_unpackhi_epi16(x[14], x[15]);

  ww0 = _mm256_unpacklo_epi32(w0, w1);
  ww1 = _mm256_unpacklo_epi32(w2, w3);

  dd[12] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[13] = _mm256_unpackhi_epi64(ww0, ww1);

  ww0 = _mm256_unpackhi_epi32(w0, w1);
  ww1 = _mm256_unpackhi_epi32(w2, w3);

  dd[14] = _mm256_unpacklo_epi64(ww0, ww1);
  dd[15] = _mm256_unpackhi_epi64(ww0, ww1);

  for (int i = 0; i < 8; i++) {
    d[i] = _mm256_insertf128_si256(dd[i], _mm256_castsi256_si128(dd[i + 8]), 1);
    d[i + 8] = _mm256_insertf128_si256(dd[i + 8],
                                       _mm256_extracti128_si256(dd[i], 1), 0);
  }
}

static AVM_FORCE_INLINE void highbd_transpose16x4_8x8_avx2(__m128i *dstvec,
                                                           __m256i *d) {
  // r0 = 00 10 01 11 02 12 03 13
  const __m128i r0 = _mm_unpacklo_epi16(dstvec[0], dstvec[1]);
  // r1 = 20 30 21 31 22 32 23 33
  const __m128i r1 = _mm_unpacklo_epi16(dstvec[2], dstvec[3]);
  // r2 = 40 50 41 51 42 52 43 53
  const __m128i r2 = _mm_unpacklo_epi16(dstvec[4], dstvec[5]);
  // r3 = 60 70 61 71 62 72 63 73
  const __m128i r3 = _mm_unpacklo_epi16(dstvec[6], dstvec[7]);
  // r4 = 80 90 81 91 82 92 83 93
  const __m128i r4 = _mm_unpacklo_epi16(dstvec[8], dstvec[9]);
  // r5 = 100 110 101 111 102 112 103 113
  const __m128i r5 = _mm_unpacklo_epi16(dstvec[10], dstvec[11]);
  // r6 = 120 130 121 131 122 132 123 133
  const __m128i r6 = _mm_unpacklo_epi16(dstvec[12], dstvec[13]);
  // r7 = 140 150 141 151 142 152 143 153
  const __m128i r7 = _mm_unpacklo_epi16(dstvec[14], dstvec[15]);

  // 00 10 01 11 02 12 03 13 | 80 90 81 91 82 92 83 93
  const __m256i dstvec256_0 =
      _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r4, 0x1);
  // 20 30 21 31 22 32 23 33 | 100 110 101 111 102 112 103 113
  const __m256i dstvec256_1 =
      _mm256_insertf128_si256(_mm256_castsi128_si256(r1), r5, 0x1);
  // 40 50 41 51 42 52 43 53 | 120 130 121 131 122 132 123 133
  const __m256i dstvec256_2 =
      _mm256_insertf128_si256(_mm256_castsi128_si256(r2), r6, 0x1);
  // 60 70 61 71 62 72 63 73 | 140 150 141 151 142 152 143 153
  const __m256i dstvec256_3 =
      _mm256_insertf128_si256(_mm256_castsi128_si256(r3), r7, 0x1);

  // 00 10 20 30 01 11 21 31 | 80 90 100 110 81 91 101 111
  const __m256i r8 = _mm256_unpacklo_epi32(dstvec256_0, dstvec256_1);
  // 02 12 22 32 03 13 23 33 | 82 92 102 112 83 93 103 113
  const __m256i r9 = _mm256_unpackhi_epi32(dstvec256_0, dstvec256_1);
  // 40 50 60 70 41 51 61 71 | 120 130 140 150 121 131 141 151
  const __m256i r10 = _mm256_unpacklo_epi32(dstvec256_2, dstvec256_3);
  // 42 52 62 72 43 53 63 73 | 122 132 142 152 123 133 143 153
  const __m256i r11 = _mm256_unpackhi_epi32(dstvec256_2, dstvec256_3);

  // 00 10 20 30 40 50 60 70 | 80 90 100 110 120 130 140 150
  d[0] = _mm256_unpacklo_epi64(r8, r10);
  // 01 11 21 31 41 51 61 71 | 81 91 101 111 121 131 141 151
  d[1] = _mm256_unpackhi_epi64(r8, r10);
  // 02 12 22 32 42 52 62 72 | 82 92 102 112 122 132 142 152
  d[2] = _mm256_unpacklo_epi64(r9, r11);
  // 03 13 23 33 43 53 63 73 | 83 93 103 113 123 133 143 153
  d[3] = _mm256_unpackhi_epi64(r9, r11);
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_4xN_internal_avx2(
    int N, __m128i *dst, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((N + 4) - 1 + (mrl_index << 1));

  assert(dx > 0);
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16;
  __m256i diff, c3f;
  __m128i a_mbase_x, max_base_x128, base_inc128, mask128;
  __m128i a0_128, a1_128;
  a16 = _mm256_set1_epi16(16);
  a_mbase_x = _mm_set1_epi16(above[max_base_x]);
  max_base_x128 = _mm_set1_epi16(max_base_x);
  c3f = _mm256_set1_epi16(0x3f);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res, shift;
    __m128i res1;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dst[i] = a_mbase_x;  // save 4 values
      }
      return;
    }

    a0_128 = _mm_loadu_si128((__m128i *)(above + base));
    a1_128 = _mm_loadu_si128((__m128i *)(above + base + 1));
    base_inc128 = _mm_setr_epi16(base, base + 1, base + 2, base + 3, base + 4,
                                 base + 5, base + 6, base + 7);
    shift = _mm256_srli_epi16(_mm256_and_si256(_mm256_set1_epi16(x), c3f), 1);
    a0 = _mm256_castsi128_si256(a0_128);
    a1 = _mm256_castsi128_si256(a1_128);
    diff = _mm256_sub_epi16(a1, a0);   // a[x+1] - a[x]
    a32 = _mm256_slli_epi16(a0, 5);    // a[x] * 32
    a32 = _mm256_add_epi16(a32, a16);  // a[x] * 32 + 16

    b = _mm256_mullo_epi16(diff, shift);
    res = _mm256_add_epi16(a32, b);
    res = _mm256_srli_epi16(res, 5);
    res1 = _mm256_castsi256_si128(res);

    mask128 = _mm_cmpgt_epi16(max_base_x128, base_inc128);
    dst[r] = _mm_blendv_epi8(a_mbase_x, res1, mask128);
    x += dx;
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_32bit_z1_4xN_internal_avx2(
    int N, __m128i *dst, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((N + 4) - 1 + (mrl_index << 1));

  assert(dx > 0);
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16;
  __m256i diff;
  __m128i a_mbase_x, max_base_x128, base_inc128, mask128;

  a16 = _mm256_set1_epi32(16);
  a_mbase_x = _mm_set1_epi16(above[max_base_x]);
  max_base_x128 = _mm_set1_epi32(max_base_x);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res, shift;
    __m128i res1;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dst[i] = a_mbase_x;  // save 4 values
      }
      return;
    }

    a0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base)));
    a1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base + 1)));

    base_inc128 = _mm_setr_epi32(base, base + 1, base + 2, base + 3);
    shift = _mm256_srli_epi32(
        _mm256_and_si256(_mm256_set1_epi32(x), _mm256_set1_epi32(0x3f)), 1);

    diff = _mm256_sub_epi32(a1, a0);   // a[x+1] - a[x]
    a32 = _mm256_slli_epi32(a0, 5);    // a[x] * 32
    a32 = _mm256_add_epi32(a32, a16);  // a[x] * 32 + 16

    b = _mm256_mullo_epi32(diff, shift);
    res = _mm256_add_epi32(a32, b);
    res = _mm256_srli_epi32(res, 5);

    res1 = _mm256_castsi256_si128(res);
    res1 = _mm_packus_epi32(res1, res1);

    mask128 = _mm_cmpgt_epi32(max_base_x128, base_inc128);
    mask128 = _mm_packs_epi32(mask128, mask128);  // goto 16 bit
    dst[r] = _mm_blendv_epi8(a_mbase_x, res1, mask128);
    x += dx;
  }
}

static void highbd_dr_prediction_z1_4xN_avx2(int N, uint16_t *dst,
                                             ptrdiff_t stride,
                                             const uint16_t *above, int dx,
                                             int bd, int mrl_index) {
  __m128i dstvec[64];
  if (bd < 12) {
    highbd_dr_prediction_z1_4xN_internal_avx2(N, dstvec, above, dx, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_avx2(N, dstvec, above, dx,
                                                    mrl_index);
  }
  for (int i = 0; i < N; i++) {
    _mm_storel_epi64((__m128i *)(dst + stride * i), dstvec[i]);
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_32bit_z1_8xN_internal_avx2(
    int N, __m128i *dst, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((8 + N) - 1 + (mrl_index << 1));

  assert(dx > 0);
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi32(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi32(max_base_x);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res, res1, shift;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dst[i] = _mm256_castsi256_si128(a_mbase_x);  // save 8 values
      }
      return;
    }

    a0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base)));
    a1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base + 1)));

    base_inc256 = _mm256_setr_epi32(base, base + 1, base + 2, base + 3,
                                    base + 4, base + 5, base + 6, base + 7);
    shift = _mm256_srli_epi32(
        _mm256_and_si256(_mm256_set1_epi32(x), _mm256_set1_epi32(0x3f)), 1);

    diff = _mm256_sub_epi32(a1, a0);   // a[x+1] - a[x]
    a32 = _mm256_slli_epi32(a0, 5);    // a[x] * 32
    a32 = _mm256_add_epi32(a32, a16);  // a[x] * 32 + 16

    b = _mm256_mullo_epi32(diff, shift);
    res = _mm256_add_epi32(a32, b);
    res = _mm256_srli_epi32(res, 5);

    res1 = _mm256_packus_epi32(
        res, _mm256_castsi128_si256(_mm256_extracti128_si256(res, 1)));

    mask256 = _mm256_cmpgt_epi32(max_base_x256, base_inc256);
    mask256 = _mm256_packs_epi32(
        mask256, _mm256_castsi128_si256(
                     _mm256_extracti128_si256(mask256, 1)));  // goto 16 bit
    res1 = _mm256_blendv_epi8(a_mbase_x, res1, mask256);
    dst[r] = _mm256_castsi256_si128(res1);
    x += dx;
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_8xN_internal_avx2(
    int N, __m128i *dst, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((8 + N) - 1 + (mrl_index << 1));

  assert(dx > 0);
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16, c3f;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;
  __m128i a0_x128, a1_x128;

  a16 = _mm256_set1_epi16(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);
  c3f = _mm256_set1_epi16(0x3f);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res, res1, shift;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dst[i] = _mm256_castsi256_si128(a_mbase_x);  // save 8 values
      }
      return;
    }

    a0_x128 = _mm_loadu_si128((__m128i *)(above + base));
    a1_x128 = _mm_loadu_si128((__m128i *)(above + base + 1));
    base_inc256 =
        _mm256_setr_epi16(base, base + 1, base + 2, base + 3, base + 4,
                          base + 5, base + 6, base + 7, 0, 0, 0, 0, 0, 0, 0, 0);
    shift = _mm256_srli_epi16(_mm256_and_si256(_mm256_set1_epi16(x), c3f), 1);
    a0 = _mm256_castsi128_si256(a0_x128);
    a1 = _mm256_castsi128_si256(a1_x128);

    diff = _mm256_sub_epi16(a1, a0);   // a[x+1] - a[x]
    a32 = _mm256_slli_epi16(a0, 5);    // a[x] * 32
    a32 = _mm256_add_epi16(a32, a16);  // a[x] * 32 + 16

    b = _mm256_mullo_epi16(diff, shift);
    res = _mm256_add_epi16(a32, b);
    res = _mm256_srli_epi16(res, 5);

    mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
    res1 = _mm256_blendv_epi8(a_mbase_x, res, mask256);
    dst[r] = _mm256_castsi256_si128(res1);
    x += dx;
  }
}

static void highbd_dr_prediction_z1_8xN_avx2(int N, uint16_t *dst,
                                             ptrdiff_t stride,
                                             const uint16_t *above, int dx,
                                             int bd, int mrl_index) {
  __m128i dstvec[64];
  if (bd < 12) {
    highbd_dr_prediction_z1_8xN_internal_avx2(N, dstvec, above, dx, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_avx2(N, dstvec, above, dx,
                                                    mrl_index);
  }
  for (int i = 0; i < N; i++) {
    _mm_storeu_si128((__m128i *)(dst + stride * i), dstvec[i]);
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_32bit_z1_16xN_internal_avx2(
    int N, __m256i *dstvec, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((16 + N) - 1 + (mrl_index << 1));
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a0_1, a1, a1_1, a32, a16;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi32(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res[2], res1;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dstvec[i] = a_mbase_x;  // save 16 values
      }
      return;
    }
    __m256i shift = _mm256_srli_epi32(
        _mm256_and_si256(_mm256_set1_epi32(x), _mm256_set1_epi32(0x3f)), 1);

    a0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base)));
    a1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base + 1)));

    diff = _mm256_sub_epi32(a1, a0);   // a[x+1] - a[x]
    a32 = _mm256_slli_epi32(a0, 5);    // a[x] * 32
    a32 = _mm256_add_epi32(a32, a16);  // a[x] * 32 + 16
    b = _mm256_mullo_epi32(diff, shift);

    res[0] = _mm256_add_epi32(a32, b);
    res[0] = _mm256_srli_epi32(res[0], 5);
    res[0] = _mm256_packus_epi32(
        res[0], _mm256_castsi128_si256(_mm256_extracti128_si256(res[0], 1)));

    int mdif = max_base_x - base;
    if (mdif > 8) {
      a0_1 =
          _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base + 8)));
      a1_1 =
          _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(above + base + 9)));

      diff = _mm256_sub_epi32(a1_1, a0_1);  // a[x+1] - a[x]
      a32 = _mm256_slli_epi32(a0_1, 5);     // a[x] * 32
      a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16
      b = _mm256_mullo_epi32(diff, shift);

      res[1] = _mm256_add_epi32(a32, b);
      res[1] = _mm256_srli_epi32(res[1], 5);
      res[1] = _mm256_packus_epi32(
          res[1], _mm256_castsi128_si256(_mm256_extracti128_si256(res[1], 1)));
    } else {
      res[1] = a_mbase_x;
    }
    res1 = _mm256_inserti128_si256(res[0], _mm256_castsi256_si128(res[1]),
                                   1);  // 16 16bit values

    base_inc256 = _mm256_setr_epi16(base, base + 1, base + 2, base + 3,
                                    base + 4, base + 5, base + 6, base + 7,
                                    base + 8, base + 9, base + 10, base + 11,
                                    base + 12, base + 13, base + 14, base + 15);
    mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
    dstvec[r] = _mm256_blendv_epi8(a_mbase_x, res1, mask256);
    x += dx;
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_16xN_internal_avx2(
    int N, __m256i *dstvec, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((16 + N) - 1 + (mrl_index << 1));

  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16, c3f;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi16(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);
  c3f = _mm256_set1_epi16(0x3f);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dstvec[i] = a_mbase_x;  // save 16 values
      }
      return;
    }
    __m256i shift =
        _mm256_srli_epi16(_mm256_and_si256(_mm256_set1_epi16(x), c3f), 1);

    a0 = _mm256_loadu_si256((__m256i *)(above + base));
    a1 = _mm256_loadu_si256((__m256i *)(above + base + 1));

    diff = _mm256_sub_epi16(a1, a0);   // a[x+1] - a[x]
    a32 = _mm256_slli_epi16(a0, 5);    // a[x] * 32
    a32 = _mm256_add_epi16(a32, a16);  // a[x] * 32 + 16
    b = _mm256_mullo_epi16(diff, shift);

    res = _mm256_add_epi16(a32, b);
    res = _mm256_srli_epi16(res, 5);  // 16 16bit values

    base_inc256 = _mm256_setr_epi16(base, base + 1, base + 2, base + 3,
                                    base + 4, base + 5, base + 6, base + 7,
                                    base + 8, base + 9, base + 10, base + 11,
                                    base + 12, base + 13, base + 14, base + 15);
    mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
    dstvec[r] = _mm256_blendv_epi8(a_mbase_x, res, mask256);
    x += dx;
  }
}

static void highbd_dr_prediction_z1_16xN_avx2(int N, uint16_t *dst,
                                              ptrdiff_t stride,
                                              const uint16_t *above, int dx,
                                              int bd, int mrl_index) {
  __m256i dstvec[64];
  if (bd < 12) {
    highbd_dr_prediction_z1_16xN_internal_avx2(N, dstvec, above, dx, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_avx2(N, dstvec, above, dx,
                                                     mrl_index);
  }
  for (int i = 0; i < N; i++) {
    _mm256_storeu_si256((__m256i *)(dst + stride * i), dstvec[i]);
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_32bit_z1_32xN_internal_avx2(
    int N, __m256i *dstvec, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((32 + N) - 1 + (mrl_index << 1));
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a0_1, a1, a1_1, a32, a16, c3f;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi32(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);
  c3f = _mm256_set1_epi16(0x3f);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res[2], res1;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dstvec[i] = a_mbase_x;  // save 32 values
        dstvec[i + N] = a_mbase_x;
      }
      return;
    }

    __m256i shift =
        _mm256_srli_epi32(_mm256_and_si256(_mm256_set1_epi32(x), c3f), 1);

    for (int j = 0; j < 32; j += 16) {
      int mdif = max_base_x - (base + j);
      if (mdif <= 0) {
        res1 = a_mbase_x;
      } else {
        a0 = _mm256_cvtepu16_epi32(
            _mm_loadu_si128((__m128i *)(above + base + j)));
        a1 = _mm256_cvtepu16_epi32(
            _mm_loadu_si128((__m128i *)(above + base + 1 + j)));

        diff = _mm256_sub_epi32(a1, a0);   // a[x+1] - a[x]
        a32 = _mm256_slli_epi32(a0, 5);    // a[x] * 32
        a32 = _mm256_add_epi32(a32, a16);  // a[x] * 32 + 16
        b = _mm256_mullo_epi32(diff, shift);

        res[0] = _mm256_add_epi32(a32, b);
        res[0] = _mm256_srli_epi32(res[0], 5);
        res[0] = _mm256_packus_epi32(
            res[0],
            _mm256_castsi128_si256(_mm256_extracti128_si256(res[0], 1)));
        if (mdif > 8) {
          a0_1 = _mm256_cvtepu16_epi32(
              _mm_loadu_si128((__m128i *)(above + base + 8 + j)));
          a1_1 = _mm256_cvtepu16_epi32(
              _mm_loadu_si128((__m128i *)(above + base + 9 + j)));

          diff = _mm256_sub_epi32(a1_1, a0_1);  // a[x+1] - a[x]
          a32 = _mm256_slli_epi32(a0_1, 5);     // a[x] * 32
          a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16
          b = _mm256_mullo_epi32(diff, shift);

          res[1] = _mm256_add_epi32(a32, b);
          res[1] = _mm256_srli_epi32(res[1], 5);
          res[1] = _mm256_packus_epi32(
              res[1],
              _mm256_castsi128_si256(_mm256_extracti128_si256(res[1], 1)));
        } else {
          res[1] = a_mbase_x;
        }
        res1 = _mm256_inserti128_si256(res[0], _mm256_castsi256_si128(res[1]),
                                       1);  // 16 16bit values
        base_inc256 = _mm256_setr_epi16(
            base + j, base + j + 1, base + j + 2, base + j + 3, base + j + 4,
            base + j + 5, base + j + 6, base + j + 7, base + j + 8,
            base + j + 9, base + j + 10, base + j + 11, base + j + 12,
            base + j + 13, base + j + 14, base + j + 15);

        mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
        res1 = _mm256_blendv_epi8(a_mbase_x, res1, mask256);
      }
      if (!j) {
        dstvec[r] = res1;
      } else {
        dstvec[r + N] = res1;
      }
    }
    x += dx;
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_32xN_internal_avx2(
    int N, __m256i *dstvec, const uint16_t *above, int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((32 + N) - 1 + (mrl_index << 1));
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16, c3f;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi16(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);
  c3f = _mm256_set1_epi16(0x3f);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        dstvec[i] = a_mbase_x;  // save 32 values
        dstvec[i + N] = a_mbase_x;
      }
      return;
    }

    __m256i shift =
        _mm256_srli_epi16(_mm256_and_si256(_mm256_set1_epi16(x), c3f), 1);

    for (int j = 0; j < 32; j += 16) {
      int mdif = max_base_x - (base + j);
      if (mdif <= 0) {
        res = a_mbase_x;
      } else {
        a0 = _mm256_loadu_si256((__m256i *)(above + base + j));
        a1 = _mm256_loadu_si256((__m256i *)(above + base + 1 + j));

        diff = _mm256_sub_epi16(a1, a0);   // a[x+1] - a[x]
        a32 = _mm256_slli_epi16(a0, 5);    // a[x] * 32
        a32 = _mm256_add_epi16(a32, a16);  // a[x] * 32 + 16
        b = _mm256_mullo_epi16(diff, shift);

        res = _mm256_add_epi16(a32, b);
        res = _mm256_srli_epi16(res, 5);

        base_inc256 = _mm256_setr_epi16(
            base + j, base + j + 1, base + j + 2, base + j + 3, base + j + 4,
            base + j + 5, base + j + 6, base + j + 7, base + j + 8,
            base + j + 9, base + j + 10, base + j + 11, base + j + 12,
            base + j + 13, base + j + 14, base + j + 15);

        mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
        res = _mm256_blendv_epi8(a_mbase_x, res, mask256);
      }
      if (!j) {
        dstvec[r] = res;
      } else {
        dstvec[r + N] = res;
      }
    }
    x += dx;
  }
}

static void highbd_dr_prediction_z1_32xN_avx2(int N, uint16_t *dst,
                                              ptrdiff_t stride,
                                              const uint16_t *above, int dx,
                                              int bd, int mrl_index) {
  __m256i dstvec[128];
  if (bd < 12) {
    highbd_dr_prediction_z1_32xN_internal_avx2(N, dstvec, above, dx, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_avx2(N, dstvec, above, dx,
                                                     mrl_index);
  }
  for (int i = 0; i < N; i++) {
    _mm256_storeu_si256((__m256i *)(dst + stride * i), dstvec[i]);
    _mm256_storeu_si256((__m256i *)(dst + stride * i + 16), dstvec[i + N]);
  }
}

static void highbd_dr_prediction_32bit_z1_64xN_avx2(int N, uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *above,
                                                    int dx, int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((64 + N) - 1 + (mrl_index << 1));

  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a0_1, a1, a1_1, a32, a16;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi32(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++, dst += stride) {
    __m256i b, res[2], res1;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        _mm256_storeu_si256((__m256i *)dst, a_mbase_x);  // save 32 values
        _mm256_storeu_si256((__m256i *)(dst + 16), a_mbase_x);
        _mm256_storeu_si256((__m256i *)(dst + 32), a_mbase_x);
        _mm256_storeu_si256((__m256i *)(dst + 48), a_mbase_x);
        dst += stride;
      }
      return;
    }

    __m256i shift = _mm256_srli_epi32(
        _mm256_and_si256(_mm256_set1_epi32(x), _mm256_set1_epi32(0x3f)), 1);

    __m128i a0_128, a0_1_128, a1_128, a1_1_128;
    for (int j = 0; j < 64; j += 16) {
      int mdif = max_base_x - (base + j);
      if (mdif <= 0) {
        _mm256_storeu_si256((__m256i *)(dst + j), a_mbase_x);
      } else {
        a0_128 = _mm_loadu_si128((__m128i *)(above + base + j));
        a1_128 = _mm_loadu_si128((__m128i *)(above + base + 1 + j));
        a0 = _mm256_cvtepu16_epi32(a0_128);
        a1 = _mm256_cvtepu16_epi32(a1_128);

        diff = _mm256_sub_epi32(a1, a0);   // a[x+1] - a[x]
        a32 = _mm256_slli_epi32(a0, 5);    // a[x] * 32
        a32 = _mm256_add_epi32(a32, a16);  // a[x] * 32 + 16
        b = _mm256_mullo_epi32(diff, shift);

        res[0] = _mm256_add_epi32(a32, b);
        res[0] = _mm256_srli_epi32(res[0], 5);
        res[0] = _mm256_packus_epi32(
            res[0],
            _mm256_castsi128_si256(_mm256_extracti128_si256(res[0], 1)));
        if (mdif > 8) {
          a0_1_128 = _mm_loadu_si128((__m128i *)(above + base + 8 + j));
          a1_1_128 = _mm_loadu_si128((__m128i *)(above + base + 9 + j));
          a0_1 = _mm256_cvtepu16_epi32(a0_1_128);
          a1_1 = _mm256_cvtepu16_epi32(a1_1_128);

          diff = _mm256_sub_epi32(a1_1, a0_1);  // a[x+1] - a[x]
          a32 = _mm256_slli_epi32(a0_1, 5);     // a[x] * 32
          a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16
          b = _mm256_mullo_epi32(diff, shift);

          res[1] = _mm256_add_epi32(a32, b);
          res[1] = _mm256_srli_epi32(res[1], 5);
          res[1] = _mm256_packus_epi32(
              res[1],
              _mm256_castsi128_si256(_mm256_extracti128_si256(res[1], 1)));
        } else {
          res[1] = a_mbase_x;
        }
        res1 = _mm256_inserti128_si256(res[0], _mm256_castsi256_si128(res[1]),
                                       1);  // 16 16bit values
        base_inc256 = _mm256_setr_epi16(
            base + j, base + j + 1, base + j + 2, base + j + 3, base + j + 4,
            base + j + 5, base + j + 6, base + j + 7, base + j + 8,
            base + j + 9, base + j + 10, base + j + 11, base + j + 12,
            base + j + 13, base + j + 14, base + j + 15);

        mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
        res1 = _mm256_blendv_epi8(a_mbase_x, res1, mask256);
        _mm256_storeu_si256((__m256i *)(dst + j), res1);
      }
    }
    x += dx;
  }
}

static void highbd_dr_prediction_z1_64xN_avx2(int N, uint16_t *dst,
                                              ptrdiff_t stride,
                                              const uint16_t *above, int dx,
                                              int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = ((64 + N) - 1 + (mrl_index << 1));
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0, a1, a32, a16, c3f;
  __m256i a_mbase_x, diff, max_base_x256, base_inc256, mask256;

  a16 = _mm256_set1_epi16(16);
  a_mbase_x = _mm256_set1_epi16(above[max_base_x]);
  max_base_x256 = _mm256_set1_epi16(max_base_x);
  c3f = _mm256_set1_epi16(0x3f);

  int x = dx * (1 + mrl_index);
  for (int r = 0; r < N; r++, dst += stride) {
    __m256i b, res;

    int base = x >> frac_bits;
    if (base >= max_base_x) {
      for (int i = r; i < N; ++i) {
        _mm256_storeu_si256((__m256i *)dst, a_mbase_x);  // save 32 values
        _mm256_storeu_si256((__m256i *)(dst + 16), a_mbase_x);
        _mm256_storeu_si256((__m256i *)(dst + 32), a_mbase_x);
        _mm256_storeu_si256((__m256i *)(dst + 48), a_mbase_x);
        dst += stride;
      }
      return;
    }

    __m256i shift =
        _mm256_srli_epi16(_mm256_and_si256(_mm256_set1_epi16(x), c3f), 1);

    for (int j = 0; j < 64; j += 16) {
      int mdif = max_base_x - (base + j);
      if (mdif <= 0) {
        _mm256_storeu_si256((__m256i *)(dst + j), a_mbase_x);
      } else {
        a0 = _mm256_loadu_si256((__m256i *)(above + base + j));
        a1 = _mm256_loadu_si256((__m256i *)(above + base + 1 + j));

        diff = _mm256_sub_epi16(a1, a0);   // a[x+1] - a[x]
        a32 = _mm256_slli_epi16(a0, 5);    // a[x] * 32
        a32 = _mm256_add_epi16(a32, a16);  // a[x] * 32 + 16
        b = _mm256_mullo_epi16(diff, shift);

        res = _mm256_add_epi16(a32, b);
        res = _mm256_srli_epi16(res, 5);

        base_inc256 = _mm256_setr_epi16(
            base + j, base + j + 1, base + j + 2, base + j + 3, base + j + 4,
            base + j + 5, base + j + 6, base + j + 7, base + j + 8,
            base + j + 9, base + j + 10, base + j + 11, base + j + 12,
            base + j + 13, base + j + 14, base + j + 15);

        mask256 = _mm256_cmpgt_epi16(max_base_x256, base_inc256);
        res = _mm256_blendv_epi8(a_mbase_x, res, mask256);
        _mm256_storeu_si256((__m256i *)(dst + j), res);  // 16 16bit values
      }
    }
    x += dx;
  }
}

// Directional prediction, zone 1: 0 < angle < 90
void av2_highbd_dr_prediction_z1_avx2(uint16_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint16_t *above,
                                      const uint16_t *left, int dx, int dy,
                                      int bd, int mrl_index) {
  (void)left;
  (void)dy;

  switch (bw) {
    case 4:
      highbd_dr_prediction_z1_4xN_avx2(bh, dst, stride, above, dx, bd,
                                       mrl_index);
      break;
    case 8:
      highbd_dr_prediction_z1_8xN_avx2(bh, dst, stride, above, dx, bd,
                                       mrl_index);
      break;
    case 16:
      highbd_dr_prediction_z1_16xN_avx2(bh, dst, stride, above, dx, bd,
                                        mrl_index);
      break;
    case 32:
      highbd_dr_prediction_z1_32xN_avx2(bh, dst, stride, above, dx, bd,
                                        mrl_index);
      break;
    case 64:
      if (bd < 12) {
        highbd_dr_prediction_z1_64xN_avx2(bh, dst, stride, above, dx,
                                          mrl_index);
      } else {
        highbd_dr_prediction_32bit_z1_64xN_avx2(bh, dst, stride, above, dx,
                                                mrl_index);
      }
      break;
    default: break;
  }
  return;
}

static void highbd_transpose_TX_16X16(const uint16_t *src, ptrdiff_t pitchSrc,
                                      uint16_t *dst, ptrdiff_t pitchDst) {
  __m256i r[16];
  __m256i d[16];
  for (int j = 0; j < 16; j++) {
    r[j] = _mm256_loadu_si256((__m256i *)(src + j * pitchSrc));
  }
  highbd_transpose16x16_avx2(r, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + j * pitchDst), d[j]);
  }
}

static void highbd_transpose(const uint16_t *src, ptrdiff_t pitchSrc,
                             uint16_t *dst, ptrdiff_t pitchDst, int width,
                             int height) {
  for (int j = 0; j < height; j += 16)
    for (int i = 0; i < width; i += 16)
      highbd_transpose_TX_16X16(src + i * pitchSrc + j, pitchSrc,
                                dst + j * pitchDst + i, pitchDst);
}

static void highbd_dr_prediction_32bit_z2_Nx4_avx2(int N, uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *above,
                                                   const uint16_t *left, int dx,
                                                   int dy, int mrl_index) {
  const int min_base_x = -(1 + mrl_index);
  const int min_base_y = -(1 + mrl_index);
  const int frac_bits_x = 6;
  const int frac_bits_y = 6;

  assert(dx > 0);
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0_x, a1_x, a32, a16;
  __m256i diff;
  __m128i c3f, min_base_y128;

  a16 = _mm256_set1_epi32(16);
  c3f = _mm_set1_epi32(0x3f);
  min_base_y128 = _mm_set1_epi32(min_base_y);
  __m128i cmrlIdx = _mm_set1_epi32(mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res, shift;
    __m128i resx, resy, resxy;
    __m128i a0_x128, a1_x128;
    int y = r + 1;
    int base_x = (-(y + mrl_index) * dx) >> frac_bits_x;
    int base_shift = 0;
    if (base_x < (min_base_x - 1)) {
      base_shift = (min_base_x - base_x - 1);
    }
    int base_min_diff = (min_base_x - base_x);
    if (base_min_diff > 4) {
      base_min_diff = 4;
    } else {
      if (base_min_diff < 0) base_min_diff = 0;
    }

    if (base_shift > 3) {
      a0_x = _mm256_setzero_si256();
      a1_x = _mm256_setzero_si256();
      shift = _mm256_setzero_si256();
    } else {
      a0_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift));
      a0_x128 =
          _mm_shuffle_epi8(a0_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);
      a1_x128 = _mm_srli_si128(a0_x128, 2);

      shift = _mm256_castsi128_si256(_mm_srli_epi32(
          _mm_and_si128(_mm_setr_epi32(-(y + mrl_index) * dx,
                                       (1 << 6) - (y + mrl_index) * dx,
                                       (2 << 6) - (y + mrl_index) * dx,
                                       (3 << 6) - (y + mrl_index) * dx),
                        c3f),
          1));
      a0_x = _mm256_cvtepu16_epi32(a0_x128);
      a1_x = _mm256_cvtepu16_epi32(a1_x128);
    }
    // y calc
    __m128i a0_y, a1_y, shifty;
    if (base_x < min_base_x) {
      __m128i r6, c1234, dy128, y_c128, base_y_c128, mask128;
      DECLARE_ALIGNED(32, int, base_y_c[4]);
      r6 = _mm_set1_epi32(r << 6);
      dy128 = _mm_set1_epi32(dy);
      c1234 = _mm_setr_epi32(1, 2, 3, 4);
      __m128i c1234_ = _mm_add_epi32(c1234, cmrlIdx);
      y_c128 = _mm_sub_epi32(r6, _mm_mullo_epi32(c1234_, dy128));
      base_y_c128 = _mm_srai_epi32(y_c128, frac_bits_y);
      mask128 = _mm_cmpgt_epi32(min_base_y128, base_y_c128);
      base_y_c128 = _mm_andnot_si128(mask128, base_y_c128);
      _mm_store_si128((__m128i *)base_y_c, base_y_c128);

      a0_y = _mm_setr_epi32(left[base_y_c[0]], left[base_y_c[1]],
                            left[base_y_c[2]], left[base_y_c[3]]);
      a1_y = _mm_setr_epi32(left[base_y_c[0] + 1], left[base_y_c[1] + 1],
                            left[base_y_c[2] + 1], left[base_y_c[3] + 1]);

      shifty = _mm_srli_epi32(_mm_and_si128(y_c128, c3f), 1);
      a0_x = _mm256_inserti128_si256(a0_x, a0_y, 1);
      a1_x = _mm256_inserti128_si256(a1_x, a1_y, 1);
      shift = _mm256_inserti128_si256(shift, shifty, 1);
    }

    diff = _mm256_sub_epi32(a1_x, a0_x);  // a[x+1] - a[x]
    a32 = _mm256_slli_epi32(a0_x, 5);     // a[x] * 32
    a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16

    b = _mm256_mullo_epi32(diff, shift);
    res = _mm256_add_epi32(a32, b);
    res = _mm256_srli_epi32(res, 5);

    resx = _mm256_castsi256_si128(res);
    resx = _mm_packus_epi32(resx, resx);

    resy = _mm256_extracti128_si256(res, 1);
    resy = _mm_packus_epi32(resy, resy);

    resxy =
        _mm_blendv_epi8(resx, resy, *(__m128i *)HighbdBaseMask[base_min_diff]);
    _mm_storel_epi64((__m128i *)(dst), resxy);
    dst += stride;
  }
}

static void highbd_dr_prediction_z2_Nx4_avx2(int N, uint16_t *dst,
                                             ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int dx,
                                             int dy, int mrl_index) {
  const int min_base_x = -(1 + mrl_index);
  const int min_base_y = -(1 + mrl_index);
  const int frac_bits_x = 6;
  const int frac_bits_y = 6;

  assert(dx > 0);
  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0_x, a1_x, a32, a16;
  __m256i diff;
  __m128i c3f, min_base_y128;

  a16 = _mm256_set1_epi16(16);
  c3f = _mm_set1_epi16(0x3f);
  min_base_y128 = _mm_set1_epi16(min_base_y);
  __m128i cmrlIdx = _mm_set1_epi16(mrl_index);

  for (int r = 0; r < N; r++) {
    __m256i b, res, shift;
    __m128i resx, resy, resxy;
    __m128i a0_x128, a1_x128;
    int y = r + 1;
    int base_x = (-(y + mrl_index) * dx) >> frac_bits_x;
    int base_shift = 0;
    if (base_x < (min_base_x - 1)) {
      base_shift = (min_base_x - base_x - 1);
    }
    int base_min_diff = (min_base_x - base_x);
    if (base_min_diff > 4) {
      base_min_diff = 4;
    } else {
      if (base_min_diff < 0) base_min_diff = 0;
    }

    if (base_shift > 3) {
      a0_x = _mm256_setzero_si256();
      a1_x = _mm256_setzero_si256();
      shift = _mm256_setzero_si256();
    } else {
      a0_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift));
      a0_x128 =
          _mm_shuffle_epi8(a0_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);
      a1_x128 = _mm_srli_si128(a0_x128, 2);

      shift = _mm256_castsi128_si256(_mm_srli_epi16(
          _mm_and_si128(
              _mm_setr_epi16(-(y + mrl_index) * dx,
                             (1 << 6) - (y + mrl_index) * dx,
                             (2 << 6) - (y + mrl_index) * dx,
                             (3 << 6) - (y + mrl_index) * dx, 0, 0, 0, 0),
              c3f),
          1));
      a0_x = _mm256_castsi128_si256(a0_x128);
      a1_x = _mm256_castsi128_si256(a1_x128);
    }
    // y calc
    __m128i a0_y, a1_y, shifty;
    if (base_x < min_base_x) {
      __m128i r6, c1234, dy128, y_c128, base_y_c128, mask128;
      DECLARE_ALIGNED(32, int16_t, base_y_c[8]);
      r6 = _mm_set1_epi16(r << 6);
      dy128 = _mm_set1_epi16(dy);
      c1234 = _mm_setr_epi16(1, 2, 3, 4, 0, 0, 0, 0);
      __m128i c1234_ = _mm_add_epi16(c1234, cmrlIdx);
      y_c128 = _mm_sub_epi16(r6, _mm_mullo_epi16(c1234_, dy128));
      base_y_c128 = _mm_srai_epi16(y_c128, frac_bits_y);
      mask128 = _mm_cmpgt_epi16(min_base_y128, base_y_c128);
      base_y_c128 = _mm_andnot_si128(mask128, base_y_c128);
      _mm_store_si128((__m128i *)base_y_c, base_y_c128);

      a0_y = _mm_setr_epi16(left[base_y_c[0]], left[base_y_c[1]],
                            left[base_y_c[2]], left[base_y_c[3]], 0, 0, 0, 0);
      a1_y = _mm_setr_epi16(left[base_y_c[0] + 1], left[base_y_c[1] + 1],
                            left[base_y_c[2] + 1], left[base_y_c[3] + 1], 0, 0,
                            0, 0);

      shifty = _mm_srli_epi16(_mm_and_si128(y_c128, c3f), 1);
      a0_x = _mm256_inserti128_si256(a0_x, a0_y, 1);
      a1_x = _mm256_inserti128_si256(a1_x, a1_y, 1);
      shift = _mm256_inserti128_si256(shift, shifty, 1);
    }

    diff = _mm256_sub_epi16(a1_x, a0_x);  // a[x+1] - a[x]
    a32 = _mm256_slli_epi16(a0_x, 5);     // a[x] * 32
    a32 = _mm256_add_epi16(a32, a16);     // a[x] * 32 + 16

    b = _mm256_mullo_epi16(diff, shift);
    res = _mm256_add_epi16(a32, b);
    res = _mm256_srli_epi16(res, 5);

    resx = _mm256_castsi256_si128(res);
    resy = _mm256_extracti128_si256(res, 1);
    resxy =
        _mm_blendv_epi8(resx, resy, *(__m128i *)HighbdBaseMask[base_min_diff]);
    _mm_storel_epi64((__m128i *)(dst), resxy);
    dst += stride;
  }
}

static void highbd_dr_prediction_32bit_z2_Nx8_avx2(int N, uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *above,
                                                   const uint16_t *left, int dx,
                                                   int dy, int mrl_index) {
  const int min_base_x = -(1 + mrl_index);
  const int min_base_y = -(1 + mrl_index);
  const int frac_bits_x = 6;
  const int frac_bits_y = 6;

  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0_x, a1_x, a0_y, a1_y, a32, a16, c3f, min_base_y256;
  __m256i diff;
  __m128i a0_x128, a1_x128;

  a16 = _mm256_set1_epi32(16);
  c3f = _mm256_set1_epi32(0x3f);
  min_base_y256 = _mm256_set1_epi32(min_base_y);
  __m256i cmrlIdx = _mm256_set1_epi32(mrl_index);

  for (int r = 0; r < N; r++) {
    __m256i b, res, shift;
    __m128i resx, resy, resxy;
    int y = r + 1;
    int base_x = (-(y + mrl_index) * dx) >> frac_bits_x;
    int base_shift = 0;
    if (base_x < (min_base_x - 1)) {
      base_shift = (min_base_x - base_x - 1);
    }
    int base_min_diff = (min_base_x - base_x);
    if (base_min_diff > 8) {
      base_min_diff = 8;
    } else {
      if (base_min_diff < 0) base_min_diff = 0;
    }

    if (base_shift > 7) {
      resx = _mm_setzero_si128();
    } else {
      a0_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift));
      a1_x128 = _mm_loadu_si128((__m128i *)(above + base_x + 1 + base_shift));
      a0_x128 =
          _mm_shuffle_epi8(a0_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);
      a1_x128 =
          _mm_shuffle_epi8(a1_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);

      shift = _mm256_srli_epi32(
          _mm256_and_si256(_mm256_setr_epi32(-(y + mrl_index) * dx,
                                             (1 << 6) - (y + mrl_index) * dx,
                                             (2 << 6) - (y + mrl_index) * dx,
                                             (3 << 6) - (y + mrl_index) * dx,
                                             (4 << 6) - (y + mrl_index) * dx,
                                             (5 << 6) - (y + mrl_index) * dx,
                                             (6 << 6) - (y + mrl_index) * dx,
                                             (7 << 6) - (y + mrl_index) * dx),
                           c3f),
          1);
      a0_x = _mm256_cvtepu16_epi32(a0_x128);
      a1_x = _mm256_cvtepu16_epi32(a1_x128);

      diff = _mm256_sub_epi32(a1_x, a0_x);  // a[x+1] - a[x]
      a32 = _mm256_slli_epi32(a0_x, 5);     // a[x] * 32
      a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16

      b = _mm256_mullo_epi32(diff, shift);
      res = _mm256_add_epi32(a32, b);
      res = _mm256_srli_epi32(res, 5);

      resx = _mm256_castsi256_si128(_mm256_packus_epi32(
          res, _mm256_castsi128_si256(_mm256_extracti128_si256(res, 1))));
    }
    // y calc
    if (base_x < min_base_x) {
      DECLARE_ALIGNED(32, int, base_y_c[8]);
      __m256i r6, c256, dy256, y_c256, base_y_c256, mask256;
      r6 = _mm256_set1_epi32(r << 6);
      dy256 = _mm256_set1_epi32(dy);
      c256 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
      __m256i c256_ = _mm256_add_epi32(c256, cmrlIdx);
      y_c256 = _mm256_sub_epi32(r6, _mm256_mullo_epi32(c256_, dy256));
      base_y_c256 = _mm256_srai_epi32(y_c256, frac_bits_y);
      mask256 = _mm256_cmpgt_epi32(min_base_y256, base_y_c256);
      base_y_c256 = _mm256_andnot_si256(mask256, base_y_c256);
      _mm256_store_si256((__m256i *)base_y_c, base_y_c256);

      a0_y = _mm256_cvtepu16_epi32(_mm_setr_epi16(
          left[base_y_c[0]], left[base_y_c[1]], left[base_y_c[2]],
          left[base_y_c[3]], left[base_y_c[4]], left[base_y_c[5]],
          left[base_y_c[6]], left[base_y_c[7]]));
      a1_y = _mm256_cvtepu16_epi32(_mm_setr_epi16(
          left[base_y_c[0] + 1], left[base_y_c[1] + 1], left[base_y_c[2] + 1],
          left[base_y_c[3] + 1], left[base_y_c[4] + 1], left[base_y_c[5] + 1],
          left[base_y_c[6] + 1], left[base_y_c[7] + 1]));

      shift = _mm256_srli_epi32(_mm256_and_si256(y_c256, c3f), 1);
      diff = _mm256_sub_epi32(a1_y, a0_y);  // a[x+1] - a[x]
      a32 = _mm256_slli_epi32(a0_y, 5);     // a[x] * 32
      a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16

      b = _mm256_mullo_epi32(diff, shift);
      res = _mm256_add_epi32(a32, b);
      res = _mm256_srli_epi32(res, 5);

      resy = _mm256_castsi256_si128(_mm256_packus_epi32(
          res, _mm256_castsi128_si256(_mm256_extracti128_si256(res, 1))));
    } else {
      resy = resx;
    }
    resxy =
        _mm_blendv_epi8(resx, resy, *(__m128i *)HighbdBaseMask[base_min_diff]);
    _mm_storeu_si128((__m128i *)(dst), resxy);
    dst += stride;
  }
}

static void highbd_dr_prediction_z2_Nx8_avx2(int N, uint16_t *dst,
                                             ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int dx,
                                             int dy, int mrl_index) {
  const int min_base_x = -(1 + mrl_index);
  const int min_base_y = -(1 + mrl_index);
  const int frac_bits_x = 6;
  const int frac_bits_y = 6;

  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m128i c3f, min_base_y128;
  __m256i a0_x, a1_x, diff, a32, a16;
  __m128i a0_x128, a1_x128;

  a16 = _mm256_set1_epi16(16);
  c3f = _mm_set1_epi16(0x3f);
  min_base_y128 = _mm_set1_epi16(min_base_y);
  __m128i cmrlIdx = _mm_set1_epi16(mrl_index);
  for (int r = 0; r < N; r++) {
    __m256i b, res, shift;
    __m128i resx, resy, resxy;
    int y = r + 1;
    int base_x = (-(y + mrl_index) * dx) >> frac_bits_x;
    int base_shift = 0;
    if (base_x < (min_base_x - 1)) {
      base_shift = (min_base_x - base_x - 1);
    }
    int base_min_diff = (min_base_x - base_x);
    if (base_min_diff > 8) {
      base_min_diff = 8;
    } else {
      if (base_min_diff < 0) base_min_diff = 0;
    }

    if (base_shift > 7) {
      a0_x = _mm256_setzero_si256();
      a1_x = _mm256_setzero_si256();
      shift = _mm256_setzero_si256();
    } else {
      a0_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift));
      a1_x128 = _mm_loadu_si128((__m128i *)(above + base_x + 1 + base_shift));
      a0_x128 =
          _mm_shuffle_epi8(a0_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);
      a1_x128 =
          _mm_shuffle_epi8(a1_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);

      shift = _mm256_castsi128_si256(_mm_srli_epi16(
          _mm_and_si128(_mm_setr_epi16(-(y + mrl_index) * dx,
                                       (1 << 6) - (y + mrl_index) * dx,
                                       (2 << 6) - (y + mrl_index) * dx,
                                       (3 << 6) - (y + mrl_index) * dx,
                                       (4 << 6) - (y + mrl_index) * dx,
                                       (5 << 6) - (y + mrl_index) * dx,
                                       (6 << 6) - (y + mrl_index) * dx,
                                       (7 << 6) - (y + mrl_index) * dx),
                        c3f),
          1));
      a0_x = _mm256_castsi128_si256(a0_x128);
      a1_x = _mm256_castsi128_si256(a1_x128);
    }

    // y calc
    __m128i a0_y, a1_y, shifty;
    if (base_x < min_base_x) {
      DECLARE_ALIGNED(32, int16_t, base_y_c[8]);
      __m128i r6, c1234, dy128, y_c128, base_y_c128, mask128;
      r6 = _mm_set1_epi16(r << 6);
      dy128 = _mm_set1_epi16(dy);
      c1234 = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
      __m128i c1234_ = _mm_add_epi16(c1234, cmrlIdx);
      y_c128 = _mm_sub_epi16(r6, _mm_mullo_epi16(c1234_, dy128));
      base_y_c128 = _mm_srai_epi16(y_c128, frac_bits_y);
      mask128 = _mm_cmpgt_epi16(min_base_y128, base_y_c128);
      base_y_c128 = _mm_andnot_si128(mask128, base_y_c128);
      _mm_store_si128((__m128i *)base_y_c, base_y_c128);

      a0_y = _mm_setr_epi16(left[base_y_c[0]], left[base_y_c[1]],
                            left[base_y_c[2]], left[base_y_c[3]],
                            left[base_y_c[4]], left[base_y_c[5]],
                            left[base_y_c[6]], left[base_y_c[7]]);
      a1_y = _mm_setr_epi16(left[base_y_c[0] + 1], left[base_y_c[1] + 1],
                            left[base_y_c[2] + 1], left[base_y_c[3] + 1],
                            left[base_y_c[4] + 1], left[base_y_c[5] + 1],
                            left[base_y_c[6] + 1], left[base_y_c[7] + 1]);

      shifty = _mm_srli_epi16(_mm_and_si128(y_c128, c3f), 1);
      a0_x = _mm256_inserti128_si256(a0_x, a0_y, 1);
      a1_x = _mm256_inserti128_si256(a1_x, a1_y, 1);
      shift = _mm256_inserti128_si256(shift, shifty, 1);
    }

    diff = _mm256_sub_epi16(a1_x, a0_x);  // a[x+1] - a[x]
    a32 = _mm256_slli_epi16(a0_x, 5);     // a[x] * 32
    a32 = _mm256_add_epi16(a32, a16);     // a[x] * 32 + 16

    b = _mm256_mullo_epi16(diff, shift);
    res = _mm256_add_epi16(a32, b);
    res = _mm256_srli_epi16(res, 5);

    resx = _mm256_castsi256_si128(res);
    resy = _mm256_extracti128_si256(res, 1);

    resxy =
        _mm_blendv_epi8(resx, resy, *(__m128i *)HighbdBaseMask[base_min_diff]);
    _mm_storeu_si128((__m128i *)(dst), resxy);
    dst += stride;
  }
}

static void highbd_dr_prediction_32bit_z2_HxW_avx2(int H, int W, uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *above,
                                                   const uint16_t *left, int dx,
                                                   int dy, int mrl_index) {
  const int min_base_x = -(1 + mrl_index);
  const int min_base_y = -(1 + mrl_index);
  const int frac_bits_x = 6;
  const int frac_bits_y = 6;

  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0_x, a1_x, a0_y, a1_y, a32, a0_1_x, a1_1_x, a16, c1;
  __m256i diff, min_base_y256, c3f, dy256, c1234, c0123, c8;
  __m128i a0_x128, a1_x128, a0_1_x128, a1_1_x128;
  DECLARE_ALIGNED(32, int, base_y_c[16]);

  a16 = _mm256_set1_epi32(16);
  c1 = _mm256_srli_epi32(a16, 4);
  c8 = _mm256_srli_epi32(a16, 1);
  min_base_y256 = _mm256_set1_epi32(min_base_y);
  c3f = _mm256_set1_epi32(0x3f);
  dy256 = _mm256_set1_epi32(dy);
  c0123 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  c1234 = _mm256_add_epi32(c0123, c1);
  __m256i cmrlIdx = _mm256_set1_epi32(mrl_index);
  for (int r = 0; r < H; r++) {
    __m256i b, res, shift, ydx;
    __m256i resx[2], resy[2];
    __m256i resxy, j256, r6;
    for (int j = 0; j < W; j += 16) {
      j256 = _mm256_set1_epi32(j);
      int y = r + 1;
      ydx = _mm256_set1_epi32((y + mrl_index) * dx);

      int base_x = ((j << 6) - (y + mrl_index) * dx) >> frac_bits_x;
      int base_shift = 0;
      if ((base_x) < (min_base_x - 1)) {
        base_shift = (min_base_x - base_x - 1);
      }
      int base_min_diff = (min_base_x - base_x);
      if (base_min_diff > 16) {
        base_min_diff = 16;
      } else {
        if (base_min_diff < 0) base_min_diff = 0;
      }

      if (base_shift > 7) {
        resx[0] = _mm256_setzero_si256();
      } else {
        a0_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift));
        a1_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift + 1));
        a0_x128 =
            _mm_shuffle_epi8(a0_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);
        a1_x128 =
            _mm_shuffle_epi8(a1_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);

        a0_x = _mm256_cvtepu16_epi32(a0_x128);
        a1_x = _mm256_cvtepu16_epi32(a1_x128);

        r6 = _mm256_slli_epi32(_mm256_add_epi32(c0123, j256), 6);
        shift = _mm256_srli_epi32(
            _mm256_and_si256(_mm256_sub_epi32(r6, ydx), c3f), 1);

        diff = _mm256_sub_epi32(a1_x, a0_x);  // a[x+1] - a[x]
        a32 = _mm256_slli_epi32(a0_x, 5);     // a[x] * 32
        a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16

        b = _mm256_mullo_epi32(diff, shift);
        res = _mm256_add_epi32(a32, b);
        res = _mm256_srli_epi32(res, 5);

        resx[0] = _mm256_packus_epi32(
            res, _mm256_castsi128_si256(_mm256_extracti128_si256(res, 1)));
      }
      int base_shift8 = 0;
      if ((base_x + 8) < (min_base_x - 1)) {
        base_shift8 = (min_base_x - (base_x + 8) - 1);
      }
      if (base_shift8 > 7) {
        resx[1] = _mm256_setzero_si256();
      } else {
        a0_1_x128 =
            _mm_loadu_si128((__m128i *)(above + base_x + base_shift8 + 8));
        a1_1_x128 =
            _mm_loadu_si128((__m128i *)(above + base_x + base_shift8 + 9));
        a0_1_x128 = _mm_shuffle_epi8(a0_1_x128,
                                     *(__m128i *)HighbdLoadMaskx[base_shift8]);
        a1_1_x128 = _mm_shuffle_epi8(a1_1_x128,
                                     *(__m128i *)HighbdLoadMaskx[base_shift8]);

        a0_1_x = _mm256_cvtepu16_epi32(a0_1_x128);
        a1_1_x = _mm256_cvtepu16_epi32(a1_1_x128);

        r6 = _mm256_slli_epi32(
            _mm256_add_epi32(c0123, _mm256_add_epi32(j256, c8)), 6);
        shift = _mm256_srli_epi32(
            _mm256_and_si256(_mm256_sub_epi32(r6, ydx), c3f), 1);

        diff = _mm256_sub_epi32(a1_1_x, a0_1_x);  // a[x+1] - a[x]
        a32 = _mm256_slli_epi32(a0_1_x, 5);       // a[x] * 32
        a32 = _mm256_add_epi32(a32, a16);         // a[x] * 32 + 16
        b = _mm256_mullo_epi32(diff, shift);

        resx[1] = _mm256_add_epi32(a32, b);
        resx[1] = _mm256_srli_epi32(resx[1], 5);
        resx[1] = _mm256_packus_epi32(
            resx[1],
            _mm256_castsi128_si256(_mm256_extracti128_si256(resx[1], 1)));
      }
      resx[0] =
          _mm256_inserti128_si256(resx[0], _mm256_castsi256_si128(resx[1]),
                                  1);  // 16 16bit values

      // y calc
      resy[0] = _mm256_setzero_si256();
      if ((base_x < min_base_x)) {
        __m256i c256, y_c256, y_c_1_256, base_y_c256, mask256;
        r6 = _mm256_set1_epi32(r << 6);
        c256 = _mm256_add_epi32(j256, c1234);
        __m256i c256_ = _mm256_add_epi32(c256, cmrlIdx);
        y_c256 = _mm256_sub_epi32(r6, _mm256_mullo_epi32(c256_, dy256));
        base_y_c256 = _mm256_srai_epi32(y_c256, frac_bits_y);
        mask256 = _mm256_cmpgt_epi32(min_base_y256, base_y_c256);
        base_y_c256 = _mm256_andnot_si256(mask256, base_y_c256);
        _mm256_store_si256((__m256i *)base_y_c, base_y_c256);
        c256 = _mm256_add_epi32(c256_, c8);
        y_c_1_256 = _mm256_sub_epi32(r6, _mm256_mullo_epi32(c256, dy256));
        base_y_c256 = _mm256_srai_epi32(y_c_1_256, frac_bits_y);
        mask256 = _mm256_cmpgt_epi32(min_base_y256, base_y_c256);
        base_y_c256 = _mm256_andnot_si256(mask256, base_y_c256);
        _mm256_store_si256((__m256i *)(base_y_c + 8), base_y_c256);

        a0_y = _mm256_cvtepu16_epi32(_mm_setr_epi16(
            left[base_y_c[0]], left[base_y_c[1]], left[base_y_c[2]],
            left[base_y_c[3]], left[base_y_c[4]], left[base_y_c[5]],
            left[base_y_c[6]], left[base_y_c[7]]));
        a1_y = _mm256_cvtepu16_epi32(_mm_setr_epi16(
            left[base_y_c[0] + 1], left[base_y_c[1] + 1], left[base_y_c[2] + 1],
            left[base_y_c[3] + 1], left[base_y_c[4] + 1], left[base_y_c[5] + 1],
            left[base_y_c[6] + 1], left[base_y_c[7] + 1]));

        shift = _mm256_srli_epi32(_mm256_and_si256(y_c256, c3f), 1);

        diff = _mm256_sub_epi32(a1_y, a0_y);  // a[x+1] - a[x]
        a32 = _mm256_slli_epi32(a0_y, 5);     // a[x] * 32
        a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16

        b = _mm256_mullo_epi32(diff, shift);
        res = _mm256_add_epi32(a32, b);
        res = _mm256_srli_epi32(res, 5);

        resy[0] = _mm256_packus_epi32(
            res, _mm256_castsi128_si256(_mm256_extracti128_si256(res, 1)));

        a0_y = _mm256_cvtepu16_epi32(_mm_setr_epi16(
            left[base_y_c[8]], left[base_y_c[9]], left[base_y_c[10]],
            left[base_y_c[11]], left[base_y_c[12]], left[base_y_c[13]],
            left[base_y_c[14]], left[base_y_c[15]]));
        a1_y = _mm256_cvtepu16_epi32(
            _mm_setr_epi16(left[base_y_c[8] + 1], left[base_y_c[9] + 1],
                           left[base_y_c[10] + 1], left[base_y_c[11] + 1],
                           left[base_y_c[12] + 1], left[base_y_c[13] + 1],
                           left[base_y_c[14] + 1], left[base_y_c[15] + 1]));
        shift = _mm256_srli_epi32(_mm256_and_si256(y_c_1_256, c3f), 1);

        diff = _mm256_sub_epi32(a1_y, a0_y);  // a[x+1] - a[x]
        a32 = _mm256_slli_epi32(a0_y, 5);     // a[x] * 32
        a32 = _mm256_add_epi32(a32, a16);     // a[x] * 32 + 16

        b = _mm256_mullo_epi32(diff, shift);
        res = _mm256_add_epi32(a32, b);
        res = _mm256_srli_epi32(res, 5);

        resy[1] = _mm256_packus_epi32(
            res, _mm256_castsi128_si256(_mm256_extracti128_si256(res, 1)));

        resy[0] =
            _mm256_inserti128_si256(resy[0], _mm256_castsi256_si128(resy[1]),
                                    1);  // 16 16bit values
      }

      resxy = _mm256_blendv_epi8(resx[0], resy[0],
                                 *(__m256i *)HighbdBaseMask[base_min_diff]);
      _mm256_storeu_si256((__m256i *)(dst + j), resxy);
    }  // for j
    dst += stride;
  }
}

static void highbd_dr_prediction_z2_HxW_avx2(int H, int W, uint16_t *dst,
                                             ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int dx,
                                             int dy, int mrl_index) {
  const int min_base_x = -(1 + mrl_index);
  const int min_base_y = -(1 + mrl_index);
  const int frac_bits_x = 6;
  const int frac_bits_y = 6;

  // pre-filter above pixels
  // store in temp buffers:
  //   above[x] * 32 + 16
  //   above[x+1] - above[x]
  // final pixels will be calculated as:
  //   (above[x] * 32 + 16 + (above[x+1] - above[x]) * shift) >> 5
  __m256i a0_x, a1_x, a32, a16, c3f, c1;
  __m256i diff, min_base_y256, dy256, c1234, c0123;
  DECLARE_ALIGNED(32, int16_t, base_y_c[16]);

  a16 = _mm256_set1_epi16(16);
  c1 = _mm256_srli_epi16(a16, 4);
  min_base_y256 = _mm256_set1_epi16(min_base_y);
  c3f = _mm256_set1_epi16(0x3f);
  dy256 = _mm256_set1_epi16(dy);
  c0123 =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  c1234 = _mm256_add_epi16(c0123, c1);
  __m256i cmrlIdx = _mm256_set1_epi16(mrl_index);

  for (int r = 0; r < H; r++) {
    __m256i b, res, shift;
    __m256i resx, resy, ydx;
    __m256i resxy, j256, r6;
    __m128i a0_x128, a1_x128, a0_1_x128, a1_1_x128;
    int y = r + 1;
    ydx = _mm256_set1_epi16((short)((y + mrl_index) * dx));

    for (int j = 0; j < W; j += 16) {
      j256 = _mm256_set1_epi16(j);
      int base_x = ((j << 6) - (y + mrl_index) * dx) >> frac_bits_x;
      int base_shift = 0;
      if ((base_x) < (min_base_x - 1)) {
        base_shift = (min_base_x - (base_x)-1);
      }
      int base_min_diff = (min_base_x - base_x);
      if (base_min_diff > 16) {
        base_min_diff = 16;
      } else {
        if (base_min_diff < 0) base_min_diff = 0;
      }

      if (base_shift < 8) {
        a0_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift));
        a1_x128 = _mm_loadu_si128((__m128i *)(above + base_x + base_shift + 1));
        a0_x128 =
            _mm_shuffle_epi8(a0_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);
        a1_x128 =
            _mm_shuffle_epi8(a1_x128, *(__m128i *)HighbdLoadMaskx[base_shift]);

        a0_x = _mm256_castsi128_si256(a0_x128);
        a1_x = _mm256_castsi128_si256(a1_x128);
      } else {
        a0_x = _mm256_setzero_si256();
        a1_x = _mm256_setzero_si256();
      }

      int base_shift1 = 0;
      if (base_shift > 8) {
        base_shift1 = base_shift - 8;
      }
      if (base_shift1 < 8) {
        a0_1_x128 =
            _mm_loadu_si128((__m128i *)(above + base_x + base_shift1 + 8));
        a1_1_x128 =
            _mm_loadu_si128((__m128i *)(above + base_x + base_shift1 + 9));
        a0_1_x128 = _mm_shuffle_epi8(a0_1_x128,
                                     *(__m128i *)HighbdLoadMaskx[base_shift1]);
        a1_1_x128 = _mm_shuffle_epi8(a1_1_x128,
                                     *(__m128i *)HighbdLoadMaskx[base_shift1]);

        a0_x = _mm256_inserti128_si256(a0_x, a0_1_x128, 1);
        a1_x = _mm256_inserti128_si256(a1_x, a1_1_x128, 1);
      }
      r6 = _mm256_slli_epi16(_mm256_add_epi16(c0123, j256), 6);
      shift = _mm256_srli_epi16(
          _mm256_and_si256(_mm256_sub_epi16(r6, ydx), c3f), 1);

      diff = _mm256_sub_epi16(a1_x, a0_x);  // a[x+1] - a[x]
      a32 = _mm256_slli_epi16(a0_x, 5);     // a[x] * 32
      a32 = _mm256_add_epi16(a32, a16);     // a[x] * 32 + 16

      b = _mm256_mullo_epi16(diff, shift);
      res = _mm256_add_epi16(a32, b);
      resx = _mm256_srli_epi16(res, 5);  // 16 16-bit values

      // y calc
      resy = _mm256_setzero_si256();
      __m256i a0_y, a1_y, shifty;
      if ((base_x < min_base_x)) {
        __m256i c256, y_c256, base_y_c256, mask256, mul16;
        r6 = _mm256_set1_epi16(r << 6);
        c256 = _mm256_add_epi16(j256, c1234);
        __m256i c256_ = _mm256_add_epi16(c256, cmrlIdx);
        mul16 = _mm256_min_epu16(_mm256_mullo_epi16(c256_, dy256),
                                 _mm256_srli_epi16(min_base_y256, 1));
        y_c256 = _mm256_sub_epi16(r6, mul16);
        base_y_c256 = _mm256_srai_epi16(y_c256, frac_bits_y);
        mask256 = _mm256_cmpgt_epi16(min_base_y256, base_y_c256);
        base_y_c256 = _mm256_andnot_si256(mask256, base_y_c256);
        _mm256_store_si256((__m256i *)base_y_c, base_y_c256);

        a0_y = _mm256_setr_epi16(
            left[base_y_c[0]], left[base_y_c[1]], left[base_y_c[2]],
            left[base_y_c[3]], left[base_y_c[4]], left[base_y_c[5]],
            left[base_y_c[6]], left[base_y_c[7]], left[base_y_c[8]],
            left[base_y_c[9]], left[base_y_c[10]], left[base_y_c[11]],
            left[base_y_c[12]], left[base_y_c[13]], left[base_y_c[14]],
            left[base_y_c[15]]);
        base_y_c256 = _mm256_add_epi16(base_y_c256, c1);
        _mm256_store_si256((__m256i *)base_y_c, base_y_c256);

        a1_y = _mm256_setr_epi16(
            left[base_y_c[0]], left[base_y_c[1]], left[base_y_c[2]],
            left[base_y_c[3]], left[base_y_c[4]], left[base_y_c[5]],
            left[base_y_c[6]], left[base_y_c[7]], left[base_y_c[8]],
            left[base_y_c[9]], left[base_y_c[10]], left[base_y_c[11]],
            left[base_y_c[12]], left[base_y_c[13]], left[base_y_c[14]],
            left[base_y_c[15]]);

        shifty = _mm256_srli_epi16(_mm256_and_si256(y_c256, c3f), 1);

        diff = _mm256_sub_epi16(a1_y, a0_y);  // a[x+1] - a[x]
        a32 = _mm256_slli_epi16(a0_y, 5);     // a[x] * 32
        a32 = _mm256_add_epi16(a32, a16);     // a[x] * 32 + 16

        b = _mm256_mullo_epi16(diff, shifty);
        res = _mm256_add_epi16(a32, b);
        resy = _mm256_srli_epi16(res, 5);
      }

      resxy = _mm256_blendv_epi8(resx, resy,
                                 *(__m256i *)HighbdBaseMask[base_min_diff]);
      _mm256_storeu_si256((__m256i *)(dst + j), resxy);
    }  // for j
    dst += stride;
  }
}

// Directional prediction, zone 2: 90 < angle < 180
void av2_highbd_dr_prediction_z2_avx2(uint16_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint16_t *above,
                                      const uint16_t *left, int dx, int dy,
                                      int bd, int mrl_index) {
  (void)bd;
  assert(dx > 0);
  assert(dy > 0);
  switch (bw) {
    case 4:
      if (bd < 12) {
        highbd_dr_prediction_z2_Nx4_avx2(bh, dst, stride, above, left, dx, dy,
                                         mrl_index);
      } else {
        highbd_dr_prediction_32bit_z2_Nx4_avx2(bh, dst, stride, above, left, dx,
                                               dy, mrl_index);
      }
      break;
    case 8:
      if (bd < 12) {
        highbd_dr_prediction_z2_Nx8_avx2(bh, dst, stride, above, left, dx, dy,
                                         mrl_index);
      } else {
        highbd_dr_prediction_32bit_z2_Nx8_avx2(bh, dst, stride, above, left, dx,
                                               dy, mrl_index);
      }
      break;
    default:
      if (bd < 12) {
        highbd_dr_prediction_z2_HxW_avx2(bh, bw, dst, stride, above, left, dx,
                                         dy, mrl_index);
      } else {
        highbd_dr_prediction_32bit_z2_HxW_avx2(bh, bw, dst, stride, above, left,
                                               dx, dy, mrl_index);
      }
      break;
  }
}

//  Directional prediction, zone 3 functions
static void highbd_dr_prediction_z3_4x4_avx2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *left, int dy,
                                             int bd, int mrl_index) {
  __m128i dstvec[4], d[4];
  if (bd < 12) {
    highbd_dr_prediction_z1_4xN_internal_avx2(4, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_avx2(4, dstvec, left, dy,
                                                    mrl_index);
  }
  highbd_transpose4x8_8x4_low_sse2(&dstvec[0], &dstvec[1], &dstvec[2],
                                   &dstvec[3], &d[0], &d[1], &d[2], &d[3]);
  _mm_storel_epi64((__m128i *)(dst + 0 * stride), d[0]);
  _mm_storel_epi64((__m128i *)(dst + 1 * stride), d[1]);
  _mm_storel_epi64((__m128i *)(dst + 2 * stride), d[2]);
  _mm_storel_epi64((__m128i *)(dst + 3 * stride), d[3]);
  return;
}

static void highbd_dr_prediction_z3_8x8_avx2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *left, int dy,
                                             int bd, int mrl_index) {
  __m128i dstvec[8], d[8];
  if (bd < 12) {
    highbd_dr_prediction_z1_8xN_internal_avx2(8, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_avx2(8, dstvec, left, dy,
                                                    mrl_index);
  }
  highbd_transpose8x8_sse2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                           &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                           &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6],
                           &d[7]);
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_4x8_avx2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *left, int dy,
                                             int bd, int mrl_index) {
  __m128i dstvec[4], d[8];
  if (bd < 12) {
    highbd_dr_prediction_z1_8xN_internal_avx2(4, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_avx2(4, dstvec, left, dy,
                                                    mrl_index);
  }

  highbd_transpose4x8_8x4_sse2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                               &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6],
                               &d[7]);
  for (int i = 0; i < 8; i++) {
    _mm_storel_epi64((__m128i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_8x4_avx2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *left, int dy,
                                             int bd, int mrl_index) {
  __m128i dstvec[8], d[4];
  if (bd < 12) {
    highbd_dr_prediction_z1_4xN_internal_avx2(8, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_avx2(8, dstvec, left, dy,
                                                    mrl_index);
  }

  highbd_transpose8x8_low_sse2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                               &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                               &d[0], &d[1], &d[2], &d[3]);
  _mm_storeu_si128((__m128i *)(dst + 0 * stride), d[0]);
  _mm_storeu_si128((__m128i *)(dst + 1 * stride), d[1]);
  _mm_storeu_si128((__m128i *)(dst + 2 * stride), d[2]);
  _mm_storeu_si128((__m128i *)(dst + 3 * stride), d[3]);
}

static void highbd_dr_prediction_z3_8x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m256i dstvec[8], d[8];
  if (bd < 12) {
    highbd_dr_prediction_z1_16xN_internal_avx2(8, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_avx2(8, dstvec, left, dy,
                                                     mrl_index);
  }
  highbd_transpose8x16_16x8_avx2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                                 &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                                 &d[0], &d[1], &d[2], &d[3], &d[4], &d[5],
                                 &d[6], &d[7]);
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride),
                     _mm256_castsi256_si128(d[i]));
  }
  for (int i = 8; i < 16; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride),
                     _mm256_extracti128_si256(d[i - 8], 1));
  }
}

static void highbd_dr_prediction_z3_16x8_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m128i dstvec[16];
  __m256i dstvec256[8], d256[8];
  if (bd < 12) {
    highbd_dr_prediction_z1_8xN_internal_avx2(16, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_avx2(16, dstvec, left, dy,
                                                    mrl_index);
  }
  for (int i = 0; i < 8; ++i) {
    dstvec256[i] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 0]), dstvec[i + 8], 0x1);
  }

  highbd_transpose8x16_16x8_avx2(
      &dstvec256[0], &dstvec256[1], &dstvec256[2], &dstvec256[3], &dstvec256[4],
      &dstvec256[5], &dstvec256[6], &dstvec256[7], &d256[0], &d256[1], &d256[2],
      &d256[3], &d256[4], &d256[5], &d256[6], &d256[7]);

  for (int i = 0; i < 8; ++i) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d256[i]);
  }
}

static void highbd_dr_prediction_z3_4x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m256i dstvec[4], d[4], d1;
  if (bd < 12) {
    highbd_dr_prediction_z1_16xN_internal_avx2(4, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_avx2(4, dstvec, left, dy,
                                                     mrl_index);
  }
  highbd_transpose4x16_avx2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                            &d[0], &d[1], &d[2], &d[3]);
  for (int i = 0; i < 4; i++) {
    _mm_storel_epi64((__m128i *)(dst + i * stride),
                     _mm256_castsi256_si128(d[i]));
    d1 = _mm256_bsrli_epi128(d[i], 8);
    _mm_storel_epi64((__m128i *)(dst + (i + 4) * stride),
                     _mm256_castsi256_si128(d1));
    _mm_storel_epi64((__m128i *)(dst + (i + 8) * stride),
                     _mm256_extracti128_si256(d[i], 1));
    _mm_storel_epi64((__m128i *)(dst + (i + 12) * stride),
                     _mm256_extracti128_si256(d1, 1));
  }
}

static void highbd_dr_prediction_z3_16x4_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m128i dstvec[16];
  __m256i d[4];
  if (bd < 12) {
    highbd_dr_prediction_z1_4xN_internal_avx2(16, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_avx2(16, dstvec, left, dy,
                                                    mrl_index);
  }
  highbd_transpose16x4_8x8_avx2(dstvec, d);

  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_8x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m256i dstvec[16], d[16];

  if (bd < 12) {
    highbd_dr_prediction_z1_32xN_internal_avx2(8, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_avx2(8, dstvec, left, dy,
                                                     mrl_index);
  }

  for (int i = 0; i < 16; i += 8) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec[i], &dstvec[i + 1], &dstvec[i + 2], &dstvec[i + 3],
        &dstvec[i + 4], &dstvec[i + 5], &dstvec[i + 6], &dstvec[i + 7], &d[i],
        &d[i + 1], &d[i + 2], &d[i + 3], &d[i + 4], &d[i + 5], &d[i + 6],
        &d[i + 7]);
  }

  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride),
                     _mm256_castsi256_si128(d[i]));
  }
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + (i + 8) * stride),
                     _mm256_extracti128_si256(d[i], 1));
  }
  for (int i = 8; i < 16; i++) {
    _mm_storeu_si128((__m128i *)(dst + (i + 8) * stride),
                     _mm256_castsi256_si128(d[i]));
  }
  for (int i = 8; i < 16; i++) {
    _mm_storeu_si128((__m128i *)(dst + (i + 16) * stride),
                     _mm256_extracti128_si256(d[i], 1));
  }
}

static void highbd_dr_prediction_z3_32x8_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m128i dstvec[32], d[32];
  if (bd < 12) {
    highbd_dr_prediction_z1_8xN_internal_avx2(32, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_avx2(32, dstvec, left, dy,
                                                    mrl_index);
  }

  for (int i = 0; i < 32; i += 8) {
    highbd_transpose8x8_sse2(&dstvec[0 + i], &dstvec[1 + i], &dstvec[2 + i],
                             &dstvec[3 + i], &dstvec[4 + i], &dstvec[5 + i],
                             &dstvec[6 + i], &dstvec[7 + i], &d[0 + i],
                             &d[1 + i], &d[2 + i], &d[3 + i], &d[4 + i],
                             &d[5 + i], &d[6 + i], &d[7 + i]);
  }
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride), d[i]);
    _mm_storeu_si128((__m128i *)(dst + i * stride + 8), d[i + 8]);
    _mm_storeu_si128((__m128i *)(dst + i * stride + 16), d[i + 16]);
    _mm_storeu_si128((__m128i *)(dst + i * stride + 24), d[i + 24]);
  }
}

static void highbd_dr_prediction_z3_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  __m256i dstvec[16], d[16];
  if (bd < 12) {
    highbd_dr_prediction_z1_16xN_internal_avx2(16, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_avx2(16, dstvec, left, dy,
                                                     mrl_index);
  }

  highbd_transpose16x16_avx2(dstvec, d);

  for (int i = 0; i < 16; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  __m256i dstvec[64], d[16];
  if (bd < 12) {
    highbd_dr_prediction_z1_32xN_internal_avx2(32, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_avx2(32, dstvec, left, dy,
                                                     mrl_index);
  }
  highbd_transpose16x16_avx2(dstvec, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + j * stride), d[j]);
  }
  highbd_transpose16x16_avx2(dstvec + 16, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + j * stride + 16), d[j]);
  }
  highbd_transpose16x16_avx2(dstvec + 32, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + (j + 16) * stride), d[j]);
  }
  highbd_transpose16x16_avx2(dstvec + 48, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + (j + 16) * stride + 16), d[j]);
  }
}

static void highbd_dr_prediction_z3_64x64_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  DECLARE_ALIGNED(16, uint16_t, dstT[64 * 64]);
  if (bd < 12) {
    highbd_dr_prediction_z1_64xN_avx2(64, dstT, 64, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_avx2(64, dstT, 64, left, dy, mrl_index);
  }
  highbd_transpose(dstT, 64, dst, stride, 64, 64);
}

static void highbd_dr_prediction_z3_16x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  __m256i dstvec[32], d[32];
  if (bd < 12) {
    highbd_dr_prediction_z1_32xN_internal_avx2(16, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_avx2(16, dstvec, left, dy,
                                                     mrl_index);
  }
  for (int i = 0; i < 32; i += 8) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec[i], &dstvec[i + 1], &dstvec[i + 2], &dstvec[i + 3],
        &dstvec[i + 4], &dstvec[i + 5], &dstvec[i + 6], &dstvec[i + 7], &d[i],
        &d[i + 1], &d[i + 2], &d[i + 3], &d[i + 4], &d[i + 5], &d[i + 6],
        &d[i + 7]);
  }
  // store
  for (int j = 0; j < 32; j += 16) {
    for (int i = 0; i < 8; i++) {
      _mm_storeu_si128((__m128i *)(dst + (i + j) * stride),
                       _mm256_castsi256_si128(d[(i + j)]));
    }
    for (int i = 0; i < 8; i++) {
      _mm_storeu_si128((__m128i *)(dst + (i + j) * stride + 8),
                       _mm256_castsi256_si128(d[(i + j) + 8]));
    }
    for (int i = 8; i < 16; i++) {
      _mm256_storeu_si256(
          (__m256i *)(dst + (i + j) * stride),
          _mm256_inserti128_si256(
              d[(i + j)], _mm256_extracti128_si256(d[(i + j) - 8], 1), 0));
    }
  }
}

static void highbd_dr_prediction_z3_32x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  __m256i dstvec[32], d[16];
  if (bd < 12) {
    highbd_dr_prediction_z1_16xN_internal_avx2(32, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_avx2(32, dstvec, left, dy,
                                                     mrl_index);
  }
  for (int i = 0; i < 32; i += 16) {
    highbd_transpose16x16_avx2((dstvec + i), d);
    for (int j = 0; j < 16; j++) {
      _mm256_storeu_si256((__m256i *)(dst + j * stride + i), d[j]);
    }
  }
}

static void highbd_dr_prediction_z3_32x64_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  uint16_t dstT[64 * 32];
  if (bd < 12) {
    highbd_dr_prediction_z1_64xN_avx2(32, dstT, 64, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_avx2(32, dstT, 64, left, dy, mrl_index);
  }
  highbd_transpose(dstT, 64, dst, stride, 32, 64);
}

static void highbd_dr_prediction_z3_64x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  DECLARE_ALIGNED(16, uint16_t, dstT[32 * 64]);
  highbd_dr_prediction_z1_32xN_avx2(64, dstT, 32, left, dy, bd, mrl_index);
  highbd_transpose(dstT, 32, dst, stride, 64, 32);
  return;
}

static void highbd_dr_prediction_z3_16x64_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  DECLARE_ALIGNED(16, uint16_t, dstT[64 * 16]);
  if (bd < 12) {
    highbd_dr_prediction_z1_64xN_avx2(16, dstT, 64, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_avx2(16, dstT, 64, left, dy, mrl_index);
  }
  highbd_transpose(dstT, 64, dst, stride, 16, 64);
}

static void highbd_dr_prediction_z3_64x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *left, int dy,
                                               int bd, int mrl_index) {
  __m256i dstvec[64], d[16];
  if (bd < 12) {
    highbd_dr_prediction_z1_16xN_internal_avx2(64, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_avx2(64, dstvec, left, dy,
                                                     mrl_index);
  }
  for (int i = 0; i < 64; i += 16) {
    highbd_transpose16x16_avx2((dstvec + i), d);
    for (int j = 0; j < 16; j++) {
      _mm256_storeu_si256((__m256i *)(dst + j * stride + i), d[j]);
    }
  }
}

static void highbd_dr_prediction_z3_32x4_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m128i dstvec[32];
  __m256i d[8];
  if (bd < 12) {
    highbd_dr_prediction_z1_4xN_internal_avx2(32, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_avx2(32, dstvec, left, dy,
                                                    mrl_index);
  }
  highbd_transpose16x4_8x8_avx2(dstvec, d);
  highbd_transpose16x4_8x8_avx2(dstvec + 16, d + 4);

  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 16), d[i + 4]);
  }
}

static void highbd_dr_prediction_z3_64x4_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m128i dstvec[64];
  __m256i d[16];
  if (bd < 12) {
    highbd_dr_prediction_z1_4xN_internal_avx2(64, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_avx2(64, dstvec, left, dy,
                                                    mrl_index);
  }
  highbd_transpose16x4_8x8_avx2(dstvec, d);
  highbd_transpose16x4_8x8_avx2(dstvec + 16, d + 4);
  highbd_transpose16x4_8x8_avx2(dstvec + 32, d + 8);
  highbd_transpose16x4_8x8_avx2(dstvec + 48, d + 12);

  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 16), d[i + 4]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 32), d[i + 8]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 48), d[i + 12]);
  }
}

static void highbd_dr_prediction_z3_4x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m256i dstvec[8], d[8];
  if (bd < 12) {
    highbd_dr_prediction_z1_32xN_internal_avx2(4, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_avx2(4, dstvec, left, dy,
                                                     mrl_index);
  }

  highbd_transpose8x16_16x8_avx2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                                 &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                                 &d[0], &d[1], &d[2], &d[3], &d[4], &d[5],
                                 &d[6], &d[7]);

  for (int i = 0; i < 8; i++) {
    __m128i temp_lo = _mm256_castsi256_si128(d[i]);
    __m128i temp_hi = _mm256_extracti128_si256(d[i], 1);
    _mm_storel_epi64((__m128i *)(dst + i * stride), temp_lo);
    _mm_storel_epi64((__m128i *)(dst + (i + 16) * stride),
                     _mm_srli_si128(temp_lo, 8));
    _mm_storel_epi64((__m128i *)(dst + (i + 8) * stride), temp_hi);
    _mm_storel_epi64((__m128i *)(dst + (i + 24) * stride),
                     _mm_srli_si128(temp_hi, 8));
  }
}

static void highbd_dr_prediction_z3_64x8_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m128i dstvec[64];
  __m256i dstvec256[32], d256[32];
  if (bd < 12) {
    highbd_dr_prediction_z1_8xN_internal_avx2(64, dstvec, left, dy, mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_avx2(64, dstvec, left, dy,
                                                    mrl_index);
  }

  for (int i = 0; i < 8; ++i) {
    dstvec256[i] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 0]), dstvec[i + 8], 0x1);
    dstvec256[i + 8] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 16]), dstvec[i + 24], 0x1);
    dstvec256[i + 16] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 32]), dstvec[i + 40], 0x1);
    dstvec256[i + 24] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 48]), dstvec[i + 56], 0x1);
  }

  for (int i = 0; i < 32; i += 8) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec256[i + 0], &dstvec256[i + 1], &dstvec256[i + 2],
        &dstvec256[i + 3], &dstvec256[i + 4], &dstvec256[i + 5],
        &dstvec256[i + 6], &dstvec256[i + 7], &d256[i + 0], &d256[i + 1],
        &d256[i + 2], &d256[i + 3], &d256[i + 4], &d256[i + 5], &d256[i + 6],
        &d256[i + 7]);
  }
  for (int i = 0; i < 8; ++i) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d256[i]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 16), d256[i + 8]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 32), d256[i + 16]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 48), d256[i + 24]);
  }
}

static void highbd_dr_prediction_z3_4x64_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m256i dstvec[16], d[16];
  if (bd < 12) {
    highbd_dr_prediction_z1_64xN_avx2(4, (uint16_t *)dstvec, 64, left, dy,
                                      mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_avx2(4, (uint16_t *)dstvec, 64, left, dy,
                                            mrl_index);
  }

  // At this point, each 4 entries of dstvec[] forms one full 64-pixel
  // row (pre transposition) / col (post transposition)
  for (int i = 0; i < 4; i++) {
    highbd_transpose4x16_avx2(&dstvec[i], &dstvec[i + 4], &dstvec[i + 8],
                              &dstvec[i + 12], &d[4 * i], &d[4 * i + 1],
                              &d[4 * i + 2], &d[4 * i + 3]);
  }

  // Now each 4 elements of d contains 16 rows of 4 output pixels,
  // ordered like:
  // {0, 4, 8, 12}, {1, 5, 9, 13}, ...
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      const __m128i temph = _mm256_extracti128_si256(d[4 * i + j], 1);
      const __m128i templ = _mm256_castsi256_si128(d[4 * i + j]);
      xx_storel_64(dst + (16 * i + j) * stride, templ);
      xx_storel_64(dst + (16 * i + j + 4) * stride, _mm_srli_si128(templ, 8));
      xx_storel_64(dst + (16 * i + j + 8) * stride, temph);
      xx_storel_64(dst + (16 * i + j + 12) * stride, _mm_srli_si128(temph, 8));
    }
  }
}

static void highbd_dr_prediction_z3_8x64_avx2(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *left, int dy,
                                              int bd, int mrl_index) {
  __m256i dstvec[32], d[32];
  if (bd < 12) {
    highbd_dr_prediction_z1_64xN_avx2(8, (uint16_t *)dstvec, 64, left, dy,
                                      mrl_index);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_avx2(8, (uint16_t *)dstvec, 64, left, dy,
                                            mrl_index);
  }

  // At this point, each 4 entries of dstvec[] forms one full 64-pixel
  // row (pre transposition) / col (post transposition)
  for (int i = 0; i < 4; i++) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec[i], &dstvec[i + 4], &dstvec[i + 8], &dstvec[i + 12],
        &dstvec[i + 16], &dstvec[i + 20], &dstvec[i + 24], &dstvec[i + 28],
        &d[8 * i], &d[8 * i + 1], &d[8 * i + 2], &d[8 * i + 3], &d[8 * i + 4],
        &d[8 * i + 5], &d[8 * i + 6], &d[8 * i + 7]);
  }

  // Now each 8 elements of d contains 16 rows of 8 output pixels,
  // ordered like:
  // {0, 8}, {1, 9}, {2, 10}, ...
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      __m128i temp_lo = _mm256_castsi256_si128(d[8 * i + j]);
      __m128i temp_hi = _mm256_extracti128_si256(d[8 * i + j], 1);
      _mm_storeu_si128((__m128i *)(dst + (16 * i + j) * stride), temp_lo);
      _mm_storeu_si128((__m128i *)(dst + (16 * i + j + 8) * stride), temp_hi);
    }
  }
}

void av2_highbd_dr_prediction_z3_avx2(uint16_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint16_t *above,
                                      const uint16_t *left, int dx, int dy,
                                      int bd, int mrl_index) {
  (void)above;
  (void)dx;

  assert(dx == 1);
  assert(dy > 0);

  if (bw == bh) {
    switch (bw) {
      case 4:
        highbd_dr_prediction_z3_4x4_avx2(dst, stride, left, dy, bd, mrl_index);
        break;
      case 8:
        highbd_dr_prediction_z3_8x8_avx2(dst, stride, left, dy, bd, mrl_index);
        break;
      case 16:
        highbd_dr_prediction_z3_16x16_avx2(dst, stride, left, dy, bd,
                                           mrl_index);
        break;
      case 32:
        highbd_dr_prediction_z3_32x32_avx2(dst, stride, left, dy, bd,
                                           mrl_index);
        break;
      case 64:
        highbd_dr_prediction_z3_64x64_avx2(dst, stride, left, dy, bd,
                                           mrl_index);
        break;
    }
  } else {
    if (bw < bh) {
      if (bw + bw == bh) {
        switch (bw) {
          case 4:
            highbd_dr_prediction_z3_4x8_avx2(dst, stride, left, dy, bd,
                                             mrl_index);
            break;
          case 8:
            highbd_dr_prediction_z3_8x16_avx2(dst, stride, left, dy, bd,
                                              mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_16x32_avx2(dst, stride, left, dy, bd,
                                               mrl_index);
            break;
          case 32:
            highbd_dr_prediction_z3_32x64_avx2(dst, stride, left, dy, bd,
                                               mrl_index);
            break;
        }
      } else {
        switch (bw) {
          case 4:
            if (bh == 32)
              highbd_dr_prediction_z3_4x32_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            else if (bh == 64)
              highbd_dr_prediction_z3_4x64_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            else
              highbd_dr_prediction_z3_4x16_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            break;
          case 8:
            if (bh == 64)
              highbd_dr_prediction_z3_8x64_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            else
              highbd_dr_prediction_z3_8x32_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_16x64_avx2(dst, stride, left, dy, bd,
                                               mrl_index);
            break;
        }
      }
    } else {
      if (bh + bh == bw) {
        switch (bh) {
          case 4:
            highbd_dr_prediction_z3_8x4_avx2(dst, stride, left, dy, bd,
                                             mrl_index);
            break;
          case 8:
            highbd_dr_prediction_z3_16x8_avx2(dst, stride, left, dy, bd,
                                              mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_32x16_avx2(dst, stride, left, dy, bd,
                                               mrl_index);
            break;
          case 32:
            highbd_dr_prediction_z3_64x32_avx2(dst, stride, left, dy, bd,
                                               mrl_index);
            break;
        }
      } else {
        switch (bh) {
          case 4:
            if (bw == 64)
              highbd_dr_prediction_z3_64x4_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            else if (bw == 32)
              highbd_dr_prediction_z3_32x4_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            else
              highbd_dr_prediction_z3_16x4_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            break;
          case 8:
            if (bw == 64)
              highbd_dr_prediction_z3_64x8_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            else
              highbd_dr_prediction_z3_32x8_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_64x16_avx2(dst, stride, left, dy, bd,
                                               mrl_index);
            break;
        }
      }
    }
  }
  return;
}

static INLINE __m256i highbd_clamp_epi16_avx2(__m256i u, int bd) {
  const __m256i zero = _mm256_setzero_si256();
  const int max_i = ((1 << bd) - 1) << POWER_DR_INTERP_FILTER;
  const __m256i max = _mm256_set1_epi16(max_i);
  __m256i t, clamped;

  t = _mm256_max_epi16(u, zero);
  clamped = _mm256_min_epi16(t, max);

  return clamped;
}

static INLINE __m128i highbd_clamp_epi16_sse2(__m128i sum, int bd) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_scaled_pixel =
      _mm_set1_epi16((int16_t)(((1 << bd) - 1) << POWER_DR_INTERP_FILTER));
  return _mm_min_epi16(_mm_max_epi16(sum, zero), max_scaled_pixel);
}

static AVM_FORCE_INLINE void highbd_load_idif_taps_128(int filter_phase,
                                                       __m128i *taps_01,
                                                       __m128i *taps_23) {
  int32_t packed_taps_01, packed_taps_23;
  memcpy(&packed_taps_01, &av2_dr_interp_filter[filter_phase][0],
         sizeof(packed_taps_01));
  memcpy(&packed_taps_23, &av2_dr_interp_filter[filter_phase][2],
         sizeof(packed_taps_23));
  *taps_01 = _mm_set1_epi32(packed_taps_01);
  *taps_23 = _mm_set1_epi32(packed_taps_23);
}

static AVM_FORCE_INLINE void highbd_load_idif_taps_256(int filter_phase,
                                                       __m256i *taps_01,
                                                       __m256i *taps_23) {
  int32_t packed_taps_01, packed_taps_23;
  memcpy(&packed_taps_01, &av2_dr_interp_filter[filter_phase][0],
         sizeof(packed_taps_01));
  memcpy(&packed_taps_23, &av2_dr_interp_filter[filter_phase][2],
         sizeof(packed_taps_23));
  *taps_01 = _mm256_set1_epi32(packed_taps_01);
  *taps_23 = _mm256_set1_epi32(packed_taps_23);
}

static AVM_FORCE_INLINE __m128i highbd_convolve4_idif_16bit_sse2(
    __m128i ref_samples, int filter_phase, __m128i round_const, int bd) {
  const __m128i tap0 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][0]);
  const __m128i tap1 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][1]);
  const __m128i tap2 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][2]);
  const __m128i tap3 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][3]);
  const __m128i ref1 = _mm_bsrli_si128(ref_samples, 2);
  const __m128i ref2 = _mm_bsrli_si128(ref_samples, 4);
  const __m128i ref3 = _mm_bsrli_si128(ref_samples, 6);
  __m128i sum = _mm_adds_epi16(
      _mm_adds_epi16(_mm_mullo_epi16(ref_samples, tap0),
                     _mm_mullo_epi16(ref1, tap1)),
      _mm_adds_epi16(_mm_mullo_epi16(ref2, tap2), _mm_mullo_epi16(ref3, tap3)));
  sum = highbd_clamp_epi16_sse2(sum, bd);
  return _mm_srli_epi16(_mm_adds_epi16(sum, round_const),
                        POWER_DR_INTERP_FILTER);
}

static AVM_FORCE_INLINE __m128i highbd_convolve4_idif_32bit_sse41(
    __m128i ref_samples, __m128i taps_01, __m128i taps_23, __m128i round_const,
    __m128i max_pixel) {
  const __m128i ref_pair_01 =
      _mm_unpacklo_epi16(ref_samples, _mm_bsrli_si128(ref_samples, 2));
  const __m128i ref_pair_23 = _mm_unpacklo_epi16(
      _mm_bsrli_si128(ref_samples, 4), _mm_bsrli_si128(ref_samples, 6));
  __m128i sum = _mm_add_epi32(_mm_madd_epi16(ref_pair_01, taps_01),
                              _mm_madd_epi16(ref_pair_23, taps_23));
  sum = _mm_srai_epi32(_mm_add_epi32(sum, round_const), POWER_DR_INTERP_FILTER);
  return _mm_min_epu16(_mm_packus_epi32(sum, sum), max_pixel);
}

static AVM_FORCE_INLINE __m128i highbd_convolve8_idif_16bit_sse2(
    __m128i ref0, __m128i ref1, __m128i ref2, __m128i ref3, int filter_phase,
    __m128i round_const, int bd) {
  const __m128i tap0 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][0]);
  const __m128i tap1 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][1]);
  const __m128i tap2 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][2]);
  const __m128i tap3 = _mm_set1_epi16(av2_dr_interp_filter[filter_phase][3]);
  __m128i sum = _mm_adds_epi16(
      _mm_adds_epi16(_mm_mullo_epi16(ref0, tap0), _mm_mullo_epi16(ref1, tap1)),
      _mm_adds_epi16(_mm_mullo_epi16(ref2, tap2), _mm_mullo_epi16(ref3, tap3)));
  sum = highbd_clamp_epi16_sse2(sum, bd);
  return _mm_srli_epi16(_mm_adds_epi16(sum, round_const),
                        POWER_DR_INTERP_FILTER);
}

static AVM_FORCE_INLINE __m128i highbd_convolve8_idif_32bit_sse41(
    __m128i ref0, __m128i ref1, __m128i ref2, __m128i ref3, __m128i taps_01,
    __m128i taps_23, __m128i round_const, __m128i max_pixel) {
  __m128i sum_lo =
      _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(ref0, ref1), taps_01),
                    _mm_madd_epi16(_mm_unpacklo_epi16(ref2, ref3), taps_23));
  __m128i sum_hi =
      _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(ref0, ref1), taps_01),
                    _mm_madd_epi16(_mm_unpackhi_epi16(ref2, ref3), taps_23));
  sum_lo = _mm_srai_epi32(_mm_add_epi32(sum_lo, round_const),
                          POWER_DR_INTERP_FILTER);
  sum_hi = _mm_srai_epi32(_mm_add_epi32(sum_hi, round_const),
                          POWER_DR_INTERP_FILTER);
  return _mm_min_epu16(_mm_packus_epi32(sum_lo, sum_hi), max_pixel);
}

static AVM_FORCE_INLINE __m256i highbd_convolve16_idif_16bit_avx2(
    __m256i ref0, __m256i ref1, __m256i ref2, __m256i ref3, int filter_phase,
    __m256i round_const, int bd) {
  const __m256i tap0 = _mm256_set1_epi16(av2_dr_interp_filter[filter_phase][0]);
  const __m256i tap1 = _mm256_set1_epi16(av2_dr_interp_filter[filter_phase][1]);
  const __m256i tap2 = _mm256_set1_epi16(av2_dr_interp_filter[filter_phase][2]);
  const __m256i tap3 = _mm256_set1_epi16(av2_dr_interp_filter[filter_phase][3]);
  __m256i sum =
      _mm256_adds_epi16(_mm256_adds_epi16(_mm256_mullo_epi16(ref0, tap0),
                                          _mm256_mullo_epi16(ref1, tap1)),
                        _mm256_adds_epi16(_mm256_mullo_epi16(ref2, tap2),
                                          _mm256_mullo_epi16(ref3, tap3)));
  sum = highbd_clamp_epi16_avx2(sum, bd);
  return _mm256_srli_epi16(_mm256_adds_epi16(sum, round_const),
                           POWER_DR_INTERP_FILTER);
}

static AVM_FORCE_INLINE __m256i highbd_convolve16_idif_32bit_avx2(
    __m256i ref0, __m256i ref1, __m256i ref2, __m256i ref3, __m256i taps_01,
    __m256i taps_23, __m256i round_const, __m256i max_pixel) {
  __m256i sum_lo = _mm256_add_epi32(
      _mm256_madd_epi16(_mm256_unpacklo_epi16(ref0, ref1), taps_01),
      _mm256_madd_epi16(_mm256_unpacklo_epi16(ref2, ref3), taps_23));
  __m256i sum_hi = _mm256_add_epi32(
      _mm256_madd_epi16(_mm256_unpackhi_epi16(ref0, ref1), taps_01),
      _mm256_madd_epi16(_mm256_unpackhi_epi16(ref2, ref3), taps_23));
  sum_lo = _mm256_srai_epi32(_mm256_add_epi32(sum_lo, round_const),
                             POWER_DR_INTERP_FILTER);
  sum_hi = _mm256_srai_epi32(_mm256_add_epi32(sum_hi, round_const),
                             POWER_DR_INTERP_FILTER);
  return _mm256_min_epu16(_mm256_packus_epi32(sum_lo, sum_hi), max_pixel);
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_4xN_internal_idif_avx2(
    int num_rows, __m128i *dst_rows, const uint16_t *above, int dx,
    int mrl_index, int bd) {
  const int frac_bits = 6;
  const int max_base_x = (num_rows + 4) - 1 + (mrl_index << 1);
  assert(dx > 0);
  const __m128i round_const = _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i last_above_vec = _mm_set1_epi16((int16_t)above[max_base_x]);
  const __m128i max_base_plus1_vec = _mm_set1_epi16((int16_t)(max_base_x + 1));
  const __m128i lane_offsets = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
      }
      return;
    }
    const __m128i ref_samples =
        _mm_loadu_si128((const __m128i *)(above + base_x - 1));
    __m128i pred_row = highbd_convolve4_idif_16bit_sse2(
        ref_samples, (subpel_x & 0x3F) >> 1, round_const, bd);
    if (base_x > max_base_x - 3) {
      const __m128i col_base_idx =
          _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
      pred_row =
          _mm_blendv_epi8(last_above_vec, pred_row,
                          _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
    }
    dst_rows[row] = pred_row;
  }
}

static AVM_FORCE_INLINE void
highbd_dr_prediction_32bit_z1_4xN_internal_idif_avx2(int num_rows,
                                                     __m128i *dst_rows,
                                                     const uint16_t *above,
                                                     int dx, int mrl_index,
                                                     int bd) {
  const int frac_bits = 6;
  const int max_base_x = (num_rows + 4) - 1 + (mrl_index << 1);
  assert(dx > 0);
  const __m128i round_const = _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i max_pixel = _mm_set1_epi16((int16_t)((1 << bd) - 1));
  const __m128i last_above_vec = _mm_set1_epi16((int16_t)above[max_base_x]);
  const __m128i max_base_plus1_vec = _mm_set1_epi16((int16_t)(max_base_x + 1));
  const __m128i lane_offsets = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
      }
      return;
    }
    __m128i taps_01, taps_23;
    highbd_load_idif_taps_128((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
    const __m128i ref_samples =
        _mm_loadu_si128((const __m128i *)(above + base_x - 1));
    __m128i pred_row = highbd_convolve4_idif_32bit_sse41(
        ref_samples, taps_01, taps_23, round_const, max_pixel);
    if (base_x > max_base_x - 3) {
      const __m128i col_base_idx =
          _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
      pred_row =
          _mm_blendv_epi8(last_above_vec, pred_row,
                          _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
    }
    dst_rows[row] = pred_row;
  }
}

static void highbd_dr_prediction_z1_4xN_idif_avx2(
    uint16_t *dst, ptrdiff_t stride, int bw, int bh, const uint16_t *above,
    const uint16_t *left, int dx, int dy, int bd, int mrl_index) {
  (void)dy;
  (void)left;
  (void)bw;
  assert(bw == 4);
  assert(bh <= 64);
  const int frac_bits = 6;
  const int max_base_x = (bh + 4) - 1 + (mrl_index << 1);
  const __m128i last_above_vec = _mm_set1_epi16((int16_t)above[max_base_x]);
  const __m128i max_base_plus1_vec = _mm_set1_epi16((int16_t)(max_base_x + 1));
  const __m128i lane_offsets = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

  int subpel_x = dx * (1 + mrl_index);
  if (bd < 10) {
    const __m128i round_const =
        _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm_storel_epi64((__m128i *)dst, last_above_vec);
        }
        return;
      }
      const __m128i ref_samples =
          _mm_loadu_si128((const __m128i *)(above + base_x - 1));
      __m128i pred_row = highbd_convolve4_idif_16bit_sse2(
          ref_samples, (subpel_x & 0x3F) >> 1, round_const, bd);
      if (base_x > max_base_x - 3) {
        const __m128i col_base_idx =
            _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
        pred_row =
            _mm_blendv_epi8(last_above_vec, pred_row,
                            _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
      }
      _mm_storel_epi64((__m128i *)dst, pred_row);
    }
  } else {
    const __m128i round_const =
        _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
    const __m128i max_pixel = _mm_set1_epi16((int16_t)((1 << bd) - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm_storel_epi64((__m128i *)dst, last_above_vec);
        }
        return;
      }
      __m128i taps_01, taps_23;
      highbd_load_idif_taps_128((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
      const __m128i ref_samples =
          _mm_loadu_si128((const __m128i *)(above + base_x - 1));
      __m128i pred_row = highbd_convolve4_idif_32bit_sse41(
          ref_samples, taps_01, taps_23, round_const, max_pixel);
      if (base_x > max_base_x - 3) {
        const __m128i col_base_idx =
            _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
        pred_row =
            _mm_blendv_epi8(last_above_vec, pred_row,
                            _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
      }
      _mm_storel_epi64((__m128i *)dst, pred_row);
    }
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_8xN_internal_idif_avx2(
    int num_rows, __m128i *dst_rows, const uint16_t *above, int dx,
    int mrl_index, int bd) {
  const int frac_bits = 6;
  const int max_base_x = (num_rows + 8) - 1 + (mrl_index << 1);
  assert(dx > 0);
  const __m128i round_const = _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i last_above_vec = _mm_set1_epi16((int16_t)above[max_base_x]);
  const __m128i max_base_plus1_vec = _mm_set1_epi16((int16_t)(max_base_x + 1));
  const __m128i lane_offsets = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
      }
      return;
    }
    const __m128i ref0 = _mm_loadu_si128((const __m128i *)(above + base_x - 1));
    const __m128i ref1 = _mm_loadu_si128((const __m128i *)(above + base_x));
    const __m128i ref2 = _mm_loadu_si128((const __m128i *)(above + base_x + 1));
    const __m128i ref3 = _mm_loadu_si128((const __m128i *)(above + base_x + 2));
    __m128i pred_row = highbd_convolve8_idif_16bit_sse2(
        ref0, ref1, ref2, ref3, (subpel_x & 0x3F) >> 1, round_const, bd);
    if (base_x > max_base_x - 7) {
      const __m128i col_base_idx =
          _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
      pred_row =
          _mm_blendv_epi8(last_above_vec, pred_row,
                          _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
    }
    dst_rows[row] = pred_row;
  }
}

static AVM_FORCE_INLINE void
highbd_dr_prediction_32bit_z1_8xN_internal_idif_avx2(int num_rows,
                                                     __m128i *dst_rows,
                                                     const uint16_t *above,
                                                     int dx, int mrl_index,
                                                     int bd) {
  const int frac_bits = 6;
  const int max_base_x = (num_rows + 8) - 1 + (mrl_index << 1);
  assert(dx > 0);
  const __m128i round_const = _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i max_pixel = _mm_set1_epi16((int16_t)((1 << bd) - 1));
  const __m128i last_above_vec = _mm_set1_epi16((int16_t)above[max_base_x]);
  const __m128i max_base_plus1_vec = _mm_set1_epi16((int16_t)(max_base_x + 1));
  const __m128i lane_offsets = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
      }
      return;
    }
    __m128i taps_01, taps_23;
    highbd_load_idif_taps_128((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
    const __m128i ref0 = _mm_loadu_si128((const __m128i *)(above + base_x - 1));
    const __m128i ref1 = _mm_loadu_si128((const __m128i *)(above + base_x));
    const __m128i ref2 = _mm_loadu_si128((const __m128i *)(above + base_x + 1));
    const __m128i ref3 = _mm_loadu_si128((const __m128i *)(above + base_x + 2));
    __m128i pred_row = highbd_convolve8_idif_32bit_sse41(
        ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
    if (base_x > max_base_x - 7) {
      const __m128i col_base_idx =
          _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
      pred_row =
          _mm_blendv_epi8(last_above_vec, pred_row,
                          _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
    }
    dst_rows[row] = pred_row;
  }
}

static void highbd_dr_prediction_z1_8xN_idif_avx2(
    uint16_t *dst, ptrdiff_t stride, int bw, int bh, const uint16_t *above,
    const uint16_t *left, int dx, int dy, int bd, int mrl_index) {
  (void)left;
  (void)dy;
  (void)bw;
  assert(bw == 8);
  assert(bh <= 64);
  const int frac_bits = 6;
  const int max_base_x = (bh + 8) - 1 + (mrl_index << 1);
  const __m128i last_above_vec = _mm_set1_epi16((int16_t)above[max_base_x]);
  const __m128i max_base_plus1_vec = _mm_set1_epi16((int16_t)(max_base_x + 1));
  const __m128i lane_offsets = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

  int subpel_x = dx * (1 + mrl_index);
  if (bd < 10) {
    const __m128i round_const =
        _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm_storeu_si128((__m128i *)dst, last_above_vec);
        }
        return;
      }
      const __m128i ref0 =
          _mm_loadu_si128((const __m128i *)(above + base_x - 1));
      const __m128i ref1 = _mm_loadu_si128((const __m128i *)(above + base_x));
      const __m128i ref2 =
          _mm_loadu_si128((const __m128i *)(above + base_x + 1));
      const __m128i ref3 =
          _mm_loadu_si128((const __m128i *)(above + base_x + 2));
      __m128i pred_row = highbd_convolve8_idif_16bit_sse2(
          ref0, ref1, ref2, ref3, (subpel_x & 0x3F) >> 1, round_const, bd);
      if (base_x > max_base_x - 7) {
        const __m128i col_base_idx =
            _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
        pred_row =
            _mm_blendv_epi8(last_above_vec, pred_row,
                            _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
      }
      _mm_storeu_si128((__m128i *)dst, pred_row);
    }
  } else {
    const __m128i round_const =
        _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
    const __m128i max_pixel = _mm_set1_epi16((int16_t)((1 << bd) - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm_storeu_si128((__m128i *)dst, last_above_vec);
        }
        return;
      }
      __m128i taps_01, taps_23;
      highbd_load_idif_taps_128((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
      const __m128i ref0 =
          _mm_loadu_si128((const __m128i *)(above + base_x - 1));
      const __m128i ref1 = _mm_loadu_si128((const __m128i *)(above + base_x));
      const __m128i ref2 =
          _mm_loadu_si128((const __m128i *)(above + base_x + 1));
      const __m128i ref3 =
          _mm_loadu_si128((const __m128i *)(above + base_x + 2));
      __m128i pred_row = highbd_convolve8_idif_32bit_sse41(
          ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
      if (base_x > max_base_x - 7) {
        const __m128i col_base_idx =
            _mm_add_epi16(_mm_set1_epi16((int16_t)base_x), lane_offsets);
        pred_row =
            _mm_blendv_epi8(last_above_vec, pred_row,
                            _mm_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
      }
      _mm_storeu_si128((__m128i *)dst, pred_row);
    }
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_16xN_internal_idif_avx2(
    int num_rows, __m256i *dst_rows, const uint16_t *above, int dx,
    int mrl_index, int bd) {
  const int frac_bits = 6;
  const int max_base_x = (16 + num_rows) - 1 + (mrl_index << 1);
  const __m256i round_const =
      _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
      }
      return;
    }
    const __m256i ref0 =
        _mm256_loadu_si256((const __m256i *)(above + base_x - 1));
    const __m256i ref1 = _mm256_loadu_si256((const __m256i *)(above + base_x));
    const __m256i ref2 =
        _mm256_loadu_si256((const __m256i *)(above + base_x + 1));
    const __m256i ref3 =
        _mm256_loadu_si256((const __m256i *)(above + base_x + 2));
    __m256i pred_row = highbd_convolve16_idif_16bit_avx2(
        ref0, ref1, ref2, ref3, (subpel_x & 0x3F) >> 1, round_const, bd);
    if (base_x > max_base_x - 15) {
      const __m256i col_base_idx =
          _mm256_add_epi16(_mm256_set1_epi16((int16_t)base_x), lane_offsets);
      pred_row = _mm256_blendv_epi8(
          last_above_vec, pred_row,
          _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
    }
    dst_rows[row] = pred_row;
  }
}

static AVM_FORCE_INLINE void
highbd_dr_prediction_32bit_z1_16xN_internal_idif_avx2(int num_rows,
                                                      __m256i *dst_rows,
                                                      const uint16_t *above,
                                                      int dx, int mrl_index,
                                                      int bd) {
  const int frac_bits = 6;
  const int max_base_x = (16 + num_rows) - 1 + (mrl_index << 1);
  const __m256i round_const =
      _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i max_pixel = _mm256_set1_epi16((int16_t)((1 << bd) - 1));
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
      }
      return;
    }
    __m256i taps_01, taps_23;
    highbd_load_idif_taps_256((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
    const __m256i ref0 =
        _mm256_loadu_si256((const __m256i *)(above + base_x - 1));
    const __m256i ref1 = _mm256_loadu_si256((const __m256i *)(above + base_x));
    const __m256i ref2 =
        _mm256_loadu_si256((const __m256i *)(above + base_x + 1));
    const __m256i ref3 =
        _mm256_loadu_si256((const __m256i *)(above + base_x + 2));
    __m256i pred_row = highbd_convolve16_idif_32bit_avx2(
        ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
    if (base_x > max_base_x - 15) {
      const __m256i col_base_idx =
          _mm256_add_epi16(_mm256_set1_epi16((int16_t)base_x), lane_offsets);
      pred_row = _mm256_blendv_epi8(
          last_above_vec, pred_row,
          _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
    }
    dst_rows[row] = pred_row;
  }
}

static void highbd_dr_prediction_z1_16xN_idif_avx2(
    uint16_t *dst, ptrdiff_t stride, int bw, int bh, const uint16_t *above,
    const uint16_t *left, int dx, int dy, int bd, int mrl_index) {
  (void)left;
  (void)dy;
  (void)bw;
  assert(bw == 16);
  const int frac_bits = 6;
  const int max_base_x = (16 + bh) - 1 + (mrl_index << 1);
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  if (bd < 10) {
    const __m256i round_const =
        _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm256_storeu_si256((__m256i *)dst, last_above_vec);
        }
        return;
      }
      const __m256i ref0 =
          _mm256_loadu_si256((const __m256i *)(above + base_x - 1));
      const __m256i ref1 =
          _mm256_loadu_si256((const __m256i *)(above + base_x));
      const __m256i ref2 =
          _mm256_loadu_si256((const __m256i *)(above + base_x + 1));
      const __m256i ref3 =
          _mm256_loadu_si256((const __m256i *)(above + base_x + 2));
      __m256i pred_row = highbd_convolve16_idif_16bit_avx2(
          ref0, ref1, ref2, ref3, (subpel_x & 0x3F) >> 1, round_const, bd);
      if (base_x > max_base_x - 15) {
        const __m256i col_base_idx =
            _mm256_add_epi16(_mm256_set1_epi16((int16_t)base_x), lane_offsets);
        pred_row = _mm256_blendv_epi8(
            last_above_vec, pred_row,
            _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
      }
      _mm256_storeu_si256((__m256i *)dst, pred_row);
    }
  } else {
    const __m256i round_const =
        _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
    const __m256i max_pixel = _mm256_set1_epi16((int16_t)((1 << bd) - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm256_storeu_si256((__m256i *)dst, last_above_vec);
        }
        return;
      }
      __m256i taps_01, taps_23;
      highbd_load_idif_taps_256((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
      const __m256i ref0 =
          _mm256_loadu_si256((const __m256i *)(above + base_x - 1));
      const __m256i ref1 =
          _mm256_loadu_si256((const __m256i *)(above + base_x));
      const __m256i ref2 =
          _mm256_loadu_si256((const __m256i *)(above + base_x + 1));
      const __m256i ref3 =
          _mm256_loadu_si256((const __m256i *)(above + base_x + 2));
      __m256i pred_row = highbd_convolve16_idif_32bit_avx2(
          ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
      if (base_x > max_base_x - 15) {
        const __m256i col_base_idx =
            _mm256_add_epi16(_mm256_set1_epi16((int16_t)base_x), lane_offsets);
        pred_row = _mm256_blendv_epi8(
            last_above_vec, pred_row,
            _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
      }
      _mm256_storeu_si256((__m256i *)dst, pred_row);
    }
  }
}

static AVM_FORCE_INLINE void highbd_dr_prediction_z1_32xN_internal_idif_avx2(
    int num_rows, __m256i *dst_rows, const uint16_t *above, int dx,
    int mrl_index, int bd) {
  const int frac_bits = 6;
  const int max_base_x = (32 + num_rows) - 1 + (mrl_index << 1);
  const __m256i round_const =
      _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
        dst_rows[fill_row + num_rows] = last_above_vec;
      }
      return;
    }

    const int filter_phase = (subpel_x & 0x3F) >> 1;
    for (int col_offset = 0; col_offset < 32; col_offset += 16) {
      const int chunk_base_x = base_x + col_offset;
      __m256i pred_chunk;
      if (chunk_base_x > max_base_x) {
        pred_chunk = last_above_vec;
      } else {
        const __m256i ref0 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x - 1));
        const __m256i ref1 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x));
        const __m256i ref2 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 1));
        const __m256i ref3 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 2));
        pred_chunk = highbd_convolve16_idif_16bit_avx2(
            ref0, ref1, ref2, ref3, filter_phase, round_const, bd);
        if (chunk_base_x > max_base_x - 15) {
          const __m256i col_base_idx = _mm256_add_epi16(
              _mm256_set1_epi16((int16_t)chunk_base_x), lane_offsets);
          pred_chunk = _mm256_blendv_epi8(
              last_above_vec, pred_chunk,
              _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
        }
      }
      dst_rows[col_offset ? (row + num_rows) : row] = pred_chunk;
    }
  }
}

static AVM_FORCE_INLINE void
highbd_dr_prediction_32bit_z1_32xN_internal_idif_avx2(int num_rows,
                                                      __m256i *dst_rows,
                                                      const uint16_t *above,
                                                      int dx, int mrl_index,
                                                      int bd) {
  const int frac_bits = 6;
  const int max_base_x = (32 + num_rows) - 1 + (mrl_index << 1);
  const __m256i round_const =
      _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i max_pixel = _mm256_set1_epi16((int16_t)((1 << bd) - 1));
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row) {
        dst_rows[fill_row] = last_above_vec;
        dst_rows[fill_row + num_rows] = last_above_vec;
      }
      return;
    }

    __m256i taps_01, taps_23;
    highbd_load_idif_taps_256((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
    for (int col_offset = 0; col_offset < 32; col_offset += 16) {
      const int chunk_base_x = base_x + col_offset;
      __m256i pred_chunk;
      if (chunk_base_x > max_base_x) {
        pred_chunk = last_above_vec;
      } else {
        const __m256i ref0 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x - 1));
        const __m256i ref1 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x));
        const __m256i ref2 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 1));
        const __m256i ref3 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 2));
        pred_chunk = highbd_convolve16_idif_32bit_avx2(
            ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
        if (chunk_base_x > max_base_x - 15) {
          const __m256i col_base_idx = _mm256_add_epi16(
              _mm256_set1_epi16((int16_t)chunk_base_x), lane_offsets);
          pred_chunk = _mm256_blendv_epi8(
              last_above_vec, pred_chunk,
              _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
        }
      }
      dst_rows[col_offset ? (row + num_rows) : row] = pred_chunk;
    }
  }
}

static void highbd_dr_prediction_z1_32xN_idif_avx2(int bh, uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *above,
                                                   int dx, int bd,
                                                   int mrl_index) {
  const int frac_bits = 6;
  const int max_base_x = (32 + bh) - 1 + (mrl_index << 1);
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  int subpel_x = dx * (1 + mrl_index);

  if (bd < 10) {
    const __m256i round_const =
        _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm256_storeu_si256((__m256i *)dst, last_above_vec);
          _mm256_storeu_si256((__m256i *)(dst + 16), last_above_vec);
        }
        return;
      }
      const int filter_phase = (subpel_x & 0x3F) >> 1;
      for (int col_offset = 0; col_offset < 32; col_offset += 16) {
        const int chunk_base_x = base_x + col_offset;
        __m256i pred_chunk;
        if (chunk_base_x > max_base_x) {
          pred_chunk = last_above_vec;
        } else {
          const __m256i ref0 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x - 1));
          const __m256i ref1 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x));
          const __m256i ref2 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 1));
          const __m256i ref3 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 2));
          pred_chunk = highbd_convolve16_idif_16bit_avx2(
              ref0, ref1, ref2, ref3, filter_phase, round_const, bd);
          if (chunk_base_x > max_base_x - 15) {
            const __m256i col_base_idx = _mm256_add_epi16(
                _mm256_set1_epi16((int16_t)chunk_base_x), lane_offsets);
            pred_chunk = _mm256_blendv_epi8(
                last_above_vec, pred_chunk,
                _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
          }
        }
        _mm256_storeu_si256((__m256i *)(dst + col_offset), pred_chunk);
      }
    }
  } else {
    const __m256i round_const =
        _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
    const __m256i max_pixel = _mm256_set1_epi16((int16_t)((1 << bd) - 1));
    for (int row = 0; row < bh; ++row, dst += stride, subpel_x += dx) {
      const int base_x = subpel_x >> frac_bits;
      if (base_x > max_base_x) {
        for (int fill_row = row; fill_row < bh; ++fill_row, dst += stride) {
          _mm256_storeu_si256((__m256i *)dst, last_above_vec);
          _mm256_storeu_si256((__m256i *)(dst + 16), last_above_vec);
        }
        return;
      }
      __m256i taps_01, taps_23;
      highbd_load_idif_taps_256((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
      for (int col_offset = 0; col_offset < 32; col_offset += 16) {
        const int chunk_base_x = base_x + col_offset;
        __m256i pred_chunk;
        if (chunk_base_x > max_base_x) {
          pred_chunk = last_above_vec;
        } else {
          const __m256i ref0 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x - 1));
          const __m256i ref1 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x));
          const __m256i ref2 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 1));
          const __m256i ref3 =
              _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 2));
          pred_chunk = highbd_convolve16_idif_32bit_avx2(
              ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
          if (chunk_base_x > max_base_x - 15) {
            const __m256i col_base_idx = _mm256_add_epi16(
                _mm256_set1_epi16((int16_t)chunk_base_x), lane_offsets);
            pred_chunk = _mm256_blendv_epi8(
                last_above_vec, pred_chunk,
                _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
          }
        }
        _mm256_storeu_si256((__m256i *)(dst + col_offset), pred_chunk);
      }
    }
  }
}

static void highbd_dr_prediction_z1_64xN_internal_idif_avx2(
    int num_rows, uint16_t *dst, ptrdiff_t stride, const uint16_t *above,
    int dx, int mrl_index, int bd) {
  const int frac_bits = 6;
  const int max_base_x = (64 + num_rows) - 1 + (mrl_index << 1);
  const __m256i round_const =
      _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, dst += stride, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row, dst += stride) {
        _mm256_storeu_si256((__m256i *)dst, last_above_vec);
        _mm256_storeu_si256((__m256i *)(dst + 16), last_above_vec);
        _mm256_storeu_si256((__m256i *)(dst + 32), last_above_vec);
        _mm256_storeu_si256((__m256i *)(dst + 48), last_above_vec);
      }
      return;
    }

    const int filter_phase = (subpel_x & 0x3F) >> 1;
    for (int col_offset = 0; col_offset < 64; col_offset += 16) {
      const int chunk_base_x = base_x + col_offset;
      if (chunk_base_x > max_base_x) {
        _mm256_storeu_si256((__m256i *)(dst + col_offset), last_above_vec);
      } else {
        const __m256i ref0 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x - 1));
        const __m256i ref1 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x));
        const __m256i ref2 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 1));
        const __m256i ref3 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 2));
        __m256i pred_chunk = highbd_convolve16_idif_16bit_avx2(
            ref0, ref1, ref2, ref3, filter_phase, round_const, bd);
        if (chunk_base_x > max_base_x - 15) {
          const __m256i col_base_idx = _mm256_add_epi16(
              _mm256_set1_epi16((int16_t)chunk_base_x), lane_offsets);
          pred_chunk = _mm256_blendv_epi8(
              last_above_vec, pred_chunk,
              _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
        }
        _mm256_storeu_si256((__m256i *)(dst + col_offset), pred_chunk);
      }
    }
  }
}

static void highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(
    int num_rows, uint16_t *dst, ptrdiff_t stride, const uint16_t *above,
    int dx, int mrl_index, int bd) {
  const int frac_bits = 6;
  const int max_base_x = (64 + num_rows) - 1 + (mrl_index << 1);
  const __m256i round_const =
      _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i max_pixel = _mm256_set1_epi16((int16_t)((1 << bd) - 1));
  const __m256i last_above_vec = _mm256_set1_epi16((int16_t)above[max_base_x]);
  const __m256i max_base_plus1_vec =
      _mm256_set1_epi16((int16_t)(max_base_x + 1));
  const __m256i lane_offsets =
      _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  int subpel_x = dx * (1 + mrl_index);
  for (int row = 0; row < num_rows; ++row, dst += stride, subpel_x += dx) {
    const int base_x = subpel_x >> frac_bits;
    if (base_x > max_base_x) {
      for (int fill_row = row; fill_row < num_rows; ++fill_row, dst += stride) {
        _mm256_storeu_si256((__m256i *)dst, last_above_vec);
        _mm256_storeu_si256((__m256i *)(dst + 16), last_above_vec);
        _mm256_storeu_si256((__m256i *)(dst + 32), last_above_vec);
        _mm256_storeu_si256((__m256i *)(dst + 48), last_above_vec);
      }
      return;
    }

    __m256i taps_01, taps_23;
    highbd_load_idif_taps_256((subpel_x & 0x3F) >> 1, &taps_01, &taps_23);
    for (int col_offset = 0; col_offset < 64; col_offset += 16) {
      const int chunk_base_x = base_x + col_offset;
      if (chunk_base_x > max_base_x) {
        _mm256_storeu_si256((__m256i *)(dst + col_offset), last_above_vec);
      } else {
        const __m256i ref0 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x - 1));
        const __m256i ref1 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x));
        const __m256i ref2 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 1));
        const __m256i ref3 =
            _mm256_loadu_si256((const __m256i *)(above + chunk_base_x + 2));
        __m256i pred_chunk = highbd_convolve16_idif_32bit_avx2(
            ref0, ref1, ref2, ref3, taps_01, taps_23, round_const, max_pixel);
        if (chunk_base_x > max_base_x - 15) {
          const __m256i col_base_idx = _mm256_add_epi16(
              _mm256_set1_epi16((int16_t)chunk_base_x), lane_offsets);
          pred_chunk = _mm256_blendv_epi8(
              last_above_vec, pred_chunk,
              _mm256_cmpgt_epi16(max_base_plus1_vec, col_base_idx));
        }
        _mm256_storeu_si256((__m256i *)(dst + col_offset), pred_chunk);
      }
    }
  }
}

static void highbd_dr_prediction_z1_64xN_idif_avx2(
    uint16_t *dst, ptrdiff_t stride, int bw, int bh, const uint16_t *above,
    const uint16_t *left, int dx, int dy, int bd, int mrl_index) {
  (void)left;
  (void)dy;
  (void)bw;
  assert(bw == 64);
  if (bd < 10) {
    highbd_dr_prediction_z1_64xN_internal_idif_avx2(bh, dst, stride, above, dx,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(
        bh, dst, stride, above, dx, mrl_index, bd);
  }
}

void av2_highbd_dr_prediction_z1_idif_avx2(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int dx, int dy,
                                           int bd, int mrl_index) {
  switch (bw) {
    case 4:
      highbd_dr_prediction_z1_4xN_idif_avx2(dst, stride, bw, bh, above, left,
                                            dx, dy, bd, mrl_index);
      break;
    case 8:
      highbd_dr_prediction_z1_8xN_idif_avx2(dst, stride, bw, bh, above, left,
                                            dx, dy, bd, mrl_index);
      break;
    case 16:
      highbd_dr_prediction_z1_16xN_idif_avx2(dst, stride, bw, bh, above, left,
                                             dx, dy, bd, mrl_index);
      break;
    case 32:
      highbd_dr_prediction_z1_32xN_idif_avx2(bh, dst, stride, above, dx, bd,
                                             mrl_index);
      break;
    case 64:
      highbd_dr_prediction_z1_64xN_idif_avx2(dst, stride, bw, bh, above, left,
                                             dx, dy, bd, mrl_index);
      break;
    default: break;
  }
  return;
}

static AVM_FORCE_INLINE __m128i highbd_dr_z2_row8_16bit_idif_sse2(
    const uint16_t *edge, int filter_phase, int base_idx, int clamp_count,
    __m128i round_const, int bd) {
  const uint16_t *edge_ptr = edge + base_idx + clamp_count;
  __m128i ref0 = _mm_loadu_si128((const __m128i *)(edge_ptr - 1));
  __m128i ref1 = _mm_loadu_si128((const __m128i *)edge_ptr);
  __m128i ref2 = _mm_loadu_si128((const __m128i *)(edge_ptr + 1));
  __m128i ref3 = _mm_loadu_si128((const __m128i *)(edge_ptr + 2));
  if (clamp_count != 0) {
    const __m128i clamp_shuf_mask =
        _mm_load_si128((const __m128i *)HighbdLoadMaskx[clamp_count]);
    ref0 = _mm_shuffle_epi8(ref0, clamp_shuf_mask);
    ref1 = _mm_shuffle_epi8(ref1, clamp_shuf_mask);
    ref2 = _mm_shuffle_epi8(ref2, clamp_shuf_mask);
    ref3 = _mm_shuffle_epi8(ref3, clamp_shuf_mask);
  }
  return highbd_convolve8_idif_16bit_sse2(ref0, ref1, ref2, ref3, filter_phase,
                                          round_const, bd);
}

static AVM_FORCE_INLINE __m128i highbd_dr_z2_row8_32bit_idif_sse41(
    const uint16_t *edge, __m128i taps_01, __m128i taps_23, int base_idx,
    int clamp_count, __m128i round_const, __m128i max_pixel) {
  const uint16_t *edge_ptr = edge + base_idx + clamp_count;
  __m128i ref0 = _mm_loadu_si128((const __m128i *)(edge_ptr - 1));
  __m128i ref1 = _mm_loadu_si128((const __m128i *)edge_ptr);
  __m128i ref2 = _mm_loadu_si128((const __m128i *)(edge_ptr + 1));
  __m128i ref3 = _mm_loadu_si128((const __m128i *)(edge_ptr + 2));
  if (clamp_count != 0) {
    const __m128i clamp_shuf_mask =
        _mm_load_si128((const __m128i *)HighbdLoadMaskx[clamp_count]);
    ref0 = _mm_shuffle_epi8(ref0, clamp_shuf_mask);
    ref1 = _mm_shuffle_epi8(ref1, clamp_shuf_mask);
    ref2 = _mm_shuffle_epi8(ref2, clamp_shuf_mask);
    ref3 = _mm_shuffle_epi8(ref3, clamp_shuf_mask);
  }
  return highbd_convolve8_idif_32bit_sse41(ref0, ref1, ref2, ref3, taps_01,
                                           taps_23, round_const, max_pixel);
}

static INLINE void highbd_dr_z2_8x8_idif_avx2(int num_rows, int num_cols,
                                              const uint16_t *edge,
                                              __m128i *tile_rows, int start_row,
                                              int start_col, int ray_step,
                                              int mrl_index, int bd) {
  const int min_base_idx = -(1 + mrl_index);
  const int frac_bits = 6;
  const int tile_h = (num_rows == 4) ? 4 : 8;
  const int tile_w = (num_cols == 4) ? 4 : 8;
  const __m128i round_const = _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));

  for (int row = start_row; row < start_row + tile_h; ++row) {
    assert(row < num_rows);
    assert(start_col < num_cols);
    const int row_offset = row + 1;
    const int base_idx =
        ((start_col << 6) - (row_offset + mrl_index) * ray_step) >> frac_bits;
    const int clamp_count =
        (base_idx < min_base_idx - 1) ? (min_base_idx - base_idx - 1) : 0;

    if (clamp_count >= tile_w) {
      tile_rows[row - start_row] = _mm_setzero_si128();
    } else {
      const int filter_phase =
          ((-(row_offset + mrl_index) * ray_step) & 0x3F) >> 1;
      tile_rows[row - start_row] = highbd_dr_z2_row8_16bit_idif_sse2(
          edge, filter_phase, base_idx, clamp_count, round_const, bd);
    }
  }
}

static INLINE void highbd_dr_z2_32bit_8x8_idif_avx2(
    int num_rows, int num_cols, const uint16_t *edge, __m128i *tile_rows,
    int start_row, int start_col, int ray_step, int mrl_index, int bd) {
  const int min_base_idx = -(1 + mrl_index);
  const int frac_bits = 6;
  const int tile_h = (num_rows == 4) ? 4 : 8;
  const int tile_w = (num_cols == 4) ? 4 : 8;
  const __m128i round_const = _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i max_pixel = _mm_set1_epi16((int16_t)((1 << bd) - 1));

  for (int row = start_row; row < start_row + tile_h; ++row) {
    assert(row < num_rows);
    assert(start_col < num_cols);
    const int row_offset = row + 1;
    const int base_idx =
        ((start_col << 6) - (row_offset + mrl_index) * ray_step) >> frac_bits;
    const int clamp_count =
        (base_idx < min_base_idx - 1) ? (min_base_idx - base_idx - 1) : 0;

    if (clamp_count >= tile_w) {
      tile_rows[row - start_row] = _mm_setzero_si128();
    } else {
      __m128i taps_01, taps_23;
      highbd_load_idif_taps_128(
          ((-(row_offset + mrl_index) * ray_step) & 0x3F) >> 1, &taps_01,
          &taps_23);
      tile_rows[row - start_row] = highbd_dr_z2_row8_32bit_idif_sse41(
          edge, taps_01, taps_23, base_idx, clamp_count, round_const,
          max_pixel);
    }
  }
}

static void highbd_dr_z2_8x8_tiling_idif_avx2(int bh, int bw, uint16_t *dst,
                                              ptrdiff_t stride,
                                              const uint16_t *above,
                                              const uint16_t *left, int dx,
                                              int dy, int mrl_index, int bd) {
  const int min_base_x = -(1 + mrl_index);
  const int frac_bits = 6;
  const int tile_h = (bh == 4) ? 4 : 8;
  const int tile_w = (bw == 4) ? 4 : 8;
  const __m128i round_16bit = _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i round_32bit = _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i max_pixel = _mm_set1_epi16((int16_t)((1 << bd) - 1));

  __m128i above_pred[8];
  __m128i left_pred[8];
  __m128i left_pred_cols[8];
  if (tile_w == 4) {
    for (int col = 4; col < 8; ++col) left_pred_cols[col] = _mm_setzero_si128();
  }

  for (int tile_row = 0; tile_row < bh; tile_row += 8) {
    for (int tile_col = 0; tile_col < bw; tile_col += tile_w) {
      assert((bw - tile_col) >= tile_w);
      assert((bh - tile_row) >= tile_h);
      const int top_row_base_x =
          ((tile_col << 6) - (tile_row + 1 + mrl_index) * dx) >> frac_bits;
      const int bot_row_base_x =
          ((tile_col << 6) - (tile_row + tile_h + mrl_index) * dx) >> frac_bits;

      if (bot_row_base_x >= min_base_x) {
        if (bd < 10) {
          for (int row_in_tile = 0; row_in_tile < tile_h; ++row_in_tile) {
            const int row_offset = tile_row + row_in_tile + 1;
            const int base_x =
                ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
            const int filter_phase =
                ((-(row_offset + mrl_index) * dx) & 0x3F) >> 1;
            __m128i pred_row;
            if (tile_w == 8) {
              pred_row = highbd_convolve8_idif_16bit_sse2(
                  _mm_loadu_si128((const __m128i *)(above + base_x - 1)),
                  _mm_loadu_si128((const __m128i *)(above + base_x)),
                  _mm_loadu_si128((const __m128i *)(above + base_x + 1)),
                  _mm_loadu_si128((const __m128i *)(above + base_x + 2)),
                  filter_phase, round_16bit, bd);
              _mm_storeu_si128(
                  (__m128i *)(dst + row_in_tile * stride + tile_col), pred_row);
            } else {
              pred_row = highbd_convolve4_idif_16bit_sse2(
                  _mm_loadu_si128((const __m128i *)(above + base_x - 1)),
                  filter_phase, round_16bit, bd);
              _mm_storel_epi64(
                  (__m128i *)(dst + row_in_tile * stride + tile_col), pred_row);
            }
          }
        } else {
          for (int row_in_tile = 0; row_in_tile < tile_h; ++row_in_tile) {
            const int row_offset = tile_row + row_in_tile + 1;
            const int base_x =
                ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
            __m128i taps_01, taps_23, pred_row;
            highbd_load_idif_taps_128(
                ((-(row_offset + mrl_index) * dx) & 0x3F) >> 1, &taps_01,
                &taps_23);
            if (tile_w == 8) {
              pred_row = highbd_convolve8_idif_32bit_sse41(
                  _mm_loadu_si128((const __m128i *)(above + base_x - 1)),
                  _mm_loadu_si128((const __m128i *)(above + base_x)),
                  _mm_loadu_si128((const __m128i *)(above + base_x + 1)),
                  _mm_loadu_si128((const __m128i *)(above + base_x + 2)),
                  taps_01, taps_23, round_32bit, max_pixel);
              _mm_storeu_si128(
                  (__m128i *)(dst + row_in_tile * stride + tile_col), pred_row);
            } else {
              pred_row = highbd_convolve4_idif_32bit_sse41(
                  _mm_loadu_si128((const __m128i *)(above + base_x - 1)),
                  taps_01, taps_23, round_32bit, max_pixel);
              _mm_storel_epi64(
                  (__m128i *)(dst + row_in_tile * stride + tile_col), pred_row);
            }
          }
        }
      } else if (min_base_x - top_row_base_x >= tile_w) {
        if (bd < 10) {
          highbd_dr_z2_8x8_idif_avx2(bw, bh, left, left_pred_cols, tile_col,
                                     tile_row, dy, mrl_index, bd);
        } else {
          highbd_dr_z2_32bit_8x8_idif_avx2(bw, bh, left, left_pred_cols,
                                           tile_col, tile_row, dy, mrl_index,
                                           bd);
        }
        highbd_transpose8x8_sse2(
            &left_pred_cols[0], &left_pred_cols[1], &left_pred_cols[2],
            &left_pred_cols[3], &left_pred_cols[4], &left_pred_cols[5],
            &left_pred_cols[6], &left_pred_cols[7], &left_pred[0],
            &left_pred[1], &left_pred[2], &left_pred[3], &left_pred[4],
            &left_pred[5], &left_pred[6], &left_pred[7]);
        for (int row_in_tile = 0; row_in_tile < tile_h; ++row_in_tile) {
          if (tile_w == 8) {
            _mm_storeu_si128((__m128i *)(dst + row_in_tile * stride + tile_col),
                             left_pred[row_in_tile]);
          } else {
            _mm_storel_epi64((__m128i *)(dst + row_in_tile * stride + tile_col),
                             left_pred[row_in_tile]);
          }
        }
      } else {
        if (bd < 10) {
          highbd_dr_z2_8x8_idif_avx2(bh, bw, above, above_pred, tile_row,
                                     tile_col, dx, mrl_index, bd);
          highbd_dr_z2_8x8_idif_avx2(bw, bh, left, left_pred_cols, tile_col,
                                     tile_row, dy, mrl_index, bd);
        } else {
          highbd_dr_z2_32bit_8x8_idif_avx2(bh, bw, above, above_pred, tile_row,
                                           tile_col, dx, mrl_index, bd);
          highbd_dr_z2_32bit_8x8_idif_avx2(bw, bh, left, left_pred_cols,
                                           tile_col, tile_row, dy, mrl_index,
                                           bd);
        }
        highbd_transpose8x8_sse2(
            &left_pred_cols[0], &left_pred_cols[1], &left_pred_cols[2],
            &left_pred_cols[3], &left_pred_cols[4], &left_pred_cols[5],
            &left_pred_cols[6], &left_pred_cols[7], &left_pred[0],
            &left_pred[1], &left_pred[2], &left_pred[3], &left_pred[4],
            &left_pred[5], &left_pred[6], &left_pred[7]);

        for (int row_in_tile = 0; row_in_tile < tile_h; ++row_in_tile) {
          const int row_offset = tile_row + row_in_tile + 1;
          const int base_x =
              ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
          int num_left_lanes = min_base_x - base_x;
          if (num_left_lanes > tile_w) {
            num_left_lanes = tile_w;
          } else if (num_left_lanes < 0) {
            num_left_lanes = 0;
          }
          const __m128i blended_row = _mm_blendv_epi8(
              above_pred[row_in_tile], left_pred[row_in_tile],
              _mm_load_si128((const __m128i *)HighbdBaseMask[num_left_lanes]));
          if (tile_w == 8) {
            _mm_storeu_si128((__m128i *)(dst + row_in_tile * stride + tile_col),
                             blended_row);
          } else {
            _mm_storel_epi64((__m128i *)(dst + row_in_tile * stride + tile_col),
                             blended_row);
          }
        }
      }
    }
    if (tile_row + 8 < bh) dst += 8 * stride;
  }
}

static void highbd_dr_z2_16x16_idif_avx2(int num_rows, int num_cols,
                                         const uint16_t *edge,
                                         __m256i *tile_rows, int start_row,
                                         int start_col, int ray_step,
                                         int mrl_index, int bd) {
  (void)num_rows;
  (void)num_cols;
  const int min_base_idx = -(1 + mrl_index);
  const int frac_bits = 6;
  const __m256i round_256 =
      _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i round_128 = _mm_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));

  for (int row = start_row; row < start_row + 16; ++row) {
    assert(row < num_rows);
    assert(start_col < num_cols);
    const int row_offset = row + 1;
    const int base_idx =
        ((start_col << 6) - (row_offset + mrl_index) * ray_step) >> frac_bits;
    const int filter_phase =
        ((-(row_offset + mrl_index) * ray_step) & 0x3F) >> 1;

    if (base_idx >= min_base_idx - 1) {
      tile_rows[row - start_row] = highbd_convolve16_idif_16bit_avx2(
          _mm256_loadu_si256((const __m256i *)(edge + base_idx - 1)),
          _mm256_loadu_si256((const __m256i *)(edge + base_idx)),
          _mm256_loadu_si256((const __m256i *)(edge + base_idx + 1)),
          _mm256_loadu_si256((const __m256i *)(edge + base_idx + 2)),
          filter_phase, round_256, bd);
    } else {
      const int clamp_count = min_base_idx - base_idx - 1;
      if (clamp_count >= 16) {
        tile_rows[row - start_row] = _mm256_setzero_si256();
      } else {
        const __m128i pred_lo =
            (clamp_count < 8)
                ? highbd_dr_z2_row8_16bit_idif_sse2(
                      edge, filter_phase, base_idx, clamp_count, round_128, bd)
                : _mm_setzero_si128();
        const int hi_clamp_count = (clamp_count > 8) ? (clamp_count - 8) : 0;
        const __m128i pred_hi = highbd_dr_z2_row8_16bit_idif_sse2(
            edge, filter_phase, base_idx + 8, hi_clamp_count, round_128, bd);
        tile_rows[row - start_row] = _mm256_inserti128_si256(
            _mm256_castsi128_si256(pred_lo), pred_hi, 1);
      }
    }
  }
}

static void highbd_dr_prediction_z2_HxW_idif_avx2(
    int bh, int bw, uint16_t *dst, ptrdiff_t stride, const uint16_t *above,
    const uint16_t *left, int dx, int dy, int mrl_index, int bd) {
  const int min_base_x = -(1 + mrl_index);
  const int frac_bits = 6;
  const __m256i round_const =
      _mm256_set1_epi16(1 << (POWER_DR_INTERP_FILTER - 1));

  __m256i above_pred[16];
  __m256i left_pred[16];

  for (int tile_row = 0; tile_row < bh; tile_row += 16) {
    for (int tile_col = 0; tile_col < bw; tile_col += 16) {
      assert((bw - tile_col) >= 16);
      assert((bh - tile_row) >= 16);
      const int top_row_base_x =
          ((tile_col << 6) - (tile_row + 1 + mrl_index) * dx) >> frac_bits;
      const int bot_row_base_x =
          ((tile_col << 6) - (tile_row + 16 + mrl_index) * dx) >> frac_bits;

      if (bot_row_base_x >= min_base_x) {
        for (int row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
          const int row_offset = row_in_tile + tile_row + 1;
          const int base_x =
              ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
          const int filter_phase =
              ((-(row_offset + mrl_index) * dx) & 0x3F) >> 1;
          const __m256i pred_row = highbd_convolve16_idif_16bit_avx2(
              _mm256_loadu_si256((const __m256i *)(above + base_x - 1)),
              _mm256_loadu_si256((const __m256i *)(above + base_x)),
              _mm256_loadu_si256((const __m256i *)(above + base_x + 1)),
              _mm256_loadu_si256((const __m256i *)(above + base_x + 2)),
              filter_phase, round_const, bd);
          _mm256_storeu_si256(
              (__m256i *)(dst + row_in_tile * stride + tile_col), pred_row);
        }
      } else if (min_base_x - top_row_base_x >= 16) {
        highbd_dr_z2_16x16_idif_avx2(bw, bh, left, left_pred, tile_col,
                                     tile_row, dy, mrl_index, bd);
        highbd_transpose16x16_avx2(left_pred, left_pred);
        for (int row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
          _mm256_storeu_si256(
              (__m256i *)(dst + row_in_tile * stride + tile_col),
              left_pred[row_in_tile]);
        }
      } else {
        highbd_dr_z2_16x16_idif_avx2(bh, bw, above, above_pred, tile_row,
                                     tile_col, dx, mrl_index, bd);
        highbd_dr_z2_16x16_idif_avx2(bw, bh, left, left_pred, tile_col,
                                     tile_row, dy, mrl_index, bd);
        highbd_transpose16x16_avx2(left_pred, left_pred);

        for (int row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
          const int row_offset = row_in_tile + tile_row + 1;
          const int base_x =
              ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
          int num_left_lanes = min_base_x - base_x;
          if (num_left_lanes > 16) {
            num_left_lanes = 16;
          } else if (num_left_lanes < 0) {
            num_left_lanes = 0;
          }
          const __m256i blended_row = _mm256_blendv_epi8(
              above_pred[row_in_tile], left_pred[row_in_tile],
              _mm256_load_si256(
                  (const __m256i *)HighbdBaseMask[num_left_lanes]));
          _mm256_storeu_si256(
              (__m256i *)(dst + row_in_tile * stride + tile_col), blended_row);
        }
      }
    }
    if (tile_row + 16 < bh) dst += 16 * stride;
  }
}

static void highbd_dr_z2_16x16_32bit_idif_avx2(
    int num_rows, int num_cols, const uint16_t *edge, __m256i *tile_rows,
    int start_row, int start_col, int ray_step, int mrl_index, int bd) {
  (void)num_rows;
  (void)num_cols;
  const int min_base_idx = -(1 + mrl_index);
  const int frac_bits = 6;
  const __m256i round_256 =
      _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i max_pixel_256 = _mm256_set1_epi16((int16_t)((1 << bd) - 1));
  const __m128i round_128 = _mm_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m128i max_pixel_128 = _mm_set1_epi16((int16_t)((1 << bd) - 1));

  for (int row = start_row; row < start_row + 16; ++row) {
    assert(row < num_rows);
    assert(start_col < num_cols);
    const int row_offset = row + 1;
    const int base_idx =
        ((start_col << 6) - (row_offset + mrl_index) * ray_step) >> frac_bits;
    const int filter_phase =
        ((-(row_offset + mrl_index) * ray_step) & 0x3F) >> 1;

    if (base_idx >= min_base_idx - 1) {
      __m256i taps_01, taps_23;
      highbd_load_idif_taps_256(filter_phase, &taps_01, &taps_23);
      tile_rows[row - start_row] = highbd_convolve16_idif_32bit_avx2(
          _mm256_loadu_si256((const __m256i *)(edge + base_idx - 1)),
          _mm256_loadu_si256((const __m256i *)(edge + base_idx)),
          _mm256_loadu_si256((const __m256i *)(edge + base_idx + 1)),
          _mm256_loadu_si256((const __m256i *)(edge + base_idx + 2)), taps_01,
          taps_23, round_256, max_pixel_256);
    } else {
      const int clamp_count = min_base_idx - base_idx - 1;
      if (clamp_count >= 16) {
        tile_rows[row - start_row] = _mm256_setzero_si256();
      } else {
        __m128i taps_01, taps_23;
        highbd_load_idif_taps_128(filter_phase, &taps_01, &taps_23);
        const __m128i pred_lo = (clamp_count < 8)
                                    ? highbd_dr_z2_row8_32bit_idif_sse41(
                                          edge, taps_01, taps_23, base_idx,
                                          clamp_count, round_128, max_pixel_128)
                                    : _mm_setzero_si128();
        const int hi_clamp_count = (clamp_count > 8) ? (clamp_count - 8) : 0;
        const __m128i pred_hi = highbd_dr_z2_row8_32bit_idif_sse41(
            edge, taps_01, taps_23, base_idx + 8, hi_clamp_count, round_128,
            max_pixel_128);
        tile_rows[row - start_row] = _mm256_inserti128_si256(
            _mm256_castsi128_si256(pred_lo), pred_hi, 1);
      }
    }
  }
}

static void highbd_dr_prediction_32bit_z2_HxW_idif_avx2(
    int bh, int bw, uint16_t *dst, ptrdiff_t stride, const uint16_t *above,
    const uint16_t *left, int dx, int dy, int mrl_index, int bd) {
  const int min_base_x = -(1 + mrl_index);
  const int frac_bits = 6;
  const __m256i round_const =
      _mm256_set1_epi32(1 << (POWER_DR_INTERP_FILTER - 1));
  const __m256i max_pixel = _mm256_set1_epi16((int16_t)((1 << bd) - 1));

  __m256i above_pred[16];
  __m256i left_pred[16];

  for (int tile_row = 0; tile_row < bh; tile_row += 16) {
    for (int tile_col = 0; tile_col < bw; tile_col += 16) {
      assert((bw - tile_col) >= 16);
      assert((bh - tile_row) >= 16);
      const int top_row_base_x =
          ((tile_col << 6) - (tile_row + 1 + mrl_index) * dx) >> frac_bits;
      const int bot_row_base_x =
          ((tile_col << 6) - (tile_row + 16 + mrl_index) * dx) >> frac_bits;

      if (bot_row_base_x >= min_base_x) {
        for (int row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
          const int row_offset = row_in_tile + tile_row + 1;
          const int base_x =
              ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
          __m256i taps_01, taps_23;
          highbd_load_idif_taps_256(
              ((-(row_offset + mrl_index) * dx) & 0x3F) >> 1, &taps_01,
              &taps_23);
          const __m256i pred_row = highbd_convolve16_idif_32bit_avx2(
              _mm256_loadu_si256((const __m256i *)(above + base_x - 1)),
              _mm256_loadu_si256((const __m256i *)(above + base_x)),
              _mm256_loadu_si256((const __m256i *)(above + base_x + 1)),
              _mm256_loadu_si256((const __m256i *)(above + base_x + 2)),
              taps_01, taps_23, round_const, max_pixel);
          _mm256_storeu_si256(
              (__m256i *)(dst + row_in_tile * stride + tile_col), pred_row);
        }
      } else if (min_base_x - top_row_base_x >= 16) {
        highbd_dr_z2_16x16_32bit_idif_avx2(bw, bh, left, left_pred, tile_col,
                                           tile_row, dy, mrl_index, bd);
        highbd_transpose16x16_avx2(left_pred, left_pred);
        for (int row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
          _mm256_storeu_si256(
              (__m256i *)(dst + row_in_tile * stride + tile_col),
              left_pred[row_in_tile]);
        }
      } else {
        highbd_dr_z2_16x16_32bit_idif_avx2(bh, bw, above, above_pred, tile_row,
                                           tile_col, dx, mrl_index, bd);
        highbd_dr_z2_16x16_32bit_idif_avx2(bw, bh, left, left_pred, tile_col,
                                           tile_row, dy, mrl_index, bd);
        highbd_transpose16x16_avx2(left_pred, left_pred);

        for (int row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
          const int row_offset = row_in_tile + tile_row + 1;
          const int base_x =
              ((tile_col << 6) - (row_offset + mrl_index) * dx) >> frac_bits;
          int num_left_lanes = min_base_x - base_x;
          if (num_left_lanes > 16) {
            num_left_lanes = 16;
          } else if (num_left_lanes < 0) {
            num_left_lanes = 0;
          }
          const __m256i blended_row = _mm256_blendv_epi8(
              above_pred[row_in_tile], left_pred[row_in_tile],
              _mm256_load_si256(
                  (const __m256i *)HighbdBaseMask[num_left_lanes]));
          _mm256_storeu_si256(
              (__m256i *)(dst + row_in_tile * stride + tile_col), blended_row);
        }
      }
    }
    if (tile_row + 16 < bh) dst += 16 * stride;
  }
}

// Directional prediction, zone 2: 90 < angle < 180 using IDIF
void av2_highbd_dr_prediction_z2_idif_avx2(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int dx, int dy,
                                           int bd, int mrl_index) {
  assert(dx > 0);
  assert(dy > 0);
  switch (bw) {
    case 4:
      highbd_dr_z2_8x8_tiling_idif_avx2(bh, bw, dst, stride, above, left, dx,
                                        dy, mrl_index, bd);
      break;
    case 8:
      highbd_dr_z2_8x8_tiling_idif_avx2(bh, bw, dst, stride, above, left, dx,
                                        dy, mrl_index, bd);
      break;
    default:
      if (bh < 16) {
        highbd_dr_z2_8x8_tiling_idif_avx2(bh, bw, dst, stride, above, left, dx,
                                          dy, mrl_index, bd);
      } else {
        if (bd < 10) {
          highbd_dr_prediction_z2_HxW_idif_avx2(bh, bw, dst, stride, above,
                                                left, dx, dy, mrl_index, bd);
        } else {
          highbd_dr_prediction_32bit_z2_HxW_idif_avx2(
              bh, bw, dst, stride, above, left, dx, dy, mrl_index, bd);
        }
      }
      break;
  }
}

//  Directional prediction, zone 3 functions
static void highbd_dr_prediction_z3_4x4_idif_avx2(uint16_t *dst,
                                                  ptrdiff_t stride,
                                                  const uint16_t *left, int dy,
                                                  int bd, int mrl_index) {
  __m128i dstvec[4], d[4];
  if (bd < 10) {
    highbd_dr_prediction_z1_4xN_internal_idif_avx2(4, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_idif_avx2(4, dstvec, left, dy,
                                                         mrl_index, bd);
  }
  highbd_transpose4x8_8x4_low_sse2(&dstvec[0], &dstvec[1], &dstvec[2],
                                   &dstvec[3], &d[0], &d[1], &d[2], &d[3]);
  _mm_storel_epi64((__m128i *)(dst + 0 * stride), d[0]);
  _mm_storel_epi64((__m128i *)(dst + 1 * stride), d[1]);
  _mm_storel_epi64((__m128i *)(dst + 2 * stride), d[2]);
  _mm_storel_epi64((__m128i *)(dst + 3 * stride), d[3]);
  return;
}

static void highbd_dr_prediction_z3_8x8_idif_avx2(uint16_t *dst,
                                                  ptrdiff_t stride,
                                                  const uint16_t *left, int dy,
                                                  int bd, int mrl_index) {
  __m128i dstvec[8], d[8];
  if (bd < 10) {
    highbd_dr_prediction_z1_8xN_internal_idif_avx2(8, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_idif_avx2(8, dstvec, left, dy,
                                                         mrl_index, bd);
  }
  highbd_transpose8x8_sse2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                           &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                           &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6],
                           &d[7]);
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_16x16_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  __m256i dstvec[16], d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_16xN_internal_idif_avx2(16, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_idif_avx2(16, dstvec, left, dy,
                                                          mrl_index, bd);
  }

  highbd_transpose16x16_avx2(dstvec, d);

  for (int i = 0; i < 16; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_32x32_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  __m256i dstvec[64], d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_32xN_internal_idif_avx2(32, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_idif_avx2(32, dstvec, left, dy,
                                                          mrl_index, bd);
  }
  highbd_transpose16x16_avx2(dstvec, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + j * stride), d[j]);
  }
  highbd_transpose16x16_avx2(dstvec + 16, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + j * stride + 16), d[j]);
  }
  highbd_transpose16x16_avx2(dstvec + 32, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + (j + 16) * stride), d[j]);
  }
  highbd_transpose16x16_avx2(dstvec + 48, d);
  for (int j = 0; j < 16; j++) {
    _mm256_storeu_si256((__m256i *)(dst + (j + 16) * stride + 16), d[j]);
  }
}

static void highbd_dr_prediction_z3_64x64_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  DECLARE_ALIGNED(16, uint16_t, dstT[64 * 64]);
  if (bd < 10) {
    highbd_dr_prediction_z1_64xN_internal_idif_avx2(64, dstT, 64, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(64, dstT, 64, left,
                                                          dy, mrl_index, bd);
  }
  highbd_transpose(dstT, 64, dst, stride, 64, 64);
}

static void highbd_dr_prediction_z3_4x8_idif_avx2(uint16_t *dst,
                                                  ptrdiff_t stride,
                                                  const uint16_t *left, int dy,
                                                  int bd, int mrl_index) {
  __m128i dstvec[4], d[8];
  if (bd < 10) {
    highbd_dr_prediction_z1_8xN_internal_idif_avx2(4, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_idif_avx2(4, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  highbd_transpose4x8_8x4_sse2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                               &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6],
                               &d[7]);
  for (int i = 0; i < 8; i++) {
    _mm_storel_epi64((__m128i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_8x4_idif_avx2(uint16_t *dst,
                                                  ptrdiff_t stride,
                                                  const uint16_t *left, int dy,
                                                  int bd, int mrl_index) {
  __m128i dstvec[8], d[4];
  if (bd < 10) {
    highbd_dr_prediction_z1_4xN_internal_idif_avx2(8, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_idif_avx2(8, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  highbd_transpose8x8_low_sse2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                               &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                               &d[0], &d[1], &d[2], &d[3]);
  _mm_storeu_si128((__m128i *)(dst + 0 * stride), d[0]);
  _mm_storeu_si128((__m128i *)(dst + 1 * stride), d[1]);
  _mm_storeu_si128((__m128i *)(dst + 2 * stride), d[2]);
  _mm_storeu_si128((__m128i *)(dst + 3 * stride), d[3]);
}

static void highbd_dr_prediction_z3_8x16_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m256i dstvec[8], d[8];
  if (bd < 10) {
    highbd_dr_prediction_z1_16xN_internal_idif_avx2(8, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_idif_avx2(8, dstvec, left, dy,
                                                          mrl_index, bd);
  }
  highbd_transpose8x16_16x8_avx2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                                 &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                                 &d[0], &d[1], &d[2], &d[3], &d[4], &d[5],
                                 &d[6], &d[7]);

  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride),
                     _mm256_castsi256_si128(d[i]));
  }
  for (int i = 8; i < 16; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride),
                     _mm256_extracti128_si256(d[i - 8], 1));
  }
}

static void highbd_dr_prediction_z3_16x8_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m128i dstvec[16];
  __m256i dstvec256[8], d256[8];
  if (bd < 10) {
    highbd_dr_prediction_z1_8xN_internal_idif_avx2(16, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_idif_avx2(16, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  for (int i = 0; i < 8; i++) {
    dstvec256[i] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 0]), dstvec[i + 8], 0x1);
  }

  highbd_transpose8x16_16x8_avx2(
      &dstvec256[0], &dstvec256[1], &dstvec256[2], &dstvec256[3], &dstvec256[4],
      &dstvec256[5], &dstvec256[6], &dstvec256[7], &d256[0], &d256[1], &d256[2],
      &d256[3], &d256[4], &d256[5], &d256[6], &d256[7]);

  for (int i = 0; i < 8; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d256[i]);
  }
}

static void highbd_dr_prediction_z3_4x16_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m256i dstvec[4], d[4], d1;
  if (bd < 10) {
    highbd_dr_prediction_z1_16xN_internal_idif_avx2(4, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_idif_avx2(4, dstvec, left, dy,
                                                          mrl_index, bd);
  }
  highbd_transpose4x16_avx2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                            &d[0], &d[1], &d[2], &d[3]);
  for (int i = 0; i < 4; i++) {
    _mm_storel_epi64((__m128i *)(dst + i * stride),
                     _mm256_castsi256_si128(d[i]));
    d1 = _mm256_bsrli_epi128(d[i], 8);
    _mm_storel_epi64((__m128i *)(dst + (i + 4) * stride),
                     _mm256_castsi256_si128(d1));
    _mm_storel_epi64((__m128i *)(dst + (i + 8) * stride),
                     _mm256_extracti128_si256(d[i], 1));
    _mm_storel_epi64((__m128i *)(dst + (i + 12) * stride),
                     _mm256_extracti128_si256(d1, 1));
  }
}

static void highbd_dr_prediction_z3_64x8_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m128i dstvec[64];
  __m256i dstvec256[32], d256[32];
  if (bd < 10) {
    highbd_dr_prediction_z1_8xN_internal_idif_avx2(64, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_idif_avx2(64, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  for (int i = 0; i < 8; i++) {
    dstvec256[i] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 0]), dstvec[i + 8], 0x1);
    dstvec256[i + 8] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 16]), dstvec[i + 24], 0x1);
    dstvec256[i + 16] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 32]), dstvec[i + 40], 0x1);
    dstvec256[i + 24] = _mm256_insertf128_si256(
        _mm256_castsi128_si256(dstvec[i + 48]), dstvec[i + 56], 0x1);
  }

  for (int i = 0; i < 32; i += 8) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec256[i + 0], &dstvec256[i + 1], &dstvec256[i + 2],
        &dstvec256[i + 3], &dstvec256[i + 4], &dstvec256[i + 5],
        &dstvec256[i + 6], &dstvec256[i + 7], &d256[i + 0], &d256[i + 1],
        &d256[i + 2], &d256[i + 3], &d256[i + 4], &d256[i + 5], &d256[i + 6],
        &d256[i + 7]);
  }
  for (int i = 0; i < 8; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d256[i]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 16), d256[i + 8]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 32), d256[i + 16]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 48), d256[i + 24]);
  }
}

static void highbd_dr_prediction_z3_64x4_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m128i dstvec[64];
  __m256i d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_4xN_internal_idif_avx2(64, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_idif_avx2(64, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  highbd_transpose16x4_8x8_avx2(dstvec, d);
  highbd_transpose16x4_8x8_avx2(dstvec + 16, d + 4);
  highbd_transpose16x4_8x8_avx2(dstvec + 32, d + 8);
  highbd_transpose16x4_8x8_avx2(dstvec + 48, d + 12);

  for (int i = 0; i < 4; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 16), d[i + 4]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 32), d[i + 8]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 48), d[i + 12]);
  }
}

static void highbd_dr_prediction_z3_4x64_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m256i dstvec[16], d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_64xN_internal_idif_avx2(4, (uint16_t *)dstvec, 64,
                                                    left, dy, mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(
        4, (uint16_t *)dstvec, 64, left, dy, mrl_index, bd);
  }

  for (int i = 0; i < 4; i++) {
    highbd_transpose4x16_avx2(&dstvec[i], &dstvec[i + 4], &dstvec[i + 8],
                              &dstvec[i + 12], &d[4 * i], &d[4 * i + 1],
                              &d[4 * i + 2], &d[4 * i + 3]);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      const __m128i temph = _mm256_extracti128_si256(d[4 * i + j], 1);
      const __m128i templ = _mm256_castsi256_si128(d[4 * i + j]);
      xx_storel_64(dst + (16 * i + j) * stride, templ);
      xx_storel_64(dst + (16 * i + j + 4) * stride, _mm_srli_si128(templ, 8));
      xx_storel_64(dst + (16 * i + j + 8) * stride, temph);
      xx_storel_64(dst + (16 * i + j + 12) * stride, _mm_srli_si128(temph, 8));
    }
  }
}

static void highbd_dr_prediction_z3_8x64_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m256i dstvec[32], d[32];
  if (bd < 10) {
    highbd_dr_prediction_z1_64xN_internal_idif_avx2(8, (uint16_t *)dstvec, 64,
                                                    left, dy, mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(
        8, (uint16_t *)dstvec, 64, left, dy, mrl_index, bd);
  }

  for (int i = 0; i < 4; i++) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec[i], &dstvec[i + 4], &dstvec[i + 8], &dstvec[i + 12],
        &dstvec[i + 16], &dstvec[i + 20], &dstvec[i + 24], &dstvec[i + 28],
        &d[8 * i], &d[8 * i + 1], &d[8 * i + 2], &d[8 * i + 3], &d[8 * i + 4],
        &d[8 * i + 5], &d[8 * i + 6], &d[8 * i + 7]);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      __m128i temp_lo = _mm256_castsi256_si128(d[8 * i + j]);
      __m128i temp_hi = _mm256_extracti128_si256(d[8 * i + j], 1);
      _mm_storeu_si128((__m128i *)(dst + (16 * i + j) * stride), temp_lo);
      _mm_storeu_si128((__m128i *)(dst + (16 * i + j + 8) * stride), temp_hi);
    }
  }
}

static void highbd_dr_prediction_z3_4x32_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m256i dstvec[8], d[8];
  if (bd < 10) {
    highbd_dr_prediction_z1_32xN_internal_idif_avx2(4, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_idif_avx2(4, dstvec, left, dy,
                                                          mrl_index, bd);
  }

  highbd_transpose8x16_16x8_avx2(&dstvec[0], &dstvec[1], &dstvec[2], &dstvec[3],
                                 &dstvec[4], &dstvec[5], &dstvec[6], &dstvec[7],
                                 &d[0], &d[1], &d[2], &d[3], &d[4], &d[5],
                                 &d[6], &d[7]);

  for (int i = 0; i < 8; i++) {
    __m128i temp_lo = _mm256_castsi256_si128(d[i]);
    __m128i temp_hi = _mm256_extracti128_si256(d[i], 1);
    _mm_storel_epi64((__m128i *)(dst + i * stride), temp_lo);
    _mm_storel_epi64((__m128i *)(dst + (i + 16) * stride),
                     _mm_srli_si128(temp_lo, 8));
    _mm_storel_epi64((__m128i *)(dst + (i + 8) * stride), temp_hi);
    _mm_storel_epi64((__m128i *)(dst + (i + 24) * stride),
                     _mm_srli_si128(temp_hi, 8));
  }
}

static void highbd_dr_prediction_z3_32x4_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m128i dstvec[32];
  __m256i d[8];
  if (bd < 10) {
    highbd_dr_prediction_z1_4xN_internal_idif_avx2(32, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_idif_avx2(32, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  highbd_transpose16x4_8x8_avx2(dstvec, d);
  highbd_transpose16x4_8x8_avx2(dstvec + 16, d + 4);

  for (int i = 0; i < 4; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
    _mm256_storeu_si256((__m256i *)(dst + i * stride + 16), d[i + 4]);
  }
}

static void highbd_dr_prediction_z3_16x4_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m128i dstvec[16];
  __m256i d[4];
  if (bd < 10) {
    highbd_dr_prediction_z1_4xN_internal_idif_avx2(16, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_4xN_internal_idif_avx2(16, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  highbd_transpose16x4_8x8_avx2(dstvec, d);

  for (int i = 0; i < 4; i++) {
    _mm256_storeu_si256((__m256i *)(dst + i * stride), d[i]);
  }
}

static void highbd_dr_prediction_z3_8x32_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m256i dstvec[16], d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_32xN_internal_idif_avx2(8, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_idif_avx2(8, dstvec, left, dy,
                                                          mrl_index, bd);
  }

  for (int i = 0; i < 16; i += 8) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec[i], &dstvec[i + 1], &dstvec[i + 2], &dstvec[i + 3],
        &dstvec[i + 4], &dstvec[i + 5], &dstvec[i + 6], &dstvec[i + 7], &d[i],
        &d[i + 1], &d[i + 2], &d[i + 3], &d[i + 4], &d[i + 5], &d[i + 6],
        &d[i + 7]);
  }

  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride),
                     _mm256_castsi256_si128(d[i]));
  }
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + (i + 8) * stride),
                     _mm256_extracti128_si256(d[i], 1));
  }
  for (int i = 8; i < 16; i++) {
    _mm_storeu_si128((__m128i *)(dst + (i + 8) * stride),
                     _mm256_castsi256_si128(d[i]));
  }
  for (int i = 8; i < 16; i++) {
    _mm_storeu_si128((__m128i *)(dst + (i + 16) * stride),
                     _mm256_extracti128_si256(d[i], 1));
  }
}

static void highbd_dr_prediction_z3_32x8_idif_avx2(uint16_t *dst,
                                                   ptrdiff_t stride,
                                                   const uint16_t *left, int dy,
                                                   int bd, int mrl_index) {
  __m128i dstvec[32], d[32];
  if (bd < 10) {
    highbd_dr_prediction_z1_8xN_internal_idif_avx2(32, dstvec, left, dy,
                                                   mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_8xN_internal_idif_avx2(32, dstvec, left, dy,
                                                         mrl_index, bd);
  }

  for (int i = 0; i < 32; i += 8) {
    highbd_transpose8x8_sse2(&dstvec[0 + i], &dstvec[1 + i], &dstvec[2 + i],
                             &dstvec[3 + i], &dstvec[4 + i], &dstvec[5 + i],
                             &dstvec[6 + i], &dstvec[7 + i], &d[0 + i],
                             &d[1 + i], &d[2 + i], &d[3 + i], &d[4 + i],
                             &d[5 + i], &d[6 + i], &d[7 + i]);
  }
  for (int i = 0; i < 8; i++) {
    _mm_storeu_si128((__m128i *)(dst + i * stride), d[i]);
    _mm_storeu_si128((__m128i *)(dst + i * stride + 8), d[i + 8]);
    _mm_storeu_si128((__m128i *)(dst + i * stride + 16), d[i + 16]);
    _mm_storeu_si128((__m128i *)(dst + i * stride + 24), d[i + 24]);
  }
}

static void highbd_dr_prediction_z3_16x32_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  __m256i dstvec[32], d[32];
  if (bd < 10) {
    highbd_dr_prediction_z1_32xN_internal_idif_avx2(16, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_32xN_internal_idif_avx2(16, dstvec, left, dy,
                                                          mrl_index, bd);
  }
  for (int i = 0; i < 32; i += 8) {
    highbd_transpose8x16_16x8_avx2(
        &dstvec[i], &dstvec[i + 1], &dstvec[i + 2], &dstvec[i + 3],
        &dstvec[i + 4], &dstvec[i + 5], &dstvec[i + 6], &dstvec[i + 7], &d[i],
        &d[i + 1], &d[i + 2], &d[i + 3], &d[i + 4], &d[i + 5], &d[i + 6],
        &d[i + 7]);
  }
  // store
  for (int j = 0; j < 32; j += 16) {
    for (int i = 0; i < 8; i++) {
      _mm_storeu_si128((__m128i *)(dst + (i + j) * stride),
                       _mm256_castsi256_si128(d[(i + j)]));
    }
    for (int i = 0; i < 8; i++) {
      _mm_storeu_si128((__m128i *)(dst + (i + j) * stride + 8),
                       _mm256_castsi256_si128(d[(i + j) + 8]));
    }
    for (int i = 8; i < 16; i++) {
      _mm256_storeu_si256(
          (__m256i *)(dst + (i + j) * stride),
          _mm256_inserti128_si256(
              d[(i + j)], _mm256_extracti128_si256(d[(i + j) - 8], 1), 0));
    }
  }
}

static void highbd_dr_prediction_z3_32x16_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  __m256i dstvec[32], d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_16xN_internal_idif_avx2(32, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_idif_avx2(32, dstvec, left, dy,
                                                          mrl_index, bd);
  }
  for (int i = 0; i < 32; i += 16) {
    highbd_transpose16x16_avx2((dstvec + i), d);
    for (int j = 0; j < 16; j++) {
      _mm256_storeu_si256((__m256i *)(dst + j * stride + i), d[j]);
    }
  }
}

static void highbd_dr_prediction_z3_32x64_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  uint16_t dstT[64 * 32];
  if (bd < 10) {
    highbd_dr_prediction_z1_64xN_internal_idif_avx2(32, dstT, 64, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(32, dstT, 64, left,
                                                          dy, mrl_index, bd);
  }
  highbd_transpose(dstT, 64, dst, stride, 32, 64);
}

static void highbd_dr_prediction_z3_64x32_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  DECLARE_ALIGNED(16, uint16_t, dstT[32 * 64]);
  highbd_dr_prediction_z1_32xN_idif_avx2(64, dstT, 32, left, dy, bd, mrl_index);
  highbd_transpose(dstT, 32, dst, stride, 64, 32);
  return;
}

static void highbd_dr_prediction_z3_16x64_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  DECLARE_ALIGNED(16, uint16_t, dstT[64 * 16]);
  if (bd < 10) {
    highbd_dr_prediction_z1_64xN_internal_idif_avx2(16, dstT, 64, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_64xN_internal_idif_avx2(16, dstT, 64, left,
                                                          dy, mrl_index, bd);
  }
  highbd_transpose(dstT, 64, dst, stride, 16, 64);
}

static void highbd_dr_prediction_z3_64x16_idif_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    const uint16_t *left,
                                                    int dy, int bd,
                                                    int mrl_index) {
  __m256i dstvec[64], d[16];
  if (bd < 10) {
    highbd_dr_prediction_z1_16xN_internal_idif_avx2(64, dstvec, left, dy,
                                                    mrl_index, bd);
  } else {
    highbd_dr_prediction_32bit_z1_16xN_internal_idif_avx2(64, dstvec, left, dy,
                                                          mrl_index, bd);
  }
  for (int i = 0; i < 64; i += 16) {
    highbd_transpose16x16_avx2((dstvec + i), d);
    for (int j = 0; j < 16; j++) {
      _mm256_storeu_si256((__m256i *)(dst + j * stride + i), d[j]);
    }
  }
}

void av2_highbd_dr_prediction_z3_idif_avx2(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int dx, int dy,
                                           int bd, int mrl_index) {
  (void)above;
  (void)dx;

  assert(dx == 1);
  assert(dy > 0);

  if (bw == bh) {
    switch (bw) {
      case 4:
        highbd_dr_prediction_z3_4x4_idif_avx2(dst, stride, left, dy, bd,
                                              mrl_index);
        break;
      case 8:
        highbd_dr_prediction_z3_8x8_idif_avx2(dst, stride, left, dy, bd,
                                              mrl_index);
        break;
      case 16:
        highbd_dr_prediction_z3_16x16_idif_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
        break;
      case 32:
        highbd_dr_prediction_z3_32x32_idif_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
        break;
      case 64:
        highbd_dr_prediction_z3_64x64_idif_avx2(dst, stride, left, dy, bd,
                                                mrl_index);
        break;
    }
  } else {
    if (bw < bh) {
      if (bw + bw == bh) {
        switch (bw) {
          case 4:
            highbd_dr_prediction_z3_4x8_idif_avx2(dst, stride, left, dy, bd,
                                                  mrl_index);
            break;
          case 8:
            highbd_dr_prediction_z3_8x16_idif_avx2(dst, stride, left, dy, bd,
                                                   mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_16x32_idif_avx2(dst, stride, left, dy, bd,
                                                    mrl_index);
            break;
          case 32:
            highbd_dr_prediction_z3_32x64_idif_avx2(dst, stride, left, dy, bd,
                                                    mrl_index);
            break;
        }
      } else {
        switch (bw) {
          case 4:
            if (bh == 32)
              highbd_dr_prediction_z3_4x32_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            else if (bh == 64)
              highbd_dr_prediction_z3_4x64_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            else
              highbd_dr_prediction_z3_4x16_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            break;
          case 8:
            if (bh == 64)
              highbd_dr_prediction_z3_8x64_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);

            else
              highbd_dr_prediction_z3_8x32_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_16x64_idif_avx2(dst, stride, left, dy, bd,
                                                    mrl_index);
            break;
        }
      }
    } else {
      if (bh + bh == bw) {
        switch (bh) {
          case 4:
            highbd_dr_prediction_z3_8x4_idif_avx2(dst, stride, left, dy, bd,
                                                  mrl_index);
            break;
          case 8:
            highbd_dr_prediction_z3_16x8_idif_avx2(dst, stride, left, dy, bd,
                                                   mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_32x16_idif_avx2(dst, stride, left, dy, bd,
                                                    mrl_index);
            break;
          case 32:
            highbd_dr_prediction_z3_64x32_idif_avx2(dst, stride, left, dy, bd,
                                                    mrl_index);
            break;
        }
      } else {
        switch (bh) {
          case 4:
            if (bw == 32)

              highbd_dr_prediction_z3_32x4_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            else if (bw == 64)
              highbd_dr_prediction_z3_64x4_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);

            else
              highbd_dr_prediction_z3_16x4_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);

            break;
          case 8:
            if (bw == 64)
              highbd_dr_prediction_z3_64x8_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            else
              highbd_dr_prediction_z3_32x8_idif_avx2(dst, stride, left, dy, bd,
                                                     mrl_index);
            break;
          case 16:
            highbd_dr_prediction_z3_64x16_idif_avx2(dst, stride, left, dy, bd,
                                                    mrl_index);
            break;
        }
      }
    }
  }
  return;
}

static INLINE __m256i highbd_smooth_dist_weights_avx2(__m256i idx, int size) {
  return _mm256_or_si256(
      _mm256_sub_epi32(_mm256_set1_epi32(size - 1), idx),
      _mm256_slli_epi32(_mm256_add_epi32(idx, _mm256_set1_epi32(1)), 16));
}

static INLINE __m256i highbd_smooth_blend_weights_avx2(__m256i idx,
                                                       __m128i scale) {
  const __m256i shift = _mm256_min_epi32(
      _mm256_sra_epi32(_mm256_slli_epi32(idx, 1), scale), _mm256_set1_epi32(6));
  const __m256i weight = _mm256_srlv_epi32(_mm256_set1_epi32(32), shift);
  return _mm256_or_si256(_mm256_sub_epi32(_mm256_set1_epi32(64), weight),
                         _mm256_slli_epi32(weight, 16));
}

static AVM_FORCE_INLINE __m256i highbd_smooth_pred_avx2(
    PREDICTION_MODE mode, __m256i top_bl, __m256i top_hi, __m256i weights_v,
    __m256i s_top, __m256i left_tr, __m256i left_hi, __m256i weights_h,
    __m256i s_left, int bh_log2, int bw_log2) {
  const __m256i round_const = _mm256_set1_epi32(32);
  __m256i predv = _mm256_setzero_si256();
  __m256i predh = _mm256_setzero_si256();
  if (mode != SMOOTH_H_PRED) {
    predv = _mm256_madd_epi16(top_bl, weights_v);
    predv = _mm256_srai_epi32(
        _mm256_add_epi32(predv, _mm256_set1_epi32(1 << (bh_log2 - 1))),
        bh_log2);
    predv = _mm256_madd_epi16(_mm256_or_si256(predv, top_hi), s_top);
    predv = _mm256_srai_epi32(_mm256_add_epi32(predv, round_const), 6);
  }

  if (mode != SMOOTH_V_PRED) {
    predh = _mm256_madd_epi16(left_tr, weights_h);
    predh = _mm256_srai_epi32(
        _mm256_add_epi32(predh, _mm256_set1_epi32(1 << (bw_log2 - 1))),
        bw_log2);
    predh = _mm256_madd_epi16(_mm256_or_si256(predh, left_hi), s_left);
    predh = _mm256_srai_epi32(_mm256_add_epi32(predh, round_const), 6);
  }

  if (mode == SMOOTH_V_PRED) return predv;
  if (mode == SMOOTH_H_PRED) return predh;

  return _mm256_srai_epi32(
      _mm256_add_epi32(_mm256_add_epi32(predv, predh), _mm256_set1_epi32(1)),
      1);
}

static AVM_FORCE_INLINE void highbd_smooth_predictor_avx2(
    uint16_t *dst, ptrdiff_t stride, int bw, int bh, const uint16_t *above,
    const uint16_t *left, PREDICTION_MODE mode) {
  const int bw_log2 = get_msb(bw);
  const int bh_log2 = get_msb(bh);
  const __m128i scale =
      _mm_cvtsi32_si128(ROUND_POWER_OF_TWO(bh_log2 - 2 + bw_log2 - 2, 2));
  const int bl = left[bh];   // estimated by bottom-left pixel
  const int tr = above[bw];  // estimated by top-right pixel
  const __m256i bl_hi = _mm256_slli_epi32(_mm256_set1_epi32(bl), 16);
  const __m256i tr_hi = _mm256_slli_epi32(_mm256_set1_epi32(tr), 16);
  const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

  if (bw == 4) {
    const __m256i col = _mm256_and_si256(idx, _mm256_set1_epi32(3));
    const __m128i above4 = _mm_loadl_epi64((const __m128i *)above);
    const __m256i top =
        _mm256_cvtepu16_epi32(_mm_unpacklo_epi64(above4, above4));
    const __m256i top_bl = _mm256_or_si256(top, bl_hi);
    const __m256i top_hi = _mm256_slli_epi32(top, 16);
    const __m256i weights_h = highbd_smooth_dist_weights_avx2(col, bw);
    const __m256i s_left = highbd_smooth_blend_weights_avx2(col, scale);
    __m256i row = _mm256_srli_epi32(idx, 2);
    for (int r = 0; r < bh; r += 2) {
      __m128i left2 = xx_loadl_32(left + r);
      left2 = _mm_unpacklo_epi16(left2, left2);
      left2 = _mm_unpacklo_epi32(left2, left2);
      const __m256i left_row = _mm256_cvtepu16_epi32(left2);
      const __m256i pred = highbd_smooth_pred_avx2(
          mode, top_bl, top_hi, highbd_smooth_dist_weights_avx2(row, bh),
          highbd_smooth_blend_weights_avx2(row, scale),
          _mm256_or_si256(left_row, tr_hi), _mm256_slli_epi32(left_row, 16),
          weights_h, s_left, bh_log2, bw_log2);
      const __m128i pred16 = _mm_packus_epi32(
          _mm256_castsi256_si128(pred), _mm256_extracti128_si256(pred, 1));
      _mm_storel_epi64((__m128i *)dst, pred16);
      _mm_storeh_pd((double *)(dst + stride), _mm_castsi128_pd(pred16));
      row = _mm256_add_epi32(row, _mm256_set1_epi32(2));
      dst += 2 * stride;
    }
    return;
  }

  DECLARE_ALIGNED(32, int32_t, weights_v_tab[64]);
  DECLARE_ALIGNED(32, int32_t, s_top_tab[64]);
  DECLARE_ALIGNED(32, int32_t, left_tr_tab[64]);
  DECLARE_ALIGNED(32, int32_t, left_hi_tab[64]);
  for (int r = 0; r < bh; r += 8) {
    const __m256i row = _mm256_add_epi32(_mm256_set1_epi32(r), idx);
    if (mode != SMOOTH_H_PRED) {
      _mm256_store_si256((__m256i *)(weights_v_tab + r),
                         highbd_smooth_dist_weights_avx2(row, bh));
      _mm256_store_si256((__m256i *)(s_top_tab + r),
                         highbd_smooth_blend_weights_avx2(row, scale));
    }
    if (mode != SMOOTH_V_PRED) {
      const __m256i left_row = _mm256_cvtepu16_epi32(
          bh == 4 ? _mm_loadl_epi64((const __m128i *)left)
                  : _mm_loadu_si128((const __m128i *)(left + r)));
      _mm256_store_si256((__m256i *)(left_tr_tab + r),
                         _mm256_or_si256(left_row, tr_hi));
      _mm256_store_si256((__m256i *)(left_hi_tab + r),
                         _mm256_slli_epi32(left_row, 16));
    }
  }

  const __m256i zero = _mm256_setzero_si256();
  const int has_v = mode != SMOOTH_H_PRED;
  const int has_h = mode != SMOOTH_V_PRED;
  for (int c = 0; c < bw; c += 8) {
    const __m256i col = _mm256_add_epi32(_mm256_set1_epi32(c), idx);
    const __m256i top =
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(above + c)));
    const __m256i top_bl = _mm256_or_si256(top, bl_hi);
    const __m256i top_hi = _mm256_slli_epi32(top, 16);
    const __m256i weights_h = highbd_smooth_dist_weights_avx2(col, bw);
    const __m256i s_left = highbd_smooth_blend_weights_avx2(col, scale);
    uint16_t *dst_col = dst + c;
    for (int r = 0; r < bh; ++r) {
      const __m256i pred = highbd_smooth_pred_avx2(
          mode, top_bl, top_hi,
          has_v ? _mm256_set1_epi32(weights_v_tab[r]) : zero,
          has_v ? _mm256_set1_epi32(s_top_tab[r]) : zero,
          has_h ? _mm256_set1_epi32(left_tr_tab[r]) : zero,
          has_h ? _mm256_set1_epi32(left_hi_tab[r]) : zero, weights_h, s_left,
          bh_log2, bw_log2);
      _mm_storeu_si128((__m128i *)dst_col,
                       _mm_packus_epi32(_mm256_castsi256_si128(pred),
                                        _mm256_extracti128_si256(pred, 1)));
      dst_col += stride;
    }
  }
}

#define HIGHBD_SMOOTH_PRED_AVX2(type, mode, w, h)                       \
  void avm_highbd_##type##_predictor_##w##x##h##_avx2(                  \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,           \
      const uint16_t *left, int bd) {                                   \
    (void)bd;                                                           \
    highbd_smooth_predictor_avx2(dst, stride, w, h, above, left, mode); \
  }

#define HIGHBD_SMOOTH_PRED_AVX2_ALL_SIZES(type, mode) \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 4, 4)           \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 4, 8)           \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 4, 16)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 4, 32)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 4, 64)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 8, 4)           \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 8, 8)           \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 8, 16)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 8, 32)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 8, 64)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 16, 4)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 16, 8)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 16, 16)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 16, 32)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 16, 64)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 32, 4)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 32, 8)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 32, 16)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 32, 32)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 32, 64)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 64, 4)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 64, 8)          \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 64, 16)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 64, 32)         \
  HIGHBD_SMOOTH_PRED_AVX2(type, mode, 64, 64)

HIGHBD_SMOOTH_PRED_AVX2_ALL_SIZES(smooth, SMOOTH_PRED)
HIGHBD_SMOOTH_PRED_AVX2_ALL_SIZES(smooth_v, SMOOTH_V_PRED)
HIGHBD_SMOOTH_PRED_AVX2_ALL_SIZES(smooth_h, SMOOTH_H_PRED)

#undef HIGHBD_SMOOTH_PRED_AVX2_ALL_SIZES
#undef HIGHBD_SMOOTH_PRED_AVX2

static INLINE __m256i highbd_ibp_weight_pairs_avx2(__m256i weights) {
  return _mm256_or_si256(
      weights,
      _mm256_slli_epi32(
          _mm256_sub_epi32(_mm256_set1_epi32(IBP_WEIGHT_MAX), weights), 16));
}

static AVM_FORCE_INLINE void highbd_ibp_blend_16_avx2(
    uint16_t *dst, const uint16_t *second_pred, __m256i weights_lo,
    __m256i weights_hi) {
  const __m256i round_const = _mm256_set1_epi32(1 << (IBP_WEIGHT_SHIFT - 1));
  const __m256i dst_vec = _mm256_loadu_si256((const __m256i *)dst);
  const __m256i second_vec = _mm256_loadu_si256((const __m256i *)second_pred);
  __m256i lo =
      _mm256_madd_epi16(_mm256_unpacklo_epi16(dst_vec, second_vec), weights_lo);
  __m256i hi =
      _mm256_madd_epi16(_mm256_unpackhi_epi16(dst_vec, second_vec), weights_hi);
  lo = _mm256_srai_epi32(_mm256_add_epi32(lo, round_const), IBP_WEIGHT_SHIFT);
  hi = _mm256_srai_epi32(_mm256_add_epi32(hi, round_const), IBP_WEIGHT_SHIFT);
  _mm256_storeu_si256((__m256i *)dst, _mm256_packus_epi32(lo, hi));
}

static AVM_FORCE_INLINE __m128i highbd_ibp_blend_4_avx2(__m128i dst_second,
                                                        __m128i weights) {
  return _mm_srai_epi32(
      _mm_add_epi32(_mm_madd_epi16(dst_second, weights),
                    _mm_set1_epi32(1 << (IBP_WEIGHT_SHIFT - 1))),
      IBP_WEIGHT_SHIFT);
}

static AVM_FORCE_INLINE void highbd_ibp_dr_prediction_avx2(
    const IbpWeightsType weights[][IBP_WEIGHT_SIZE][DIR_MODES_0_90],
    int mode_idx, uint16_t *dst, ptrdiff_t stride, const uint16_t *second_pred,
    ptrdiff_t second_stride, int bw, int bh, int is_z3) {
  assert(bw >= 4 && bw <= 64 && bh >= 4 && bh <= 64);
  const int row_shift = bh >> (IBP_WEIGHT_SIZE_LOG2 + 1);
  const ptrdiff_t col_step =
      is_z3 ? IBP_WEIGHT_SIZE * DIR_MODES_0_90 : DIR_MODES_0_90;
  const ptrdiff_t row_step =
      is_z3 ? DIR_MODES_0_90 : IBP_WEIGHT_SIZE * DIR_MODES_0_90;
  const IbpWeightsType *weight_row = &weights[0][0][mode_idx];
  __m256i weight_pairs[8];
  __m128i weight_pairs_128[2];

  for (int r = 0; r < bh; ++r) {
    if ((r & ((1 << row_shift) - 1)) == 0) {
#define IBP_WEIGHT(i) ((uint8_t)weight_row[(i) * col_step])
      if (bw == 4) {
        weight_pairs_128[0] =
            _mm256_castsi256_si128(highbd_ibp_weight_pairs_avx2(
                _mm256_setr_epi32(IBP_WEIGHT(0), IBP_WEIGHT(1), IBP_WEIGHT(2),
                                  IBP_WEIGHT(3), 0, 0, 0, 0)));
      } else if (bw == 8) {
        const __m256i pairs = highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
            IBP_WEIGHT(0), IBP_WEIGHT(1), IBP_WEIGHT(2), IBP_WEIGHT(3),
            IBP_WEIGHT(4), IBP_WEIGHT(5), IBP_WEIGHT(6), IBP_WEIGHT(7)));
        weight_pairs_128[0] = _mm256_castsi256_si128(pairs);
        weight_pairs_128[1] = _mm256_extracti128_si256(pairs, 1);
      } else if (bw == 16) {
        weight_pairs[0] = highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
            IBP_WEIGHT(0), IBP_WEIGHT(1), IBP_WEIGHT(2), IBP_WEIGHT(3),
            IBP_WEIGHT(8), IBP_WEIGHT(9), IBP_WEIGHT(10), IBP_WEIGHT(11)));
        weight_pairs[1] = highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
            IBP_WEIGHT(4), IBP_WEIGHT(5), IBP_WEIGHT(6), IBP_WEIGHT(7),
            IBP_WEIGHT(12), IBP_WEIGHT(13), IBP_WEIGHT(14), IBP_WEIGHT(15)));
      } else {
        const __m256i pairs_lo = highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
            IBP_WEIGHT(0), IBP_WEIGHT(1), IBP_WEIGHT(2), IBP_WEIGHT(3),
            IBP_WEIGHT(4), IBP_WEIGHT(5), IBP_WEIGHT(6), IBP_WEIGHT(7)));
        const __m256i pairs_hi = highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
            IBP_WEIGHT(8), IBP_WEIGHT(9), IBP_WEIGHT(10), IBP_WEIGHT(11),
            IBP_WEIGHT(12), IBP_WEIGHT(13), IBP_WEIGHT(14), IBP_WEIGHT(15)));
        if (bw == 32) {
          // Column c uses weight c >> 1.
          weight_pairs[0] = _mm256_unpacklo_epi32(pairs_lo, pairs_lo);
          weight_pairs[1] = _mm256_unpackhi_epi32(pairs_lo, pairs_lo);
          weight_pairs[2] = _mm256_unpacklo_epi32(pairs_hi, pairs_hi);
          weight_pairs[3] = _mm256_unpackhi_epi32(pairs_hi, pairs_hi);
        } else {
          // Column c uses weight c >> 2.
          const __m256i idx_lo = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2);
          const __m256i idx_hi = _mm256_setr_epi32(1, 1, 1, 1, 3, 3, 3, 3);
          const __m256i four = _mm256_set1_epi32(4);
          weight_pairs[0] = _mm256_permutevar8x32_epi32(pairs_lo, idx_lo);
          weight_pairs[1] = _mm256_permutevar8x32_epi32(pairs_lo, idx_hi);
          weight_pairs[2] = _mm256_permutevar8x32_epi32(
              pairs_lo, _mm256_add_epi32(idx_lo, four));
          weight_pairs[3] = _mm256_permutevar8x32_epi32(
              pairs_lo, _mm256_add_epi32(idx_hi, four));
          weight_pairs[4] = _mm256_permutevar8x32_epi32(pairs_hi, idx_lo);
          weight_pairs[5] = _mm256_permutevar8x32_epi32(pairs_hi, idx_hi);
          weight_pairs[6] = _mm256_permutevar8x32_epi32(
              pairs_hi, _mm256_add_epi32(idx_lo, four));
          weight_pairs[7] = _mm256_permutevar8x32_epi32(
              pairs_hi, _mm256_add_epi32(idx_hi, four));
        }
      }
#undef IBP_WEIGHT
      weight_row += row_step;
    }
    if (bw >= 16) {
      for (int i = 0; i < bw / 16; ++i) {
        highbd_ibp_blend_16_avx2(dst + 16 * i, second_pred + 16 * i,
                                 weight_pairs[2 * i], weight_pairs[2 * i + 1]);
      }
    } else if (bw == 8) {
      const __m128i dst_vec = _mm_loadu_si128((const __m128i *)dst);
      const __m128i second_vec = _mm_loadu_si128((const __m128i *)second_pred);
      const __m128i lo = highbd_ibp_blend_4_avx2(
          _mm_unpacklo_epi16(dst_vec, second_vec), weight_pairs_128[0]);
      const __m128i hi = highbd_ibp_blend_4_avx2(
          _mm_unpackhi_epi16(dst_vec, second_vec), weight_pairs_128[1]);
      _mm_storeu_si128((__m128i *)dst, _mm_packus_epi32(lo, hi));
    } else {
      const __m128i dst_vec = _mm_loadl_epi64((const __m128i *)dst);
      const __m128i second_vec = _mm_loadl_epi64((const __m128i *)second_pred);
      const __m128i lo = highbd_ibp_blend_4_avx2(
          _mm_unpacklo_epi16(dst_vec, second_vec), weight_pairs_128[0]);
      _mm_storel_epi64((__m128i *)dst, _mm_packus_epi32(lo, lo));
    }
    dst += stride;
    second_pred += second_stride;
  }
}

#define HIGHBD_IBP_DR_PREDICTION_AVX2(is_z3)                                   \
  switch (bw) {                                                                \
    case 4:                                                                    \
      highbd_ibp_dr_prediction_avx2(weights, mode_idx, dst, stride,            \
                                    second_pred, second_stride, 4, bh, is_z3); \
      break;                                                                   \
    case 8:                                                                    \
      highbd_ibp_dr_prediction_avx2(weights, mode_idx, dst, stride,            \
                                    second_pred, second_stride, 8, bh, is_z3); \
      break;                                                                   \
    case 16:                                                                   \
      highbd_ibp_dr_prediction_avx2(weights, mode_idx, dst, stride,            \
                                    second_pred, second_stride, 16, bh,        \
                                    is_z3);                                    \
      break;                                                                   \
    case 32:                                                                   \
      highbd_ibp_dr_prediction_avx2(weights, mode_idx, dst, stride,            \
                                    second_pred, second_stride, 32, bh,        \
                                    is_z3);                                    \
      break;                                                                   \
    default:                                                                   \
      assert(bw == 64);                                                        \
      highbd_ibp_dr_prediction_avx2(weights, mode_idx, dst, stride,            \
                                    second_pred, second_stride, 64, bh,        \
                                    is_z3);                                    \
      break;                                                                   \
  }

void av2_highbd_ibp_dr_prediction_z1_avx2(
    const IbpWeightsType weights[][IBP_WEIGHT_SIZE][DIR_MODES_0_90],
    int mode_idx, uint16_t *dst, ptrdiff_t stride, uint16_t *second_pred,
    ptrdiff_t second_stride, int bw, int bh) {
  HIGHBD_IBP_DR_PREDICTION_AVX2(/*is_z3=*/0)
}

void av2_highbd_ibp_dr_prediction_z3_avx2(
    const IbpWeightsType weights[][IBP_WEIGHT_SIZE][DIR_MODES_0_90],
    int mode_idx, uint16_t *dst, ptrdiff_t stride, uint16_t *second_pred,
    ptrdiff_t second_stride, int bw, int bh) {
  HIGHBD_IBP_DR_PREDICTION_AVX2(/*is_z3=*/1)
}

#undef HIGHBD_IBP_DR_PREDICTION_AVX2

extern const uint8_t ibp_weights[5][16];

static AVM_FORCE_INLINE void highbd_fill_block_avx2(uint16_t *dst,
                                                    ptrdiff_t stride,
                                                    int block_w, int block_h,
                                                    uint16_t fill_val) {
  if (block_w == 4) {
    const __m128i fill_vec = _mm_set1_epi16((int16_t)fill_val);
    for (int row = 0; row < block_h; row += 4) {
      _mm_storel_epi64((__m128i *)dst, fill_vec);
      _mm_storel_epi64((__m128i *)(dst + stride), fill_vec);
      _mm_storel_epi64((__m128i *)(dst + 2 * stride), fill_vec);
      _mm_storel_epi64((__m128i *)(dst + 3 * stride), fill_vec);
      dst += 4 * stride;
    }
  } else if (block_w == 8) {
    const __m128i fill_vec = _mm_set1_epi16((int16_t)fill_val);
    for (int row = 0; row < block_h; row += 4) {
      _mm_storeu_si128((__m128i *)dst, fill_vec);
      _mm_storeu_si128((__m128i *)(dst + stride), fill_vec);
      _mm_storeu_si128((__m128i *)(dst + 2 * stride), fill_vec);
      _mm_storeu_si128((__m128i *)(dst + 3 * stride), fill_vec);
      dst += 4 * stride;
    }
  } else if (block_w == 16) {
    const __m256i fill_vec = _mm256_set1_epi16((int16_t)fill_val);
    for (int row = 0; row < block_h; row += 4) {
      _mm256_storeu_si256((__m256i *)dst, fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + stride), fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + 2 * stride), fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + 3 * stride), fill_vec);
      dst += 4 * stride;
    }
  } else if (block_w == 32) {
    const __m256i fill_vec = _mm256_set1_epi16((int16_t)fill_val);
    for (int row = 0; row < block_h; ++row, dst += stride) {
      _mm256_storeu_si256((__m256i *)dst, fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + 16), fill_vec);
    }
  } else {
    const __m256i fill_vec = _mm256_set1_epi16((int16_t)fill_val);
    for (int row = 0; row < block_h; ++row, dst += stride) {
      _mm256_storeu_si256((__m256i *)dst, fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + 16), fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + 32), fill_vec);
      _mm256_storeu_si256((__m256i *)(dst + 48), fill_vec);
    }
  }
}

static AVM_FORCE_INLINE uint32_t highbd_sum_edge_avx2(const uint16_t *edge,
                                                      int num_px) {
  if (num_px == 4) {
    return (uint32_t)edge[0] + edge[1] + edge[2] + edge[3];
  }
  if (num_px == 8) {
    __m128i sum_128 = _mm_madd_epi16(_mm_loadu_si128((const __m128i *)edge),
                                     _mm_set1_epi16(1));
    sum_128 = _mm_add_epi32(sum_128, _mm_bsrli_si128(sum_128, 8));
    sum_128 = _mm_add_epi32(sum_128, _mm_bsrli_si128(sum_128, 4));
    return (uint32_t)_mm_cvtsi128_si32(sum_128);
  }
  const __m256i ones_vec = _mm256_set1_epi16(1);
  __m256i sum_256 =
      _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)edge), ones_vec);
  if (num_px >= 32) {
    sum_256 = _mm256_add_epi32(
        sum_256,
        _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(edge + 16)),
                          ones_vec));
  }
  if (num_px == 64) {
    sum_256 = _mm256_add_epi32(
        sum_256,
        _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(edge + 32)),
                          ones_vec));
    sum_256 = _mm256_add_epi32(
        sum_256,
        _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(edge + 48)),
                          ones_vec));
  }
  __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(sum_256),
                                  _mm256_extracti128_si256(sum_256, 1));
  sum_128 = _mm_add_epi32(sum_128, _mm_bsrli_si128(sum_128, 8));
  sum_128 = _mm_add_epi32(sum_128, _mm_bsrli_si128(sum_128, 4));
  return (uint32_t)_mm_cvtsi128_si32(sum_128);
}

static AVM_FORCE_INLINE void highbd_v_predictor_avx2(uint16_t *dst,
                                                     ptrdiff_t stride,
                                                     int block_w, int block_h,
                                                     const uint16_t *above) {
  if (block_w == 4) {
    const __m128i above_vec = _mm_loadl_epi64((const __m128i *)above);
    for (int row = 0; row < block_h; row += 4) {
      _mm_storel_epi64((__m128i *)dst, above_vec);
      _mm_storel_epi64((__m128i *)(dst + stride), above_vec);
      _mm_storel_epi64((__m128i *)(dst + 2 * stride), above_vec);
      _mm_storel_epi64((__m128i *)(dst + 3 * stride), above_vec);
      dst += 4 * stride;
    }
  } else if (block_w == 8) {
    const __m128i above_vec = _mm_loadu_si128((const __m128i *)above);
    for (int row = 0; row < block_h; row += 4) {
      _mm_storeu_si128((__m128i *)dst, above_vec);
      _mm_storeu_si128((__m128i *)(dst + stride), above_vec);
      _mm_storeu_si128((__m128i *)(dst + 2 * stride), above_vec);
      _mm_storeu_si128((__m128i *)(dst + 3 * stride), above_vec);
      dst += 4 * stride;
    }
  } else if (block_w == 16) {
    const __m256i above_vec = _mm256_loadu_si256((const __m256i *)above);
    for (int row = 0; row < block_h; row += 4) {
      _mm256_storeu_si256((__m256i *)dst, above_vec);
      _mm256_storeu_si256((__m256i *)(dst + stride), above_vec);
      _mm256_storeu_si256((__m256i *)(dst + 2 * stride), above_vec);
      _mm256_storeu_si256((__m256i *)(dst + 3 * stride), above_vec);
      dst += 4 * stride;
    }
  } else if (block_w == 32) {
    const __m256i above_vec0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i above_vec1 =
        _mm256_loadu_si256((const __m256i *)(above + 16));
    for (int row = 0; row < block_h; ++row, dst += stride) {
      _mm256_storeu_si256((__m256i *)dst, above_vec0);
      _mm256_storeu_si256((__m256i *)(dst + 16), above_vec1);
    }
  } else {
    const __m256i above_vec0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i above_vec1 =
        _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i above_vec2 =
        _mm256_loadu_si256((const __m256i *)(above + 32));
    const __m256i above_vec3 =
        _mm256_loadu_si256((const __m256i *)(above + 48));
    for (int row = 0; row < block_h; ++row, dst += stride) {
      _mm256_storeu_si256((__m256i *)dst, above_vec0);
      _mm256_storeu_si256((__m256i *)(dst + 16), above_vec1);
      _mm256_storeu_si256((__m256i *)(dst + 32), above_vec2);
      _mm256_storeu_si256((__m256i *)(dst + 48), above_vec3);
    }
  }
}

static AVM_FORCE_INLINE void highbd_h_predictor_avx2(uint16_t *dst,
                                                     ptrdiff_t stride,
                                                     int block_w, int block_h,
                                                     const uint16_t *left) {
  if (block_w == 4) {
    for (int row = 0; row < block_h; row += 2, dst += 2 * stride) {
      _mm_storel_epi64((__m128i *)dst, _mm_set1_epi16((int16_t)left[row]));
      _mm_storel_epi64((__m128i *)(dst + stride),
                       _mm_set1_epi16((int16_t)left[row + 1]));
    }
  } else if (block_w == 8) {
    for (int row = 0; row < block_h; row += 2, dst += 2 * stride) {
      _mm_storeu_si128((__m128i *)dst, _mm_set1_epi16((int16_t)left[row]));
      _mm_storeu_si128((__m128i *)(dst + stride),
                       _mm_set1_epi16((int16_t)left[row + 1]));
    }
  } else if (block_w == 16) {
    for (int row = 0; row < block_h; ++row, dst += stride) {
      _mm256_storeu_si256((__m256i *)dst,
                          _mm256_set1_epi16((int16_t)left[row]));
    }
  } else if (block_w == 32) {
    for (int row = 0; row < block_h; ++row, dst += stride) {
      const __m256i left_vec = _mm256_set1_epi16((int16_t)left[row]);
      _mm256_storeu_si256((__m256i *)dst, left_vec);
      _mm256_storeu_si256((__m256i *)(dst + 16), left_vec);
    }
  } else {
    for (int row = 0; row < block_h; ++row, dst += stride) {
      const __m256i left_vec = _mm256_set1_epi16((int16_t)left[row]);
      _mm256_storeu_si256((__m256i *)dst, left_vec);
      _mm256_storeu_si256((__m256i *)(dst + 16), left_vec);
      _mm256_storeu_si256((__m256i *)(dst + 32), left_vec);
      _mm256_storeu_si256((__m256i *)(dst + 48), left_vec);
    }
  }
}

static AVM_FORCE_INLINE __m256i highbd_paeth_select_16_avx2(
    __m256i top_vec, __m256i top_left_vec, __m256i top_minus_tl,
    __m256i dist_left, __m256i left_vec, __m256i left_minus_tl,
    __m256i dist_top) {
  const __m256i dist_top_left =
      _mm256_abs_epi16(_mm256_add_epi16(top_minus_tl, left_minus_tl));
  const __m256i nearest_top_or_tl = _mm256_blendv_epi8(
      top_vec, top_left_vec, _mm256_cmpgt_epi16(dist_top, dist_top_left));
  return _mm256_blendv_epi8(
      left_vec, nearest_top_or_tl,
      _mm256_cmpgt_epi16(dist_left, _mm256_min_epi16(dist_top, dist_top_left)));
}

static AVM_FORCE_INLINE void highbd_paeth_predictor_avx2(
    uint16_t *dst, ptrdiff_t stride, int block_w, int block_h,
    const uint16_t *above, const uint16_t *left) {
  const int top_left_scalar = (int)above[-1];
  if (block_w <= 8) {
    const __m128i top_left_vec = _mm_set1_epi16((int16_t)top_left_scalar);
    const __m128i top_vec = (block_w == 4)
                                ? _mm_loadl_epi64((const __m128i *)above)
                                : _mm_loadu_si128((const __m128i *)above);
    const __m128i top_minus_tl = _mm_sub_epi16(top_vec, top_left_vec);
    const __m128i dist_left = _mm_abs_epi16(top_minus_tl);
    for (int row = 0; row < block_h; ++row, dst += stride) {
      const int left_scalar = (int)left[row];
      const int left_minus_tl_scalar = left_scalar - top_left_scalar;
      const __m128i left_vec = _mm_set1_epi16((int16_t)left_scalar);
      const __m128i left_minus_tl =
          _mm_set1_epi16((int16_t)left_minus_tl_scalar);
      const __m128i dist_top = _mm_set1_epi16(
          (int16_t)(left_minus_tl_scalar < 0 ? -left_minus_tl_scalar
                                             : left_minus_tl_scalar));
      const __m128i dist_top_left =
          _mm_abs_epi16(_mm_add_epi16(top_minus_tl, left_minus_tl));
      const __m128i nearest_top_or_tl = _mm_blendv_epi8(
          top_vec, top_left_vec, _mm_cmpgt_epi16(dist_top, dist_top_left));
      const __m128i pred_row = _mm_blendv_epi8(
          left_vec, nearest_top_or_tl,
          _mm_cmpgt_epi16(dist_left, _mm_min_epi16(dist_top, dist_top_left)));
      if (block_w == 4) {
        _mm_storel_epi64((__m128i *)dst, pred_row);
      } else {
        _mm_storeu_si128((__m128i *)dst, pred_row);
      }
    }
    return;
  }

  const __m256i top_left_vec = _mm256_set1_epi16((int16_t)top_left_scalar);
  __m256i top_vec[4], top_minus_tl[4], dist_left[4];
  const int num_chunks = block_w >> 4;
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    top_vec[chunk] = _mm256_loadu_si256((const __m256i *)(above + 16 * chunk));
    top_minus_tl[chunk] = _mm256_sub_epi16(top_vec[chunk], top_left_vec);
    dist_left[chunk] = _mm256_abs_epi16(top_minus_tl[chunk]);
  }
  for (int row = 0; row < block_h; ++row, dst += stride) {
    const int left_scalar = (int)left[row];
    const int left_minus_tl_scalar = left_scalar - top_left_scalar;
    const __m256i left_vec = _mm256_set1_epi16((int16_t)left_scalar);
    const __m256i left_minus_tl =
        _mm256_set1_epi16((int16_t)left_minus_tl_scalar);
    const __m256i dist_top = _mm256_set1_epi16(
        (int16_t)(left_minus_tl_scalar < 0 ? -left_minus_tl_scalar
                                           : left_minus_tl_scalar));
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      _mm256_storeu_si256(
          (__m256i *)(dst + 16 * chunk),
          highbd_paeth_select_16_avx2(top_vec[chunk], top_left_vec,
                                      top_minus_tl[chunk], dist_left[chunk],
                                      left_vec, left_minus_tl, dist_top));
    }
  }
}

static AVM_FORCE_INLINE void highbd_ibp_blend_8_sse41(uint16_t *dst,
                                                      __m128i ref_vec,
                                                      __m128i weight_pairs_lo,
                                                      __m128i weight_pairs_hi) {
  const __m128i dst_vec = _mm_loadu_si128((const __m128i *)dst);
  const __m128i blended_lo = highbd_ibp_blend_4_avx2(
      _mm_unpacklo_epi16(dst_vec, ref_vec), weight_pairs_lo);
  const __m128i blended_hi = highbd_ibp_blend_4_avx2(
      _mm_unpackhi_epi16(dst_vec, ref_vec), weight_pairs_hi);
  _mm_storeu_si128((__m128i *)dst, _mm_packus_epi32(blended_lo, blended_hi));
}

static AVM_FORCE_INLINE void highbd_ibp_dc_blend_left_avx2(
    uint16_t *dst, ptrdiff_t stride, const uint16_t *left, int block_w,
    int row_start, int block_h) {
  const uint8_t *col_weights = ibp_weights[get_msb(block_w) - 2];
  if (block_w == 4) {
    const int dst_weight = col_weights[0];
    const int left_weight = IBP_WEIGHT_REF - dst_weight;
    for (int row = row_start; row < block_h; ++row, dst += stride) {
      dst[0] = (uint16_t)ROUND_POWER_OF_TWO(
          left[row] * left_weight + dst[0] * dst_weight, IBP_WEIGHT_SHIFT);
    }
  } else if (block_w == 8) {
    const __m128i weight_pairs = _mm_setr_epi16(
        col_weights[0], IBP_WEIGHT_REF - col_weights[0], col_weights[1],
        IBP_WEIGHT_REF - col_weights[1], 0, 0, 0, 0);
    for (int row = row_start; row < block_h; ++row, dst += stride) {
      const __m128i dst_pixels = xx_loadl_32(dst);
      const __m128i left_pixel_vec = _mm_set1_epi16((int16_t)left[row]);
      const __m128i blended_32 = highbd_ibp_blend_4_avx2(
          _mm_unpacklo_epi16(dst_pixels, left_pixel_vec), weight_pairs);
      xx_storel_32(dst, _mm_packus_epi32(blended_32, blended_32));
    }
  } else if (block_w == 16) {
    const __m128i weight_pairs =
        _mm256_castsi256_si128(highbd_ibp_weight_pairs_avx2(
            _mm256_setr_epi32(col_weights[0], col_weights[1], col_weights[2],
                              col_weights[3], 0, 0, 0, 0)));
    for (int row = row_start; row < block_h; ++row, dst += stride) {
      const __m128i dst_pixels = _mm_loadl_epi64((const __m128i *)dst);
      const __m128i left_pixel_vec = _mm_set1_epi16((int16_t)left[row]);
      const __m128i blended_32 = highbd_ibp_blend_4_avx2(
          _mm_unpacklo_epi16(dst_pixels, left_pixel_vec), weight_pairs);
      _mm_storel_epi64((__m128i *)dst,
                       _mm_packus_epi32(blended_32, blended_32));
    }
  } else if (block_w == 32) {
    const __m256i weight_pairs = highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
        col_weights[0], col_weights[1], col_weights[2], col_weights[3],
        col_weights[4], col_weights[5], col_weights[6], col_weights[7]));
    const __m128i weight_pairs_lo = _mm256_castsi256_si128(weight_pairs);
    const __m128i weight_pairs_hi = _mm256_extracti128_si256(weight_pairs, 1);
    for (int row = row_start; row < block_h; ++row, dst += stride) {
      highbd_ibp_blend_8_sse41(dst, _mm_set1_epi16((int16_t)left[row]),
                               weight_pairs_lo, weight_pairs_hi);
    }
  } else {
    const __m256i weight_pairs_lo =
        highbd_ibp_weight_pairs_avx2(_mm256_setr_epi32(
            col_weights[0], col_weights[1], col_weights[2], col_weights[3],
            col_weights[8], col_weights[9], col_weights[10], col_weights[11]));
    const __m256i weight_pairs_hi = highbd_ibp_weight_pairs_avx2(
        _mm256_setr_epi32(col_weights[4], col_weights[5], col_weights[6],
                          col_weights[7], col_weights[12], col_weights[13],
                          col_weights[14], col_weights[15]));
    const __m256i round_const = _mm256_set1_epi32(1 << (IBP_WEIGHT_SHIFT - 1));
    for (int row = row_start; row < block_h; ++row, dst += stride) {
      const __m256i dst_pixels = _mm256_loadu_si256((const __m256i *)dst);
      const __m256i left_pixel_vec = _mm256_set1_epi16((int16_t)left[row]);
      __m256i blended_lo = _mm256_madd_epi16(
          _mm256_unpacklo_epi16(dst_pixels, left_pixel_vec), weight_pairs_lo);
      __m256i blended_hi = _mm256_madd_epi16(
          _mm256_unpackhi_epi16(dst_pixels, left_pixel_vec), weight_pairs_hi);
      blended_lo = _mm256_srai_epi32(_mm256_add_epi32(blended_lo, round_const),
                                     IBP_WEIGHT_SHIFT);
      blended_hi = _mm256_srai_epi32(_mm256_add_epi32(blended_hi, round_const),
                                     IBP_WEIGHT_SHIFT);
      _mm256_storeu_si256((__m256i *)dst,
                          _mm256_packus_epi32(blended_lo, blended_hi));
    }
  }
}

static AVM_FORCE_INLINE void highbd_ibp_dc_blend_top_avx2(
    uint16_t *dst, ptrdiff_t stride, const uint16_t *above, int block_w,
    int block_h, int col_start) {
  const int num_blend_rows = block_h >> 2;
  const uint8_t *row_weights = ibp_weights[get_msb(block_h) - 2];
  if (block_w == 4) {
    const __m128i above_pixels = _mm_loadl_epi64((const __m128i *)above);
    for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
      const int dst_weight = row_weights[row];
      const __m128i weight_pairs =
          _mm_set1_epi32(dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16));
      const __m128i dst_pixels = _mm_loadl_epi64((const __m128i *)dst);
      __m128i blended_row = _mm_packus_epi32(
          highbd_ibp_blend_4_avx2(_mm_unpacklo_epi16(dst_pixels, above_pixels),
                                  weight_pairs),
          dst_pixels);
      if (col_start) blended_row = _mm_blend_epi16(blended_row, dst_pixels, 1);
      _mm_storel_epi64((__m128i *)dst, blended_row);
    }
  } else if (block_w == 8) {
    const __m128i above_pixels = _mm_loadu_si128((const __m128i *)above);
    for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
      const int dst_weight = row_weights[row];
      const __m128i weight_pairs =
          _mm_set1_epi32(dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16));
      const __m128i dst_pixels = _mm_loadu_si128((const __m128i *)dst);
      const __m128i blended_lo = highbd_ibp_blend_4_avx2(
          _mm_unpacklo_epi16(dst_pixels, above_pixels), weight_pairs);
      const __m128i blended_hi = highbd_ibp_blend_4_avx2(
          _mm_unpackhi_epi16(dst_pixels, above_pixels), weight_pairs);
      __m128i blended_row = _mm_packus_epi32(blended_lo, blended_hi);
      if (col_start) blended_row = _mm_blend_epi16(blended_row, dst_pixels, 3);
      _mm_storeu_si128((__m128i *)dst, blended_row);
    }
  } else if (block_w == 16) {
    if (col_start == 0) {
      for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
        const int dst_weight = row_weights[row];
        const __m256i weight_pairs = _mm256_set1_epi32(
            dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16));
        highbd_ibp_blend_16_avx2(dst, above, weight_pairs, weight_pairs);
      }
    } else {
      const __m128i above_cols_4_7 =
          _mm_loadl_epi64((const __m128i *)(above + 4));
      const __m128i above_cols_8_15 =
          _mm_loadu_si128((const __m128i *)(above + 8));
      for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
        const int dst_weight = row_weights[row];
        const __m128i weight_pairs =
            _mm_set1_epi32(dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16));
        const __m128i dst_cols_4_7 =
            _mm_loadl_epi64((const __m128i *)(dst + 4));
        const __m128i blended_cols_4_7 = highbd_ibp_blend_4_avx2(
            _mm_unpacklo_epi16(dst_cols_4_7, above_cols_4_7), weight_pairs);
        _mm_storel_epi64((__m128i *)(dst + 4),
                         _mm_packus_epi32(blended_cols_4_7, blended_cols_4_7));
        highbd_ibp_blend_8_sse41(dst + 8, above_cols_8_15, weight_pairs,
                                 weight_pairs);
      }
    }
  } else if (block_w == 32) {
    if (col_start == 0) {
      for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
        const int dst_weight = row_weights[row];
        const __m256i weight_pairs = _mm256_set1_epi32(
            dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16));
        highbd_ibp_blend_16_avx2(dst, above, weight_pairs, weight_pairs);
        highbd_ibp_blend_16_avx2(dst + 16, above + 16, weight_pairs,
                                 weight_pairs);
      }
    } else {
      const __m128i above_cols_8_15 =
          _mm_loadu_si128((const __m128i *)(above + 8));
      for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
        const int dst_weight = row_weights[row];
        const int32_t packed_weight_pair =
            dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16);
        const __m128i weight_pairs_128 = _mm_set1_epi32(packed_weight_pair);
        const __m256i weight_pairs_256 = _mm256_set1_epi32(packed_weight_pair);
        highbd_ibp_blend_8_sse41(dst + 8, above_cols_8_15, weight_pairs_128,
                                 weight_pairs_128);
        highbd_ibp_blend_16_avx2(dst + 16, above + 16, weight_pairs_256,
                                 weight_pairs_256);
      }
    }
  } else {
    for (int row = 0; row < num_blend_rows; ++row, dst += stride) {
      const int dst_weight = row_weights[row];
      const __m256i weight_pairs =
          _mm256_set1_epi32(dst_weight | ((IBP_WEIGHT_REF - dst_weight) << 16));
      highbd_ibp_blend_16_avx2(dst, above, weight_pairs, weight_pairs);
      highbd_ibp_blend_16_avx2(dst + 16, above + 16, weight_pairs,
                               weight_pairs);
      highbd_ibp_blend_16_avx2(dst + 32, above + 32, weight_pairs,
                               weight_pairs);
      highbd_ibp_blend_16_avx2(dst + 48, above + 48, weight_pairs,
                               weight_pairs);
    }
  }
}

#define DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(w, h)                            \
  void avm_highbd_v_predictor_##w##x##h##_avx2(                              \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)left;                                                              \
    (void)bd;                                                                \
    highbd_v_predictor_avx2(dst, stride, w, h, above);                       \
  }                                                                          \
  void avm_highbd_h_predictor_##w##x##h##_avx2(                              \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)above;                                                             \
    (void)bd;                                                                \
    highbd_h_predictor_avx2(dst, stride, w, h, left);                        \
  }                                                                          \
  void avm_highbd_paeth_predictor_##w##x##h##_avx2(                          \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)bd;                                                                \
    highbd_paeth_predictor_avx2(dst, stride, w, h, above, left);             \
  }                                                                          \
  void avm_highbd_dc_128_predictor_##w##x##h##_avx2(                         \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)above;                                                             \
    (void)left;                                                              \
    highbd_fill_block_avx2(dst, stride, w, h, (uint16_t)(128 << (bd - 8)));  \
  }                                                                          \
  void avm_highbd_dc_top_predictor_##w##x##h##_avx2(                         \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)left;                                                              \
    (void)bd;                                                                \
    const uint32_t edge_sum = highbd_sum_edge_avx2(above, w);                \
    const uint16_t dc_val = (uint16_t)((edge_sum + ((w) >> 1)) / (w));       \
    highbd_fill_block_avx2(dst, stride, w, h, dc_val);                       \
  }                                                                          \
  void avm_highbd_dc_left_predictor_##w##x##h##_avx2(                        \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)above;                                                             \
    (void)bd;                                                                \
    const uint32_t edge_sum = highbd_sum_edge_avx2(left, h);                 \
    const uint16_t dc_val = (uint16_t)((edge_sum + ((h) >> 1)) / (h));       \
    highbd_fill_block_avx2(dst, stride, w, h, dc_val);                       \
  }                                                                          \
  void avm_highbd_dc_predictor_##w##x##h##_avx2(                             \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    const uint32_t edge_sum =                                                \
        highbd_sum_edge_avx2(above, w) + highbd_sum_edge_avx2(left, h);      \
    uint16_t dc_val;                                                         \
    if ((w) == (h)) {                                                        \
      (void)bd;                                                              \
      const uint32_t num_edge_px = (w) + (h);                                \
      dc_val = (uint16_t)((edge_sum + (num_edge_px >> 1)) / num_edge_px);    \
    } else {                                                                 \
      int16_t inv_count_shift = 0;                                           \
      const uint32_t inv_count_scale =                                       \
          (uint16_t)resolve_divisor_32((w) + (h), &inv_count_shift);         \
      const uint32_t round_offset = (1u << inv_count_shift) >> 1;            \
      dc_val = clip_pixel_highbd(                                            \
          (int)((edge_sum * inv_count_scale + round_offset) >>               \
                inv_count_shift),                                            \
          bd);                                                               \
    }                                                                        \
    highbd_fill_block_avx2(dst, stride, w, h, dc_val);                       \
  }                                                                          \
  void avm_highbd_ibp_dc_left_predictor_##w##x##h##_avx2(                    \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)above;                                                             \
    (void)bd;                                                                \
    highbd_ibp_dc_blend_left_avx2(dst, stride, left, w, 0, h);               \
  }                                                                          \
  void avm_highbd_ibp_dc_top_predictor_##w##x##h##_avx2(                     \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)left;                                                              \
    (void)bd;                                                                \
    highbd_ibp_dc_blend_top_avx2(dst, stride, above, w, h, 0);               \
  }                                                                          \
  void avm_highbd_ibp_dc_predictor_##w##x##h##_avx2(                         \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,                \
      const uint16_t *left, int bd) {                                        \
    (void)bd;                                                                \
    const int row_start = ((w) >= (h)) ? ((h) >> 2) : 0;                     \
    const int col_start = ((w) >= (h)) ? 0 : ((w) >> 2);                     \
    highbd_ibp_dc_blend_top_avx2(dst, stride, above, w, h, col_start);       \
    highbd_ibp_dc_blend_left_avx2(dst + row_start * stride, stride, left, w, \
                                  row_start, h);                             \
  }

DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(4, 4)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(4, 8)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(4, 16)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(4, 32)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(4, 64)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(8, 4)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(8, 8)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(8, 16)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(8, 32)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(8, 64)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(16, 4)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(16, 8)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(16, 16)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(16, 32)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(16, 64)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(32, 4)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(32, 8)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(32, 16)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(32, 32)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(32, 64)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(64, 4)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(64, 8)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(64, 16)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(64, 32)
DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2(64, 64)

#undef DEFINE_HIGHBD_INTRA_PREDICTORS_AVX2
