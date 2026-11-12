/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <assert.h>
#include <immintrin.h>

#include "config/av2_rtcd.h"
#include "config/avm_config.h"

// Advance 16-sample window by 1 sample, filling lane 15 from right_pad.
static AVM_FORCE_INLINE __m256i shift_edge_next1_avx2(__m256i samples,
                                                      __m256i right_pad) {
  const __m256i cross_lane =
      _mm256_permute2x128_si256(samples, right_pad, 0x21);
  return _mm256_alignr_epi8(cross_lane, samples, 2);
}

// Advance 16-sample window by 2 samples, filling lanes 14..15 from right_pad.
static AVM_FORCE_INLINE __m256i shift_edge_next2_avx2(__m256i samples,
                                                      __m256i right_pad) {
  const __m256i cross_lane =
      _mm256_permute2x128_si256(samples, right_pad, 0x21);
  return _mm256_alignr_epi8(cross_lane, samples, 4);
}

// Retreat 16-sample window by 1 sample, filling lane 0 from left_pad.
static AVM_FORCE_INLINE __m256i shift_edge_prev1_avx2(__m256i samples,
                                                      __m256i left_pad) {
  const __m256i cross_lane = _mm256_permute2x128_si256(left_pad, samples, 0x20);
  return _mm256_alignr_epi8(samples, cross_lane, 14);
}

// Strength-1 3-tap [4, 8, 4]/16 = [1, 2, 1]/4 filter.
static AVM_FORCE_INLINE __m256i filter_edge_3tap_s1_avx2(__m256i prev,
                                                         __m256i cur,
                                                         __m256i next,
                                                         __m256i round_const) {
  const __m256i weighted_sum =
      _mm256_add_epi16(_mm256_add_epi16(prev, next), _mm256_slli_epi16(cur, 1));
  return _mm256_srli_epi16(_mm256_add_epi16(weighted_sum, round_const), 2);
}

static AVM_FORCE_INLINE __m128i filter_edge_3tap_s1_sse2(__m128i prev,
                                                         __m128i cur,
                                                         __m128i next,
                                                         __m128i round_const) {
  const __m128i weighted_sum =
      _mm_add_epi16(_mm_add_epi16(prev, next), _mm_slli_epi16(cur, 1));
  return _mm_srli_epi16(_mm_add_epi16(weighted_sum, round_const), 2);
}

// Strength-2 3-tap [5, 6, 5]/16 filter.
static AVM_FORCE_INLINE __m256i filter_edge_3tap_s2_avx2(__m256i prev,
                                                         __m256i cur,
                                                         __m256i next,
                                                         __m256i round_const) {
  const __m256i outer_sum = _mm256_add_epi16(prev, next);
  const __m256i sum_all_x4 =
      _mm256_slli_epi16(_mm256_add_epi16(outer_sum, cur), 2);
  const __m256i cur_x2_plus_rnd =
      _mm256_add_epi16(_mm256_slli_epi16(cur, 1), round_const);
  const __m256i weighted_sum = _mm256_add_epi16(
      _mm256_add_epi16(outer_sum, sum_all_x4), cur_x2_plus_rnd);
  return _mm256_srli_epi16(weighted_sum, 4);
}

static AVM_FORCE_INLINE __m128i filter_edge_3tap_s2_sse2(__m128i prev,
                                                         __m128i cur,
                                                         __m128i next,
                                                         __m128i round_const) {
  const __m128i outer_sum = _mm_add_epi16(prev, next);
  const __m128i sum_all_x4 = _mm_slli_epi16(_mm_add_epi16(outer_sum, cur), 2);
  const __m128i cur_x2_plus_rnd =
      _mm_add_epi16(_mm_slli_epi16(cur, 1), round_const);
  const __m128i weighted_sum =
      _mm_add_epi16(_mm_add_epi16(outer_sum, sum_all_x4), cur_x2_plus_rnd);
  return _mm_srli_epi16(weighted_sum, 4);
}

// Strength-3 5-tap [2, 4, 4, 4, 2]/16 = [1, 2, 2, 2, 1]/8 filter.
static AVM_FORCE_INLINE __m256i
filter_edge_5tap_s3_avx2(__m256i prev2, __m256i prev1, __m256i cur,
                         __m256i next1, __m256i next2, __m256i round_const) {
  const __m256i outer_sum = _mm256_add_epi16(prev2, next2);
  const __m256i inner_sum_x2 = _mm256_slli_epi16(
      _mm256_add_epi16(_mm256_add_epi16(prev1, cur), next1), 1);
  const __m256i weighted_sum =
      _mm256_add_epi16(_mm256_add_epi16(outer_sum, inner_sum_x2), round_const);
  return _mm256_srli_epi16(weighted_sum, 3);
}

static AVM_FORCE_INLINE __m128i
filter_edge_5tap_s3_sse2(__m128i prev2, __m128i prev1, __m128i cur,
                         __m128i next1, __m128i next2, __m128i round_const) {
  const __m128i outer_sum = _mm_add_epi16(prev2, next2);
  const __m128i inner_sum_x2 =
      _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(prev1, cur), next1), 1);
  const __m128i weighted_sum =
      _mm_add_epi16(_mm_add_epi16(outer_sum, inner_sum_x2), round_const);
  return _mm_srli_epi16(weighted_sum, 3);
}

void av2_filter_intra_edge_high_avx2(uint16_t *edge, int num_px, int strength) {
  if (!strength || num_px <= 1) return;

  uint16_t *const dst_edge = edge + 1;
  const int num_filtered = num_px - 1;
  const uint16_t first_sample = edge[0];
  const uint16_t last_sample = edge[num_px - 1];

  if (strength == 1) {
    if (num_filtered >= 16) {
      const __m256i round_const = _mm256_set1_epi16(2);
      const __m256i last_vec = _mm256_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 16;
      const __m256i tail_prev =
          _mm256_loadu_si256((const __m256i *)(edge + tail_offset));
      const __m256i tail_cur =
          _mm256_loadu_si256((const __m256i *)(edge + tail_offset + 1));
      const __m256i tail_next = shift_edge_next1_avx2(tail_cur, last_vec);
      const __m256i filtered_tail =
          filter_edge_3tap_s1_avx2(tail_prev, tail_cur, tail_next, round_const);
      if (tail_offset == 0) {
        _mm256_storeu_si256((__m256i *)dst_edge, filtered_tail);
        return;
      }
      __m256i filtered_cur = filter_edge_3tap_s1_avx2(
          _mm256_loadu_si256((const __m256i *)edge),
          _mm256_loadu_si256((const __m256i *)(edge + 1)),
          _mm256_loadu_si256((const __m256i *)(edge + 2)), round_const);
      int offset = 0;
      for (; offset + 16 < tail_offset; offset += 16) {
        const __m256i filtered_next = filter_edge_3tap_s1_avx2(
            _mm256_loadu_si256((const __m256i *)(edge + offset + 16)),
            _mm256_loadu_si256((const __m256i *)(edge + offset + 17)),
            _mm256_loadu_si256((const __m256i *)(edge + offset + 18)),
            round_const);
        _mm256_storeu_si256((__m256i *)(dst_edge + offset), filtered_cur);
        filtered_cur = filtered_next;
      }
      _mm256_storeu_si256((__m256i *)(dst_edge + offset), filtered_cur);
      _mm256_storeu_si256((__m256i *)(dst_edge + tail_offset), filtered_tail);
    } else if (num_filtered >= 8) {
      const __m128i round_const = _mm_set1_epi16(2);
      const __m128i last_vec = _mm_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 8;
      const __m128i tail_prev =
          _mm_loadu_si128((const __m128i *)(edge + tail_offset));
      const __m128i tail_cur =
          _mm_loadu_si128((const __m128i *)(edge + tail_offset + 1));
      const __m128i tail_next = _mm_alignr_epi8(last_vec, tail_cur, 2);
      const __m128i filtered_tail =
          filter_edge_3tap_s1_sse2(tail_prev, tail_cur, tail_next, round_const);
      if (tail_offset > 0) {
        const __m128i filtered_head = filter_edge_3tap_s1_sse2(
            _mm_loadu_si128((const __m128i *)edge),
            _mm_loadu_si128((const __m128i *)(edge + 1)),
            _mm_loadu_si128((const __m128i *)(edge + 2)), round_const);
        _mm_storeu_si128((__m128i *)dst_edge, filtered_head);
      }
      _mm_storeu_si128((__m128i *)(dst_edge + tail_offset), filtered_tail);
    } else if (num_filtered >= 4) {
      const __m128i round_const = _mm_set1_epi16(2);
      const __m128i last_vec = _mm_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 4;
      const __m128i tail_prev =
          _mm_loadl_epi64((const __m128i *)(edge + tail_offset));
      const __m128i tail_cur =
          _mm_loadl_epi64((const __m128i *)(edge + tail_offset + 1));
      const __m128i tail_next =
          _mm_alignr_epi8(last_vec, _mm_unpacklo_epi64(last_vec, tail_cur), 10);
      const __m128i filtered_tail =
          filter_edge_3tap_s1_sse2(tail_prev, tail_cur, tail_next, round_const);
      if (tail_offset > 0) {
        const __m128i filtered_head = filter_edge_3tap_s1_sse2(
            _mm_loadl_epi64((const __m128i *)edge),
            _mm_loadl_epi64((const __m128i *)(edge + 1)),
            _mm_loadl_epi64((const __m128i *)(edge + 2)), round_const);
        _mm_storel_epi64((__m128i *)dst_edge, filtered_head);
      }
      _mm_storel_epi64((__m128i *)(dst_edge + tail_offset), filtered_tail);
    } else {
      uint16_t prev_sample = first_sample;
      for (int i = 1; i < num_px - 1; ++i) {
        const uint16_t cur_sample = edge[i];
        edge[i] = (prev_sample + (cur_sample << 1) + edge[i + 1] + 2) >> 2;
        prev_sample = cur_sample;
      }
      edge[num_px - 1] = (prev_sample + 3 * last_sample + 2) >> 2;
    }
  } else if (strength == 2) {
    if (num_filtered >= 16) {
      const __m256i round_const = _mm256_set1_epi16(8);
      const __m256i last_vec = _mm256_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 16;
      const __m256i tail_prev =
          _mm256_loadu_si256((const __m256i *)(edge + tail_offset));
      const __m256i tail_cur =
          _mm256_loadu_si256((const __m256i *)(edge + tail_offset + 1));
      const __m256i tail_next = shift_edge_next1_avx2(tail_cur, last_vec);
      const __m256i filtered_tail =
          filter_edge_3tap_s2_avx2(tail_prev, tail_cur, tail_next, round_const);
      if (tail_offset == 0) {
        _mm256_storeu_si256((__m256i *)dst_edge, filtered_tail);
        return;
      }
      __m256i filtered_cur = filter_edge_3tap_s2_avx2(
          _mm256_loadu_si256((const __m256i *)edge),
          _mm256_loadu_si256((const __m256i *)(edge + 1)),
          _mm256_loadu_si256((const __m256i *)(edge + 2)), round_const);
      int offset = 0;
      for (; offset + 16 < tail_offset; offset += 16) {
        const __m256i filtered_next = filter_edge_3tap_s2_avx2(
            _mm256_loadu_si256((const __m256i *)(edge + offset + 16)),
            _mm256_loadu_si256((const __m256i *)(edge + offset + 17)),
            _mm256_loadu_si256((const __m256i *)(edge + offset + 18)),
            round_const);
        _mm256_storeu_si256((__m256i *)(dst_edge + offset), filtered_cur);
        filtered_cur = filtered_next;
      }
      _mm256_storeu_si256((__m256i *)(dst_edge + offset), filtered_cur);
      _mm256_storeu_si256((__m256i *)(dst_edge + tail_offset), filtered_tail);
    } else if (num_filtered >= 8) {
      const __m128i round_const = _mm_set1_epi16(8);
      const __m128i last_vec = _mm_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 8;
      const __m128i tail_prev =
          _mm_loadu_si128((const __m128i *)(edge + tail_offset));
      const __m128i tail_cur =
          _mm_loadu_si128((const __m128i *)(edge + tail_offset + 1));
      const __m128i tail_next = _mm_alignr_epi8(last_vec, tail_cur, 2);
      const __m128i filtered_tail =
          filter_edge_3tap_s2_sse2(tail_prev, tail_cur, tail_next, round_const);
      if (tail_offset > 0) {
        const __m128i filtered_head = filter_edge_3tap_s2_sse2(
            _mm_loadu_si128((const __m128i *)edge),
            _mm_loadu_si128((const __m128i *)(edge + 1)),
            _mm_loadu_si128((const __m128i *)(edge + 2)), round_const);
        _mm_storeu_si128((__m128i *)dst_edge, filtered_head);
      }
      _mm_storeu_si128((__m128i *)(dst_edge + tail_offset), filtered_tail);
    } else if (num_filtered >= 4) {
      const __m128i round_const = _mm_set1_epi16(8);
      const __m128i last_vec = _mm_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 4;
      const __m128i tail_prev =
          _mm_loadl_epi64((const __m128i *)(edge + tail_offset));
      const __m128i tail_cur =
          _mm_loadl_epi64((const __m128i *)(edge + tail_offset + 1));
      const __m128i tail_next =
          _mm_alignr_epi8(last_vec, _mm_unpacklo_epi64(last_vec, tail_cur), 10);
      const __m128i filtered_tail =
          filter_edge_3tap_s2_sse2(tail_prev, tail_cur, tail_next, round_const);
      if (tail_offset > 0) {
        const __m128i filtered_head = filter_edge_3tap_s2_sse2(
            _mm_loadl_epi64((const __m128i *)edge),
            _mm_loadl_epi64((const __m128i *)(edge + 1)),
            _mm_loadl_epi64((const __m128i *)(edge + 2)), round_const);
        _mm_storel_epi64((__m128i *)dst_edge, filtered_head);
      }
      _mm_storel_epi64((__m128i *)(dst_edge + tail_offset), filtered_tail);
    } else {
      uint16_t prev_sample = first_sample;
      for (int i = 1; i < num_px - 1; ++i) {
        const uint16_t cur_sample = edge[i];
        edge[i] = (5 * prev_sample + 6 * cur_sample + 5 * edge[i + 1] + 8) >> 4;
        prev_sample = cur_sample;
      }
      edge[num_px - 1] = (5 * prev_sample + 11 * last_sample + 8) >> 4;
    }
  } else {
    if (num_filtered >= 16) {
      const __m256i round_const = _mm256_set1_epi16(4);
      const __m256i first_vec = _mm256_set1_epi16((int16_t)first_sample);
      const __m256i last_vec = _mm256_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 16;
      const __m256i tail_prev1 =
          _mm256_loadu_si256((const __m256i *)(edge + tail_offset));
      const __m256i tail_cur =
          _mm256_loadu_si256((const __m256i *)(edge + tail_offset + 1));
      const __m256i tail_prev2 =
          (tail_offset >= 1)
              ? _mm256_loadu_si256((const __m256i *)(edge + tail_offset - 1))
              : shift_edge_prev1_avx2(tail_prev1, first_vec);
      const __m256i tail_next1 = shift_edge_next1_avx2(tail_cur, last_vec);
      const __m256i tail_next2 = shift_edge_next2_avx2(tail_cur, last_vec);
      const __m256i filtered_tail =
          filter_edge_5tap_s3_avx2(tail_prev2, tail_prev1, tail_cur, tail_next1,
                                   tail_next2, round_const);
      if (tail_offset == 0) {
        _mm256_storeu_si256((__m256i *)dst_edge, filtered_tail);
        return;
      }
      const __m256i head_prev1 = _mm256_loadu_si256((const __m256i *)edge);
      const __m256i head_cur = _mm256_loadu_si256((const __m256i *)(edge + 1));
      const __m256i head_next1 =
          _mm256_loadu_si256((const __m256i *)(edge + 2));
      const __m256i head_next2 =
          (num_px >= 19) ? _mm256_loadu_si256((const __m256i *)(edge + 3))
                         : shift_edge_next1_avx2(head_next1, last_vec);
      __m256i filtered_cur = filter_edge_5tap_s3_avx2(
          shift_edge_prev1_avx2(head_prev1, first_vec), head_prev1, head_cur,
          head_next1, head_next2, round_const);
      int offset = 0;
      for (; offset + 16 < tail_offset; offset += 16) {
        const int next_offset = offset + 16;
        const __m256i blk_next1 =
            _mm256_loadu_si256((const __m256i *)(edge + next_offset + 2));
        const __m256i blk_next2 =
            (next_offset + 19 <= num_px)
                ? _mm256_loadu_si256((const __m256i *)(edge + next_offset + 3))
                : shift_edge_next1_avx2(blk_next1, last_vec);
        const __m256i filtered_next = filter_edge_5tap_s3_avx2(
            _mm256_loadu_si256((const __m256i *)(edge + next_offset - 1)),
            _mm256_loadu_si256((const __m256i *)(edge + next_offset)),
            _mm256_loadu_si256((const __m256i *)(edge + next_offset + 1)),
            blk_next1, blk_next2, round_const);
        _mm256_storeu_si256((__m256i *)(dst_edge + offset), filtered_cur);
        filtered_cur = filtered_next;
      }
      _mm256_storeu_si256((__m256i *)(dst_edge + offset), filtered_cur);
      _mm256_storeu_si256((__m256i *)(dst_edge + tail_offset), filtered_tail);
    } else if (num_filtered >= 8) {
      const __m128i round_const = _mm_set1_epi16(4);
      const __m128i first_vec = _mm_set1_epi16((int16_t)first_sample);
      const __m128i last_vec = _mm_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 8;
      const __m128i tail_prev1 =
          _mm_loadu_si128((const __m128i *)(edge + tail_offset));
      const __m128i tail_cur =
          _mm_loadu_si128((const __m128i *)(edge + tail_offset + 1));
      const __m128i tail_prev2 =
          (tail_offset >= 1)
              ? _mm_loadu_si128((const __m128i *)(edge + tail_offset - 1))
              : _mm_alignr_epi8(tail_prev1, first_vec, 14);
      const __m128i tail_next1 = _mm_alignr_epi8(last_vec, tail_cur, 2);
      const __m128i tail_next2 = _mm_alignr_epi8(last_vec, tail_cur, 4);
      const __m128i filtered_tail =
          filter_edge_5tap_s3_sse2(tail_prev2, tail_prev1, tail_cur, tail_next1,
                                   tail_next2, round_const);
      if (tail_offset > 0) {
        const __m128i head_prev1 = _mm_loadu_si128((const __m128i *)edge);
        const __m128i head_cur = _mm_loadu_si128((const __m128i *)(edge + 1));
        const __m128i head_next1 = _mm_loadu_si128((const __m128i *)(edge + 2));
        const __m128i head_next2 =
            (num_px >= 11) ? _mm_loadu_si128((const __m128i *)(edge + 3))
                           : _mm_alignr_epi8(last_vec, head_next1, 2);
        const __m128i filtered_head = filter_edge_5tap_s3_sse2(
            _mm_alignr_epi8(head_prev1, first_vec, 14), head_prev1, head_cur,
            head_next1, head_next2, round_const);
        _mm_storeu_si128((__m128i *)dst_edge, filtered_head);
      }
      _mm_storeu_si128((__m128i *)(dst_edge + tail_offset), filtered_tail);
    } else if (num_filtered >= 4) {
      const __m128i round_const = _mm_set1_epi16(4);
      const __m128i first_vec = _mm_set1_epi16((int16_t)first_sample);
      const __m128i last_vec = _mm_set1_epi16((int16_t)last_sample);
      const int tail_offset = num_filtered - 4;
      const __m128i tail_prev1 = _mm_unpacklo_epi64(
          _mm_loadl_epi64((const __m128i *)(edge + tail_offset)), last_vec);
      const __m128i tail_cur = _mm_unpacklo_epi64(
          _mm_loadl_epi64((const __m128i *)(edge + tail_offset + 1)), last_vec);
      const __m128i tail_prev2 =
          (tail_offset >= 1)
              ? _mm_loadl_epi64((const __m128i *)(edge + tail_offset - 1))
              : _mm_alignr_epi8(tail_prev1, first_vec, 14);
      const __m128i tail_next1 = _mm_alignr_epi8(last_vec, tail_cur, 2);
      const __m128i tail_next2 = _mm_alignr_epi8(last_vec, tail_cur, 4);
      const __m128i filtered_tail =
          filter_edge_5tap_s3_sse2(tail_prev2, tail_prev1, tail_cur, tail_next1,
                                   tail_next2, round_const);
      if (tail_offset > 0) {
        const __m128i head_prev1 = _mm_loadl_epi64((const __m128i *)edge);
        const __m128i head_cur = _mm_loadl_epi64((const __m128i *)(edge + 1));
        const __m128i head_next1 = _mm_unpacklo_epi64(
            _mm_loadl_epi64((const __m128i *)(edge + 2)), last_vec);
        const __m128i head_next2 =
            (num_px >= 7) ? _mm_loadl_epi64((const __m128i *)(edge + 3))
                          : _mm_alignr_epi8(last_vec, head_next1, 2);
        const __m128i filtered_head = filter_edge_5tap_s3_sse2(
            _mm_alignr_epi8(head_prev1, first_vec, 14), head_prev1, head_cur,
            head_next1, head_next2, round_const);
        _mm_storel_epi64((__m128i *)dst_edge, filtered_head);
      }
      _mm_storel_epi64((__m128i *)(dst_edge + tail_offset), filtered_tail);
    } else {
      uint16_t prev2 = first_sample;
      uint16_t prev1 = first_sample;
      for (int i = 1; i < num_px; ++i) {
        const uint16_t cur = edge[i];
        const uint16_t next1 = (i + 1 < num_px) ? edge[i + 1] : last_sample;
        const uint16_t next2 = (i + 2 < num_px) ? edge[i + 2] : last_sample;
        edge[i] = (prev2 + ((prev1 + cur + next1) << 1) + next2 + 4) >> 3;
        prev2 = prev1;
        prev1 = cur;
      }
    }
  }
}
