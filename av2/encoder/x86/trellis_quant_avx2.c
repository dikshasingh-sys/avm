/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
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
#include <string.h>
#include <immintrin.h> /* AVX2 */

#include "avm/avm_integer.h"
#include "avm_dsp/x86/mem_sse2.h"
#include "av2/common/av2_common_int.h"
#include "av2/common/quant_common.h"
#include "av2/encoder/encodetxb.h"
#include "av2/encoder/trellis_quant.h"
#include "av2/encoder/x86/trellis_quant_avx2.h"
#include "avm_dsp/x86/synonyms.h"
#include "avm_dsp/x86/synonyms_avx2.h"

static const uint8_t kConst[4][16] = {
  { 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 },
};

static __m128i map_state(__m128i state, __m128i prev_st) {
  // Track previous states.
  __m128i map_st = _mm_shuffle_epi8(state, prev_st);
  // Set state to -1 to indicate eob truncation.
  map_st = _mm_blendv_epi8(map_st, prev_st, prev_st);
  return map_st;
}

static AVM_FORCE_INLINE void update_nbr_diagonal_core_avx2(
    struct tcq_ctx_t *tcq_ctx, __m256i orig_st, int row, int col, int bwl) {
  const int diag = row + col;
  const int idx_start = col;
  const int idx_end = AVMMIN(diag + 1, 1 << bwl);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i mask8 = _mm256_loadu_si256((const __m256i *)kConst[0]);
  orig_st = _mm256_or_si256(orig_st, mask8);

  static const int8_t max_tbl[4] = { 0, 8, 6, 4 };
  const __m256i lev_max1 = _mm256_set1_epi8(diag < 5 ? 5 : 3);
  const __m256i lev_max2 = _mm256_set1_epi8(diag < 6 ? 5 : 3);
  const __m256i base_ctx_max = _mm256_set1_epi8(max_tbl[AVMMIN(diag, 3)]);
  const __m256i mid_ctx_max = _mm256_set1_epi8(6);

  const int idx0 = AVMMAX(idx_start - 2, 0);
  const __m128i state0 = _mm_loadu_si128((const __m128i *)kConst[2]);
  __m128i lev01 = _mm_loadu_si128((const __m128i *)&tcq_ctx->lev_new[idx0]);
  const __m128i prev_st0 =
      _mm_loadu_si128((const __m128i *)&tcq_ctx->prev_st[idx0]);
  const __m128i prev_st1 = _mm_srli_si128(prev_st0, 8);
  const __m128i state1 = map_state(prev_st0, state0);
  __m128i state2 = map_state(prev_st1, state1);
  __m128i state01 = _mm_unpacklo_epi64(state0, state1);
  state01 = _mm_or_si128(state01, _mm256_castsi256_si128(mask8));
  lev01 = _mm_shuffle_epi8(lev01, state01);
  __m256i lev__01 = _mm256_set_m128i(lev01, lev01);

  for (int i = idx0;; i += 4) {
    const __m128i prev_st2 =
        _mm_loadu_si128((const __m128i *)&tcq_ctx->prev_st[i + 2]);
    const __m128i prev_st3 = _mm_srli_si128(prev_st2, 8);
    const __m128i prev_st4 =
        _mm_loadu_si128((const __m128i *)&tcq_ctx->prev_st[i + 4]);
    const __m128i state3 = map_state(prev_st2, state2);
    const __m128i state4 = map_state(prev_st3, state3);
    const __m128i state5 = map_state(prev_st4, state4);
    const __m128i state23 = _mm_unpacklo_epi64(state2, state3);
    const __m128i state45 = _mm_unpacklo_epi64(state4, state5);
    __m256i state2345 = _mm256_set_m128i(state45, state23);
    __m256i lev2345 =
        _mm256_loadu_si256((const __m256i *)&tcq_ctx->lev_new[i + 2]);
    state2345 = _mm256_or_si256(state2345, mask8);
    lev2345 = _mm256_shuffle_epi8(lev2345, state2345);
    const __m256i lev0123 = _mm256_permute2x128_si256(lev__01, lev2345, 0x21);
    const __m256i lev1234 = _mm256_alignr_epi8(lev2345, lev0123, 8);

    __m256i base = _mm256_loadu_si256((const __m256i *)tcq_ctx->mag_base[i]);
    __m256i mid = _mm256_loadu_si256((const __m256i *)tcq_ctx->mag_mid[i]);
    base = _mm256_shuffle_epi8(base, orig_st);
    mid = _mm256_shuffle_epi8(mid, orig_st);
    const __m256i lev0123_max1 = _mm256_min_epu8(lev0123, lev_max1);
    const __m256i lev1234_max1 = _mm256_min_epu8(lev1234, lev_max1);
    const __m256i base_sum2 = _mm256_adds_epu8(lev0123_max1, lev1234_max1);
    base = _mm256_adds_epu8(base, base_sum2);
    base = _mm256_avg_epu8(base, zero);
    base = _mm256_min_epu8(base, base_ctx_max);
    const __m256i mid_sum2 = _mm256_adds_epu8(lev0123, lev1234);
    mid = _mm256_adds_epu8(mid, mid_sum2);
    mid = _mm256_avg_epu8(mid, zero);
    mid = _mm256_min_epu8(mid, mid_ctx_max);
    mid = _mm256_slli_epi16(mid, 4);
    _mm256_storeu_si256((__m256i *)tcq_ctx->ctx[i], _mm256_or_si256(base, mid));

    const __m256i lev0123_max2 = _mm256_min_epu8(lev0123, lev_max2);
    const __m256i lev1234_max2 = _mm256_min_epu8(lev1234, lev_max2);
    const __m256i lev2345_max2 = _mm256_min_epu8(lev2345, lev_max2);
    __m256i base_sum3 = _mm256_adds_epu8(lev0123_max2, lev1234_max2);
    base_sum3 = _mm256_adds_epu8(base_sum3, lev2345_max2);
    _mm256_storeu_si256((__m256i *)tcq_ctx->mag_base[i], base_sum3);
    _mm256_storeu_si256((__m256i *)tcq_ctx->mag_mid[i], lev1234);

    if (i + 4 >= idx_end) break;
    lev__01 = lev2345;
    state2 = map_state(_mm_srli_si128(prev_st4, 8), state5);
  }
}

// Update neighbor coeff magnitudes state for current diagonal.
// Diagonal is indicated by its last (col, row) position.
void av2_update_nbr_diagonal_avx2(struct tcq_ctx_t *tcq_ctx, int row, int col,
                                  int bwl) {
  __m256i orig_st = _mm256_castsi128_si256(_mm_loadu_si64(tcq_ctx->orig_st));
  orig_st = _mm256_permute4x64_epi64(orig_st, 0);
  _mm_storeu_si64(&tcq_ctx->orig_st,
                  _mm_loadu_si128((const __m128i *)kConst[2]));
  update_nbr_diagonal_core_avx2(tcq_ctx, orig_st, row, col, bwl);
}

// Pre-calculate eob bits (rate) for each EOB candidate position from 1
// to the initial eob location. Store rate in array block_eob_rate[],
// starting with index.
void av2_calc_block_eob_rate_avx2(struct macroblock *x, int plane,
                                  TX_SIZE tx_size, int eob,
                                  uint16_t *block_eob_rate) {
  const MACROBLOCKD *xd = &x->e_mbd;
  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const CoeffCosts *coeff_costs = &x->coeff_costs;
  const LV_MAP_COEFF_COST *txb_costs =
      &coeff_costs->coeff_costs[txs_ctx][plane_type];
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *txb_eob_costs =
      &coeff_costs->eob_costs[eob_multi_size][plane_type];

  const int *tbl_eob_cost = txb_eob_costs->eob_cost[is_inter];
  const int(*tbl_eob_extra)[2] = txb_costs->eob_extra_cost;

  static const int8_t kShuf[4][32] = {
    { -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13,
      0,  1,  0,  1,  0, 1, 0, 1, 4, 5, 4, 5, 4,  5,  4,  5 },
    { 0, 1, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13, 12, 13, 12, 13,
      0, 1, 0, 1, 0, 1, 0, 1, 0,  1,  0,  1,  0,  1,  0,  1 },
    { 8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13 },
  };
#define BC1 (1 << AV2_PROB_COST_SHIFT)
#define BC2 (2 * BC1)
  static const uint16_t kBitCost[16] = {
    0, 0, 0, 0, BC1, BC1, BC1, BC1, BC2, BC2, BC2, BC2, BC2, BC2, BC2, BC2
  };

  // Write first 16 costs, block_eob_rate[0:15]
  // Convert 32-bit eob_pt costs { 0 1 2 3 4 5 6 7 } + eob_extra_cost
  // to expanded 16-bit costs { 0 1 2 2 3 3 3 3 4 4 4 4 4 4 4 4 }.
  __m256i eob_cost0_7 = _mm256_lddqu_si256((__m256i *)tbl_eob_cost);
  __m256i eob_extra0_7 = _mm256_lddqu_si256((__m256i *)tbl_eob_extra);
  __m256i shuf0 = _mm256_lddqu_si256((__m256i *)kShuf[0]);
  __m256i shuf1 = _mm256_lddqu_si256((__m256i *)kShuf[1]);
  __m256i eob_extra = _mm256_shuffle_epi8(eob_extra0_7, shuf0);
  __m256i eob_rate0_15 = _mm256_shuffle_epi8(eob_cost0_7, shuf1);
  eob_rate0_15 = _mm256_add_epi16(eob_rate0_15, eob_extra);
  __m256i bit_cost = _mm256_lddqu_si256((__m256i *)kBitCost);
  eob_rate0_15 = _mm256_add_epi16(eob_rate0_15, bit_cost);
  _mm256_storeu_si256((__m256i *)&block_eob_rate[0], eob_rate0_15);
  if (eob <= 16) return;

  // Write second 16 costs, block_eob_rate[16:31]
  __m256i eob_cost4_7 = _mm256_permute4x64_epi64(eob_cost0_7, 0xEE);
  __m256i eob_extra4_7 = _mm256_permute4x64_epi64(eob_extra0_7, 0xEE);
  __m256i shuf2 = _mm256_lddqu_si256((__m256i *)kShuf[2]);
  __m256i shuf3 = _mm256_set1_epi16(0x0504);
  __m256i eob_extra16_31 = _mm256_shuffle_epi8(eob_extra4_7, shuf2);
  __m256i eob_rate16_31 = _mm256_shuffle_epi8(eob_cost4_7, shuf3);
  eob_rate16_31 = _mm256_add_epi16(eob_rate16_31, eob_extra16_31);
  __m256i bit_cost16_31 = _mm256_set1_epi16(3 * BC1);
  eob_rate16_31 = _mm256_add_epi16(eob_rate16_31, bit_cost16_31);
  _mm256_storeu_si256((__m256i *)&block_eob_rate[16], eob_rate16_31);

  // Write costs beyond position 32, block_eob_rate[32+]
  int scan_pos = 32;
  int n_offset_bits = 4;
  while (scan_pos < eob) {
    int eob_pt_low = AVMMIN(2 + n_offset_bits, EOB_PT_INDEX_COUNT - 1);
    int eob_pt_rate = tbl_eob_cost[eob_pt_low];
    if (eob_multi_size == 4 && (eob_pt_low == EOB_PT_INDEX_COUNT - 1))
      eob_pt_rate += av2_cost_literal(1);
    else if (eob_multi_size > 4 && (eob_pt_low == EOB_PT_INDEX_COUNT - 1))
      eob_pt_rate += av2_cost_literal(2);
    for (int bit = 0; bit < 2; bit++) {
      int eob_ctx = n_offset_bits;
      int extra_bit_rate = tbl_eob_extra[eob_ctx][bit];
      int eob_rate =
          eob_pt_rate + extra_bit_rate + av2_cost_literal(n_offset_bits);
      for (int i = 0; i < (1 << n_offset_bits); i += 16) {
        __m256i rate = _mm256_set1_epi16(eob_rate);
        _mm256_storeu_si256((__m256i *)&block_eob_rate[scan_pos], rate);
        scan_pos += 16;
      }
    }
    n_offset_bits++;
  }
}

int av2_find_best_path_avx2(const struct tcq_node_t *trellis,
                            const int16_t *scan, const int32_t *dequant,
                            const qm_val_t *iqmatrix, const tran_low_t *tcoeff,
                            int first_scan_pos, int log_scale,
                            tran_low_t *qcoeff, tran_low_t *dqcoeff,
                            int *min_rate, int64_t *min_cost) {
  int64_t min_path_cost = INT64_MAX;
  int trel_min_rate = INT32_MAX;
  int prev_id = -2;
  for (int state = 0; state < TCQ_N_STATES; state++) {
    const tcq_node_t *decision = &trellis[state];
    if (decision->rdCost < min_path_cost) {
      prev_id = state;
      min_path_cost = decision->rdCost;
      trel_min_rate = decision->rate;
    }
  }

  // Backtrack to reconstruct qcoeff / dqcoeff blocks using scalar GPRs to
  // avoid XMM->GPR domain-crossing latency on the serial prev_id chain.
  int scan_pos = 0;
  if (!iqmatrix) {
    uint32_t dqv = (uint32_t)dequant[0];
    const uint32_t dqv_ac = (uint32_t)dequant[1];
    const uint64_t round = 1ULL << (QUANT_TABLE_BITS - 1);
    const int shift = QUANT_TABLE_BITS + log_scale;
    for (; prev_id >= 0; scan_pos++) {
      const uint8_t *decision =
          (const uint8_t *)&trellis[(scan_pos << TCQ_N_STATES_LOG) + prev_id];
      int32_t info;
      memcpy(&info, decision + 12, sizeof(info));
      const int next_id = info >> 24;
      const int abs_lev = (int)((uint32_t)info & 0xFFFFFFu);
      const int blk_pos = scan[scan_pos];
      const int sign = tcoeff[blk_pos] >> 31;
      const int dq = (next_id >= 0) ? ((next_id >> 1) & 1) : 0;
      const uint32_t qc = abs_lev ? (uint32_t)(2 * abs_lev - dq) : 0u;
      const int dqc = (int)(((uint64_t)qc * dqv + round) >> shift);
      qcoeff[blk_pos] = (abs_lev ^ sign) - sign;
      dqcoeff[blk_pos] = (dqc ^ sign) - sign;
      dqv = dqv_ac;
      prev_id = next_id;
    }
  } else {
    for (; prev_id >= 0; scan_pos++) {
      const tcq_node_t *decision =
          &trellis[(scan_pos << TCQ_N_STATES_LOG) + prev_id];
      prev_id = decision->prevId;
      int abs_level = decision->absLevel;
      int blk_pos = scan[scan_pos];
      int sign = tcoeff[blk_pos] < 0;
      qcoeff[blk_pos] = sign ? -abs_level : abs_level;
      int dqv = get_dqv(dequant, blk_pos, iqmatrix);
      int dq = prev_id >= 0 ? tcq_quant(prev_id) : 0;
      int qc = (abs_level == 0) ? 0 : (2 * abs_level - dq);
      int dqc = (tran_low_t)ROUND_POWER_OF_TWO_64((tran_high_t)qc * dqv,
                                                  QUANT_TABLE_BITS) >>
                log_scale;
      dqcoeff[blk_pos] = sign ? -dqc : dqc;
    }
  }
  int eob = scan_pos;

  for (; scan_pos <= first_scan_pos; scan_pos++) {
    int blk_pos = scan[scan_pos];
    qcoeff[blk_pos] = 0;
    dqcoeff[blk_pos] = 0;
  }

  *min_rate = trel_min_rate;
  *min_cost = min_path_cost;
  return eob;
}

// -----------------------------------------------------------------------------
// 8-state trellis loop that keeps the trellis state in registers. The vectors
// "a" and "b" hold the states { 0, 4, 6, 2 } and { 1, 5, 7, 3 }, so each
// state shares a lane with its partner (state ^ 1).

typedef struct {
  __m256i rd_a, rd_b;
  __m256i info_a, info_b;  // rate | absLevel << 32 | prevId << 56
  __m256i orig;            // origin state of each lane, -1 for a new eob
} TcqRegs;

typedef struct {
  __m256i rdmult, mask16, mask32;
  __m256i pid_a, pid_b;    // prevId << 56 of each lane
  __m256i pid1_a, pid1_b;  // prevId << 56 | 1 << 32
  __m256i dq_shift, sh_shift, v_dqv_ac;
} TcqConsts;

// State of each 32-bit lane, and lane of each state.
DECLARE_ALIGNED(32, static const int32_t, kLaneState[8]) = { 0, 1, 4, 5,
                                                             6, 7, 2, 3 };
DECLARE_ALIGNED(32, static const int32_t, kStateLane[8]) = { 0, 1, 6, 7,
                                                             2, 3, 4, 5 };
DECLARE_ALIGNED(32, static const int32_t, kNextOrigPerm[8]) = { 0, 6, 1, 7,
                                                                2, 4, 3, 5 };
// Moves lev_new (byte 0) and prev_st (byte 1) of each lane to the bytes
// { lev 0-7, prev 0-7 }, in state order; the other bytes become zero.
DECLARE_ALIGNED(32, static const int8_t,
                kLevPrevShuf[32]) = { 0,  4, -1, -1, 8,  12, -1, -1, 1,  5,  -1,
                                      -1, 9, 13, -1, -1, -1, -1, 8,  12, -1, -1,
                                      0,  4, -1, -1, 9,  13, -1, -1, 1,  5 };

// Hides v from the optimizer, which would turn the ANDs with the 16-bit mask
// into word blends (port 5 only on Intel CPUs).
static AVM_FORCE_INLINE __m256i tcq_opaque(__m256i v) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__("" : "+x"(v));
#endif
  return v;
}

static AVM_FORCE_INLINE void tcq_init_consts(const tcq_param_t *p,
                                             TcqConsts *c) {
  // rdmult < 2^32 (asserted by the caller): zero-extending it lets the
  // compiler emit a single vpmuludq per rate.
  c->rdmult = _mm256_set1_epi64x((int64_t)(uint32_t)p->rdmult);
  c->mask16 = tcq_opaque(_mm256_set1_epi64x(0xFFFF));
  c->mask32 = _mm256_set1_epi64x(0xFFFFFFFF);
  c->pid_a = _mm256_setr_epi64x(0LL << 56, 4LL << 56, 6LL << 56, 2LL << 56);
  c->pid_b = _mm256_setr_epi64x(1LL << 56, 5LL << 56, 7LL << 56, 3LL << 56);
  const __m256i one = _mm256_set1_epi64x(1LL << 32);
  c->pid1_a = _mm256_add_epi64(c->pid_a, one);
  c->pid1_b = _mm256_add_epi64(c->pid_b, one);
  c->dq_shift = _mm256_set1_epi64x(QUANT_TABLE_BITS + p->log_scale);
  c->sh_shift = _mm256_set1_epi64x(p->log_scale - 1);
  c->v_dqv_ac = _mm256_set1_epi64x((uint32_t)p->dequant[1]);
}

static AVM_FORCE_INLINE void tcq_load_regs(const tcq_node_t *prev,
                                           const tcq_ctx_t *tcq_ctx,
                                           TcqRegs *s) {
  const __m256i n01 = _mm256_loadu_si256((const __m256i *)&prev[0]);
  const __m256i n23 = _mm256_loadu_si256((const __m256i *)&prev[2]);
  const __m256i n45 = _mm256_loadu_si256((const __m256i *)&prev[4]);
  const __m256i n67 = _mm256_loadu_si256((const __m256i *)&prev[6]);
  const __m256i n06 = _mm256_permute2x128_si256(n01, n67, 0x20);
  const __m256i n42 = _mm256_permute2x128_si256(n45, n23, 0x20);
  const __m256i n17 = _mm256_permute2x128_si256(n01, n67, 0x31);
  const __m256i n53 = _mm256_permute2x128_si256(n45, n23, 0x31);
  s->rd_a = _mm256_unpacklo_epi64(n06, n42);
  s->info_a = _mm256_unpackhi_epi64(n06, n42);
  s->rd_b = _mm256_unpacklo_epi64(n17, n53);
  s->info_b = _mm256_unpackhi_epi64(n17, n53);
  const __m256i orig =
      _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)tcq_ctx->orig_st));
  s->orig = _mm256_permutevar8x32_epi32(
      orig, _mm256_load_si256((const __m256i *)kLaneState));
}

// Convert s->orig (in lane order) to repeated 8-byte state-order shuffle mask
// for update_nbr_diagonal_core_avx2() without storing/reloading memory.
static AVM_FORCE_INLINE __m256i tcq_orig_by_state_bytes(const TcqRegs *s) {
  const __m256i orig = _mm256_permutevar8x32_epi32(
      s->orig, _mm256_load_si256((const __m256i *)kStateLane));
  const __m256i w = _mm256_packs_epi32(orig, orig);
  const __m256i b = _mm256_packs_epi16(w, w);
  const __m128i st = _mm_unpacklo_epi32(_mm256_castsi256_si128(b),
                                        _mm256_extracti128_si256(b, 1));
  return _mm256_permute4x64_epi64(_mm256_castsi128_si256(st), 0);
}

static AVM_FORCE_INLINE void tcq_store_nodes(const TcqRegs *s,
                                             tcq_node_t *decision) {
  const __m256i n06 = _mm256_unpacklo_epi64(s->rd_a, s->info_a);
  const __m256i n42 = _mm256_unpackhi_epi64(s->rd_a, s->info_a);
  const __m256i n17 = _mm256_unpacklo_epi64(s->rd_b, s->info_b);
  const __m256i n53 = _mm256_unpackhi_epi64(s->rd_b, s->info_b);
  _mm256_storeu_si256((__m256i *)&decision[0],
                      _mm256_permute2x128_si256(n06, n17, 0x20));
  _mm256_storeu_si256((__m256i *)&decision[2],
                      _mm256_permute2x128_si256(n42, n53, 0x31));
  _mm256_storeu_si256((__m256i *)&decision[4],
                      _mm256_permute2x128_si256(n42, n53, 0x20));
  _mm256_storeu_si256((__m256i *)&decision[6],
                      _mm256_permute2x128_si256(n06, n17, 0x31));
}

// Writes lev_new / prev_st for update_nbr_diagonal() and returns the packed
// 32-bit (absLevel | prevId << 24) words in lane order. s->orig is already
// updated directly inside tcq_select() and tcq_try_eob().
static AVM_FORCE_INLINE __m256i tcq_update_lev_prev(const TcqRegs *s,
                                                    uint8_t *lev_new,
                                                    int8_t *prev_st) {
  const __m256i hi =
      _mm256_blend_epi32(_mm256_srli_epi64(s->info_a, 32), s->info_b, 0xAA);
  __m256i lev =
      _mm256_min_epi32(_mm256_and_si256(hi, _mm256_set1_epi32(0xFFFFFF)),
                       _mm256_set1_epi32(MAX_VAL_BR_CTX));
  lev = _mm256_or_si256(lev, _mm256_and_si256(_mm256_srli_epi32(hi, 16),
                                              _mm256_set1_epi32(0xFF00)));
  lev = _mm256_shuffle_epi8(lev,
                            _mm256_load_si256((const __m256i *)kLevPrevShuf));
  const __m128i lp = _mm_or_si128(_mm256_castsi256_si128(lev),
                                  _mm256_extracti128_si256(lev, 1));
  _mm_storel_epi64((__m128i *)lev_new, lp);
  _mm_storeh_pi((__m64 *)prev_st, _mm_castsi128_ps(lp));
  return hi;
}

static AVM_FORCE_INLINE __m256i tcq_rdc(__m256i rate, const TcqConsts *c) {
  return tcq_rate_cost(rate, c->rdmult);
}

// Each state keeps the cheaper of its even candidate and its partner's odd
// candidate (ties go to the even one), then moves to its next state along with
// its origin state in s->orig.
static AVM_FORCE_INLINE void tcq_select(TcqRegs *s, __m256i ev_rd_a,
                                        __m256i ev_info_a, __m256i ev_rd_b,
                                        __m256i ev_info_b, __m256i od_rd_a,
                                        __m256i od_info_a, __m256i od_rd_b,
                                        __m256i od_info_b) {
  const __m256i use_odd_a = _mm256_cmpgt_epi64(ev_rd_a, od_rd_b);
  const __m256i use_odd_b = _mm256_cmpgt_epi64(ev_rd_b, od_rd_a);
  const __m256i orig_od = _mm256_shuffle_epi32(s->orig, 0xB1);
  const __m256i use_odd = _mm256_blend_epi32(use_odd_a, use_odd_b, 0xAA);
  const __m256i orig_sel = _mm256_blendv_epi8(s->orig, orig_od, use_odd);
  s->orig = _mm256_permutevar8x32_epi32(
      orig_sel, _mm256_load_si256((const __m256i *)kNextOrigPerm));

  const __m256i ra = _mm256_blendv_epi8(ev_rd_a, od_rd_b, use_odd_a);
  const __m256i rb = _mm256_blendv_epi8(ev_rd_b, od_rd_a, use_odd_b);
  const __m256i ia = _mm256_blendv_epi8(ev_info_a, od_info_b, use_odd_a);
  const __m256i ib = _mm256_blendv_epi8(ev_info_b, od_info_a, use_odd_b);
  const __m256i rlo = _mm256_unpacklo_epi64(ra, rb);
  const __m256i rhi = _mm256_unpackhi_epi64(ra, rb);
  const __m256i ilo = _mm256_unpacklo_epi64(ia, ib);
  const __m256i ihi = _mm256_unpackhi_epi64(ia, ib);
  s->rd_a = _mm256_permute2x128_si256(rlo, rhi, 0x20);
  s->rd_b = _mm256_permute2x128_si256(rhi, rlo, 0x31);
  s->info_a = _mm256_permute2x128_si256(ilo, ihi, 0x20);
  s->info_b = _mm256_permute2x128_si256(ihi, ilo, 0x31);
}

// New eob candidates for states 0 (a lane 0, orig lane 0) and 4 (a lane 1,
// orig lane 2). Unused lanes of eob_rd are INT64_MAX.
static AVM_FORCE_INLINE void tcq_try_eob(TcqRegs *s, __m256i eob_rd,
                                         __m256i eob_node_info) {
  const __m256i use = _mm256_cmpgt_epi64(s->rd_a, eob_rd);
  s->rd_a = _mm256_blendv_epi8(s->rd_a, eob_rd, use);
  s->info_a = _mm256_blendv_epi8(s->info_a, eob_node_info, use);
  s->orig = _mm256_or_si256(s->orig, _mm256_srli_epi64(use, 32));
}

// Fetch the packed rates of each lane from the per-origin tables.
static AVM_FORCE_INLINE __m256i tcq_fetch(const TcqRegs *s, __m256i t0,
                                          __m256i t1, __m256i neg) {
  const __m256i v =
      _mm256_blend_epi32(_mm256_permutevar8x32_epi32(t0, s->orig),
                         _mm256_permutevar8x32_epi32(t1, s->orig), 0xF0);
  return _mm256_blendv_epi8(v, neg, s->orig);
}

// Position with orig_qIdx < 2: candidates are zero and the odd level 1.
static AVM_FORCE_INLINE void tcq_step_q1_rates(TcqRegs *s, const TcqConsts *c,
                                               __m256i za, __m256i oa,
                                               __m256i zb, __m256i ob,
                                               __m256i dist_odd) {
  const __m256i rate_a = _mm256_and_si256(s->info_a, c->mask32);
  const __m256i rate_b = _mm256_and_si256(s->info_b, c->mask32);

  const __m256i ev_rd_a = _mm256_add_epi64(s->rd_a, tcq_rdc(za, c));
  const __m256i ev_rd_b = _mm256_add_epi64(s->rd_b, tcq_rdc(zb, c));
  const __m256i ev_info_a =
      _mm256_add_epi64(_mm256_add_epi64(rate_a, za), c->pid_a);
  const __m256i ev_info_b =
      _mm256_add_epi64(_mm256_add_epi64(rate_b, zb), c->pid_b);
  const __m256i od_rd_a =
      _mm256_add_epi64(s->rd_a, _mm256_add_epi64(tcq_rdc(oa, c), dist_odd));
  const __m256i od_rd_b =
      _mm256_add_epi64(s->rd_b, _mm256_add_epi64(tcq_rdc(ob, c), dist_odd));
  const __m256i od_info_a =
      _mm256_add_epi64(_mm256_add_epi64(rate_a, oa), c->pid1_a);
  const __m256i od_info_b =
      _mm256_add_epi64(_mm256_add_epi64(rate_b, ob), c->pid1_b);
  tcq_select(s, ev_rd_a, ev_info_a, ev_rd_b, ev_info_b, od_rd_a, od_info_a,
             od_rd_b, od_info_b);
}

static AVM_FORCE_INLINE void tcq_step_q1(TcqRegs *s, const TcqConsts *c,
                                         __m256i tq0, __m256i tq1, __m256i neg,
                                         __m256i dist_odd) {
  const __m256i v = tcq_fetch(s, tq0, tq1, neg);
  const __m256i za = _mm256_and_si256(v, c->mask16);
  const __m256i oa = _mm256_and_si256(_mm256_srli_epi64(v, 16), c->mask16);
  const __m256i zb = _mm256_and_si256(_mm256_srli_epi64(v, 32), c->mask16);
  const __m256i ob = _mm256_srli_epi64(v, 48);
  tcq_step_q1_rates(s, c, za, oa, zb, ob, dist_odd);
}

// Position with orig_qIdx >= 2: candidates are zero, one even and one odd
// level per quantizer.
static AVM_FORCE_INLINE void tcq_step_full_rates(TcqRegs *s, const TcqConsts *c,
                                                 __m256i ea, __m256i oa,
                                                 __m256i eb, __m256i ob,
                                                 __m256i za, __m256i zb,
                                                 const TcqFullPos *pos) {
  const __m256i dist_even = pos->dist_even;
  const __m256i dist_odd = pos->dist_odd;
  const __m256i info_even = pos->info_even;
  const __m256i info_odd = pos->info_odd;
  const __m256i rate_a = _mm256_and_si256(s->info_a, c->mask32);
  const __m256i rate_b = _mm256_and_si256(s->info_b, c->mask32);

  // Even candidate: the cheaper of zero and the even level (ties: level).
  const __m256i rd_ea =
      _mm256_add_epi64(s->rd_a, _mm256_add_epi64(tcq_rdc(ea, c), dist_even));
  const __m256i rd_eb =
      _mm256_add_epi64(s->rd_b, _mm256_add_epi64(tcq_rdc(eb, c), dist_even));
  const __m256i rd_za = _mm256_add_epi64(s->rd_a, tcq_rdc(za, c));
  const __m256i rd_zb = _mm256_add_epi64(s->rd_b, tcq_rdc(zb, c));
  const __m256i use_zero_a = _mm256_cmpgt_epi64(rd_ea, rd_za);
  const __m256i use_zero_b = _mm256_cmpgt_epi64(rd_eb, rd_zb);
  const __m256i info_ea = _mm256_add_epi64(
      _mm256_add_epi64(rate_a, ea), _mm256_add_epi64(info_even, c->pid_a));
  const __m256i info_eb = _mm256_add_epi64(
      _mm256_add_epi64(rate_b, eb), _mm256_add_epi64(info_even, c->pid_b));
  const __m256i info_za =
      _mm256_add_epi64(_mm256_add_epi64(rate_a, za), c->pid_a);
  const __m256i info_zb =
      _mm256_add_epi64(_mm256_add_epi64(rate_b, zb), c->pid_b);
  const __m256i ev_rd_a = _mm256_blendv_epi8(rd_ea, rd_za, use_zero_a);
  const __m256i ev_rd_b = _mm256_blendv_epi8(rd_eb, rd_zb, use_zero_b);
  const __m256i ev_info_a = _mm256_blendv_epi8(info_ea, info_za, use_zero_a);
  const __m256i ev_info_b = _mm256_blendv_epi8(info_eb, info_zb, use_zero_b);

  const __m256i od_rd_a =
      _mm256_add_epi64(s->rd_a, _mm256_add_epi64(tcq_rdc(oa, c), dist_odd));
  const __m256i od_rd_b =
      _mm256_add_epi64(s->rd_b, _mm256_add_epi64(tcq_rdc(ob, c), dist_odd));
  const __m256i od_info_a = _mm256_add_epi64(
      _mm256_add_epi64(rate_a, oa), _mm256_add_epi64(info_odd, c->pid_a));
  const __m256i od_info_b = _mm256_add_epi64(
      _mm256_add_epi64(rate_b, ob), _mm256_add_epi64(info_odd, c->pid_b));
  tcq_select(s, ev_rd_a, ev_info_a, ev_rd_b, ev_info_b, od_rd_a, od_info_a,
             od_rd_b, od_info_b);
}

static AVM_FORCE_INLINE void tcq_step_full(TcqRegs *s, const TcqConsts *c,
                                           __m256i teo0, __m256i teo1,
                                           __m256i neg_eo, __m256i tz,
                                           __m256i neg_z,
                                           const TcqFullPos *pos) {
  const __m256i v = tcq_fetch(s, teo0, teo1, neg_eo);
  const __m256i vz = _mm256_blendv_epi8(
      _mm256_permutevar8x32_epi32(tz, s->orig), neg_z, s->orig);
  __m256i ea = _mm256_and_si256(v, c->mask16);
  __m256i oa = _mm256_and_si256(_mm256_srli_epi64(v, 16), c->mask16);
  __m256i eb = _mm256_and_si256(_mm256_srli_epi64(v, 32), c->mask16);
  __m256i ob = _mm256_srli_epi64(v, 48);
  if (pos->has_gol) {
    ea = _mm256_add_epi64(ea, pos->gol_even);
    oa = _mm256_add_epi64(oa, pos->gol_odd);
    eb = _mm256_add_epi64(eb, pos->gol_even);
    ob = _mm256_add_epi64(ob, pos->gol_odd);
  }
  // Zero rates are packed as Q0 | Q1 << 16; lanes 2 and 3 use Q1.
  const __m256i za = _mm256_and_si256(
      _mm256_blend_epi32(vz, _mm256_srli_epi64(vz, 16), 0xF0), c->mask16);
  const __m256i zb =
      _mm256_and_si256(_mm256_blend_epi32(_mm256_srli_epi64(vz, 32),
                                          _mm256_srli_epi64(vz, 48), 0xF0),
                       c->mask16);
  tcq_step_full_rates(s, c, ea, oa, eb, ob, za, zb, pos);
}

// Per-origin tables of the general path, even | odd << 16 for Q0 and Q1. The
// context offsets on AC diagonals are in 0..6; loading entries 0..3 and 3..6
// and separating Q0 / Q1 via shuffle_ps places contexts {0,1,3,4,2,3,5,6} in
// one register per quantizer, reducing 13 vpermd instructions to 2-4.
static AVM_FORCE_INLINE void tcq_tables_full(const uint16_t *low,
                                             const uint16_t *mid, __m128i cb,
                                             __m128i nib8, __m128i nib_mask,
                                             __m256i *teo0, __m256i *teo1,
                                             __m256i *neg_eo) {
  const __m128i ctx_map =
      _mm_setr_epi8(0, 1, 4, 2, 3, 6, 7, 7, 0, 1, 4, 2, 3, 6, 7, 7);
  const __m256i nib_idx = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(ctx_map, nib8));
  const __m256i l0 = _mm256_loadu_si256((const __m256i *)low);
  const __m256i l3 = _mm256_loadu_si256((const __m256i *)(low + 12));
  const __m256i lq0 = _mm256_castps_si256(_mm256_shuffle_ps(
      _mm256_castsi256_ps(l0), _mm256_castsi256_ps(l3), 0x88));
  const __m256i lq1 = _mm256_castps_si256(_mm256_shuffle_ps(
      _mm256_castsi256_ps(l0), _mm256_castsi256_ps(l3), 0xDD));
  __m256i t0 = _mm256_permutevar8x32_epi32(lq0, nib_idx);
  __m256i t1 = _mm256_permutevar8x32_epi32(lq1, nib_idx);
  __m128i z0 = _mm256_castsi256_si128(lq0);
  __m128i z1 = _mm256_castsi256_si128(lq1);
  if (mid) {
    const __m128i mnib8 = _mm_and_si128(_mm_srli_epi16(cb, 4), nib_mask);
    const __m256i mnib_idx =
        _mm256_cvtepu8_epi32(_mm_shuffle_epi8(ctx_map, mnib8));
    const __m256i m0 = _mm256_loadu_si256((const __m256i *)mid);
    const __m256i m3 = _mm256_loadu_si256((const __m256i *)(mid + 12));
    const __m256i mq0 = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(m0), _mm256_castsi256_ps(m3), 0x88));
    const __m256i mq1 = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(m0), _mm256_castsi256_ps(m3), 0xDD));
    t0 = _mm256_add_epi16(t0, _mm256_permutevar8x32_epi32(mq0, mnib_idx));
    t1 = _mm256_add_epi16(t1, _mm256_permutevar8x32_epi32(mq1, mnib_idx));
    z0 = _mm_add_epi16(z0, _mm256_castsi256_si128(mq0));
    z1 = _mm_add_epi16(z1, _mm256_castsi256_si128(mq1));
  }
  *teo0 = t0;
  *teo1 = _mm256_or_si256(_mm256_slli_epi32(t1, 16), _mm256_srli_epi32(t1, 16));
  const __m128i z1_swap = _mm_shufflelo_epi16(z1, 0xB1);
  *neg_eo =
      _mm256_set_m128i(_mm_broadcastd_epi32(z1_swap), _mm_broadcastd_epi32(z0));
}

// Helper for DC lookup of 9-entry (contexts 0..8) low-frequency table.
static AVM_FORCE_INLINE __m256i tcq_dc_lookup9(const uint16_t *tbl,
                                               __m256i base_lane,
                                               __m256i mask8) {
  const __m256i order = _mm256_loadu_si256((const __m256i *)kShufOrder);
  const __m256i la = _mm256_loadu_si256((const __m256i *)tbl);
  const __m256i lb = _mm256_loadu_si256((const __m256i *)(tbl + 16));
  const __m256i q0_07 = _mm256_permutevar8x32_epi32(
      _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(la),
                                            _mm256_castsi256_ps(lb), 0x88)),
      order);
  const __m256i q1_raw = _mm256_permutevar8x32_epi32(
      _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(la),
                                            _mm256_castsi256_ps(lb), 0xDD)),
      order);
  const __m256i q1_07 = _mm256_or_si256(_mm256_slli_epi32(q1_raw, 16),
                                        _mm256_srli_epi32(q1_raw, 16));
  const __m256i q0_8 = _mm256_set1_epi32((int)loadu_uint32(tbl + 32));
  const uint32_t raw8 = loadu_uint32(tbl + 34);
  const __m256i q1_8 = _mm256_set1_epi32((int)((raw8 << 16) | (raw8 >> 16)));
  const __m256i v0 = _mm256_blendv_epi8(
      _mm256_permutevar8x32_epi32(q0_07, base_lane), q0_8, mask8);
  const __m256i v1 = _mm256_blendv_epi8(
      _mm256_permutevar8x32_epi32(q1_07, base_lane), q1_8, mask8);
  return _mm256_blend_epi32(v0, v1, 0xF0);
}

// Vectorized DC position (scan_pos = 0, blk_pos = 0, col = 0) operating
// directly on TcqRegs s in AVX2 registers.
static AVM_FORCE_INLINE __m256i tcq_dc_position_avx2(const tcq_param_t *p,
                                                     const TcqConsts *c,
                                                     TcqRegs *s,
                                                     tcq_ctx_t *tcq_ctx) {
  const LV_MAP_COEFF_COST *tc = p->txb_costs;
  const int log_scale = p->log_scale;
  const int try_eob = p->sharpness == 0;
  const tran_low_t tc0 = p->tcoeff[0];
  const int shift = 16 - log_scale + QUANT_FP_BITS;
  const int dc_sign = tc0 < 0;
  const int orig_qidx = (int16_t)(((int64_t)abs(tc0) * p->quant[0]) >> shift);
  const int dqv = get_dqv(p->dequant, p->scan[0], p->iqmatrix);
  const int dc_add =
      tc->dc_sign_cost[0][p->dc_sign_ctx][dc_sign] - av2_cost_literal(1);
  const __m256i v_dc_add = _mm256_set1_epi64x(dc_add);

  // Per-lane context byte: orig_st == -1 ? 0 : tcq_ctx->ctx[0][orig_st].
  const __m256i ctx_orig =
      _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)tcq_ctx->ctx[0]));
  const __m256i ctx_lane =
      _mm256_andnot_si256(_mm256_srai_epi32(s->orig, 31),
                          _mm256_permutevar8x32_epi32(ctx_orig, s->orig));
  const __m256i base_lane = _mm256_and_si256(ctx_lane, _mm256_set1_epi32(15));
  const __m256i mask8 = _mm256_cmpeq_epi32(base_lane, _mm256_set1_epi32(8));

  // Zero rates for Q0 (lanes 0..3) and Q1 (lanes 4..7), contexts 0..8.
  const __m256i z0_07 = _mm256_cvtepu16_epi32(
      _mm_loadu_si128((const __m128i *)&tc->base_lf_cost_zero[0][0]));
  const __m256i z1_07 = _mm256_cvtepu16_epi32(
      _mm_loadu_si128((const __m128i *)&tc->base_lf_cost_zero[1][0]));
  const __m256i z0 =
      _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(z0_07, base_lane),
                         _mm256_set1_epi32(tc->base_lf_cost_zero[0][8]), mask8);
  const __m256i z1 =
      _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(z1_07, base_lane),
                         _mm256_set1_epi32(tc->base_lf_cost_zero[1][8]), mask8);
  const __m256i z_all = _mm256_blend_epi32(z0, z1, 0xF0);
  const __m256i za = _mm256_and_si256(z_all, c->mask32);
  const __m256i zb = _mm256_srli_epi64(z_all, 32);

  if (orig_qidx < 2) {
    const __m256i v =
        tcq_dc_lookup9(&tc->base_lf_cost_low_tbl[0][0][0][0], base_lane, mask8);
    const __m256i oa = _mm256_add_epi64(
        _mm256_and_si256(_mm256_srli_epi64(v, 16), c->mask16), v_dc_add);
    const __m256i ob = _mm256_add_epi64(_mm256_srli_epi64(v, 48), v_dc_add);
    const int32_t abs_sh = abs(tc0) * (1 << (log_scale - 1));
    const int64_t dqc1 = tcq_dq_level(dqv, 1, log_scale);
    const int64_t dqc2 = tcq_dq_level(dqv, 2, log_scale);
    const int64_t dist2 = dqc2 * (dqc2 - 2 * (int64_t)abs_sh);
    const int64_t dist1_sh =
        dqc1 * (dqc1 - 2 * (int64_t)abs_sh) * (1 << RDDIV_BITS);
    const int64_t dist2_sh = dist2 * (1 << RDDIV_BITS);
    const __m256i dist_odd =
        _mm256_setr_epi64x(dist2_sh, dist2_sh, dist1_sh, dist1_sh);
    tcq_step_q1_rates(s, c, za, oa, zb, ob, dist_odd);
    if (try_eob) {
      const int rate_eob =
          tc->base_lf_eob_cost_tbl[0][0][1] + p->block_eob_rate[0] + dc_add;
      const int64_t eob1_rd = RDCOST(p->rdmult, rate_eob, dist2);
      const int64_t eob1_info =
          rate_eob | (INT64_C(1) << 32) | TCQ_NEW_EOB_PREV_ID;
      tcq_try_eob(s,
                  _mm256_blend_epi32(_mm256_set1_epi64x(INT64_MAX),
                                     _mm256_set1_epi64x(eob1_rd), 0x0C),
                  _mm256_set1_epi64x(eob1_info));
    }
  } else {
    const int qidx = AVMMAX(orig_qidx - 2, 1);
    TcqFullPos ps;
    tcq_prepare_full(p, 0, 0, 1, qidx, c->dq_shift, c->sh_shift,
                     _mm256_set1_epi64x((uint32_t)dqv), c->rdmult, &ps);
    __m256i v = tcq_dc_lookup9(&tc->base_lf_cost_low_tbl[ps.low_idx][0][0][0],
                               base_lane, mask8);
    if (ps.mid_idx >= 0) {
      // mid_ctx is in 0..6, so all entries reside in 0..7.
      const __m256i mid_lane = _mm256_srli_epi32(ctx_lane, 4);
      const uint16_t *mid_tbl = &tc->mid_lf_cost_tbl[ps.mid_idx][0][0][0];
      const __m256i ma = _mm256_loadu_si256((const __m256i *)mid_tbl);
      const __m256i mb = _mm256_loadu_si256((const __m256i *)(mid_tbl + 12));
      const __m256i kMap32 = _mm256_setr_epi32(0, 1, 4, 2, 3, 6, 7, 7);
      const __m256i m_idx = _mm256_permutevar8x32_epi32(kMap32, mid_lane);
      const __m256i mq0 = _mm256_castps_si256(_mm256_shuffle_ps(
          _mm256_castsi256_ps(ma), _mm256_castsi256_ps(mb), 0x88));
      const __m256i mq1_raw = _mm256_castps_si256(_mm256_shuffle_ps(
          _mm256_castsi256_ps(ma), _mm256_castsi256_ps(mb), 0xDD));
      const __m256i mq1 = _mm256_or_si256(_mm256_slli_epi32(mq1_raw, 16),
                                          _mm256_srli_epi32(mq1_raw, 16));
      const __m256i mv =
          _mm256_blend_epi32(_mm256_permutevar8x32_epi32(mq0, m_idx),
                             _mm256_permutevar8x32_epi32(mq1, m_idx), 0xF0);
      v = _mm256_add_epi16(v, mv);
    }
    __m256i ea = _mm256_add_epi64(_mm256_and_si256(v, c->mask16), v_dc_add);
    __m256i oa = _mm256_add_epi64(
        _mm256_and_si256(_mm256_srli_epi64(v, 16), c->mask16), v_dc_add);
    __m256i eb = _mm256_add_epi64(
        _mm256_and_si256(_mm256_srli_epi64(v, 32), c->mask16), v_dc_add);
    __m256i ob = _mm256_add_epi64(_mm256_srli_epi64(v, 48), v_dc_add);
    if (ps.has_gol) {
      ea = _mm256_add_epi64(ea, ps.gol_even);
      oa = _mm256_add_epi64(oa, ps.gol_odd);
      eb = _mm256_add_epi64(eb, ps.gol_even);
      ob = _mm256_add_epi64(ob, ps.gol_odd);
    }
    tcq_step_full_rates(s, c, ea, oa, eb, ob, za, zb, &ps);
    if (try_eob) tcq_try_eob(s, ps.eob_rd, ps.eob_info);
  }
  return tcq_update_lev_prev(s, tcq_ctx->lev_new[0], tcq_ctx->prev_st[0]);
}

// Process one diagonal, from scan_hi down to scan_lo. lf selects the
// low-frequency region tables; the DC position is not handled here.
// When bt_info != NULL, stores compact 32B backtrack rows (L1D-resident)
// instead of 128B tcq_node_t rows.
static AVM_FORCE_INLINE void tcq_diagonal(
    const tcq_param_t *p, const TcqConsts *c, const TcqSegPos *d,
    const TcqDiagTables *t, int seg_lo, int scan_hi, int scan_lo, int blk_pos,
    int col, int diag_ctx, const int lf, TcqRegs *s, tcq_ctx_t *tcq_ctx,
    tcq_node_t *trellis, uint32_t (*bt_info)[8]) {
  const LV_MAP_COEFF_COST *tc = p->txb_costs;
  const int try_eob = p->sharpness == 0;
  const int bd = get_base_diag_ctx(diag_ctx);
  const int md = get_mid_diag_ctx(diag_ctx);
  const __m128i nib_mask = _mm_set1_epi8(15);
  const int blk_pos_inc = (1 << p->bwl) - 1;
  const __m256i inf = _mm256_set1_epi64x(INT64_MAX);
  const int has_qm = p->iqmatrix != NULL;

  assert(scan_hi - scan_lo < MAX_DIAG);
  assert(scan_lo >= seg_lo && scan_hi - seg_lo < TCQ_PREP_SEG);

  for (int i = scan_hi - seg_lo, scan_pos = scan_hi; scan_pos >= scan_lo;
       --scan_pos, --i, --col, blk_pos += blk_pos_inc) {
    const __m128i cb = _mm_loadl_epi64((const __m128i *)tcq_ctx->ctx[col]);
    const __m128i nib8 = _mm_and_si128(cb, nib_mask);
    const __m256i nib = _mm256_cvtepu8_epi32(nib8);
    const int qidx = d->qidx[i];
    if (qidx == 0) {
      const __m256i dist_odd = _mm256_permute4x64_epi64(
          _mm256_castsi128_si256(_mm_load_si128((const __m128i *)d->dist[i])),
          0x50);
      tcq_step_q1(s, c, _mm256_permutevar8x32_epi32(t->lq0, nib),
                  _mm256_permutevar8x32_epi32(t->lq1, nib), t->neg_q1,
                  dist_odd);
      if (try_eob) {
        tcq_try_eob(
            s, _mm256_blend_epi32(inf, _mm256_set1_epi64x(d->eob1_rd[i]), 0x0C),
            _mm256_set1_epi64x(d->eob1_info[i]));
      }
    } else {
      const __m256i v_dqv = has_qm ? _mm256_set1_epi64x((uint32_t)get_dqv(
                                         p->dequant, blk_pos, p->iqmatrix))
                                   : c->v_dqv_ac;
      TcqFullPos ps;
      tcq_prepare_full(p, scan_pos, blk_pos, lf, qidx, c->dq_shift, c->sh_shift,
                       v_dqv, c->rdmult, &ps);
      const uint16_t *low = lf ? &tc->base_lf_cost_low_tbl[ps.low_idx][bd][0][0]
                               : &tc->base_cost_low_tbl[ps.low_idx][bd][0][0];
      const uint16_t *mid =
          ps.mid_idx >= 0 ? (lf ? &tc->mid_lf_cost_tbl[ps.mid_idx][md][0][0]
                                : &tc->mid_cost_tbl[ps.mid_idx][md][0][0])
                          : NULL;
      __m256i teo0, teo1, neg_eo;
      tcq_tables_full(low, mid, cb, nib8, nib_mask, &teo0, &teo1, &neg_eo);
      tcq_step_full(s, c, teo0, teo1, neg_eo,
                    _mm256_permutevar8x32_epi32(t->lz, nib), t->neg_z, &ps);
      if (try_eob) tcq_try_eob(s, ps.eob_rd, ps.eob_info);
    }
    const __m256i hi =
        tcq_update_lev_prev(s, tcq_ctx->lev_new[col], tcq_ctx->prev_st[col]);
    if (bt_info != NULL) {
      _mm256_store_si256((__m256i *)bt_info[scan_pos], hi);
    } else {
      tcq_store_nodes(s, &trellis[scan_pos << TCQ_N_STATES_LOG]);
    }
  }
}

void av2_trellis_loop_diagonal_st8_avx2(const tcq_param_t *p, int scan_hi,
                                        int scan_lo, tcq_ctx_t *tcq_ctx,
                                        tcq_node_t *trellis) {
  (void)scan_lo;
  assert(p->plane == 0);
  assert(p->tx_class == TX_CLASS_2D);
  assert((p->rdmult >> 32) == 0);
  const int16_t *scan = p->scan;
  const int bwl = p->bwl;
  const int height = p->txb_height;

  TcqConsts c;
  tcq_init_consts(p, &c);
  TcqSegPos d;
  TcqRegs s;
  tcq_load_regs(&trellis[(scan_hi + 1) << TCQ_N_STATES_LOG], tcq_ctx, &s);

  int seg_lo = scan_hi + 1;  // no position prepared yet
  TcqDiagTables t;
  memset(&t, 0, sizeof(t));
  int t_key = -1;  // lf << 16 | base context offset of t; -1 = none
  while (scan_hi > 0) {
    const int blk_pos = scan[scan_hi];
    const int row = blk_pos >> bwl;
    const int col = blk_pos - (row << bwl);
    const int inc = AVMMIN(height - 1 - row, col);
    const int lo = scan_hi - inc;
    assert(lo >= 1);
    assert(scan_hi < TCQ_LF_SCAN_END || lo >= TCQ_LF_SCAN_END);
    if (lo < seg_lo) {
      seg_lo = AVMMAX(scan_hi - TCQ_PREP_SEG + 1, 1);
      tcq_prepare_segment(p, seg_lo, scan_hi, &d);
    }
    const int lf = scan_hi < TCQ_LF_SCAN_END;
    const int diag_ctx = get_diag_ctx(lf, blk_pos, lo, bwl);
    const int bd = get_base_diag_ctx(diag_ctx);
    const int key = (lf << 16) | bd;
    if (key != t_key) {
      tcq_diag_tables_bd(p->txb_costs, lf, bd, &t);
      t_key = key;
    }
    if (!lf) {
      tcq_diagonal(p, &c, &d, &t, seg_lo, scan_hi, lo, blk_pos, col, diag_ctx,
                   0, &s, tcq_ctx, trellis, NULL);
    } else {
      tcq_diagonal(p, &c, &d, &t, seg_lo, scan_hi, lo, blk_pos, col, diag_ctx,
                   1, &s, tcq_ctx, trellis, NULL);
    }
    const __m256i orig_bytes = tcq_orig_by_state_bytes(&s);
    update_nbr_diagonal_core_avx2(tcq_ctx, orig_bytes, row + inc, col - inc,
                                  bwl);
    _mm_storeu_si64(&tcq_ctx->orig_st,
                    _mm_loadu_si128((const __m128i *)kConst[2]));
    s.orig = _mm256_load_si256((const __m256i *)kLaneState);
    scan_hi = lo - 1;
  }
  assert(scan_hi == 0);
  (void)tcq_dc_position_avx2(p, &c, &s, tcq_ctx);
  tcq_store_nodes(&s, trellis);
  // Final orig_st in state order for unit-test verification.
  const __m256i orig_bytes = tcq_orig_by_state_bytes(&s);
  _mm_storel_epi64((__m128i *)tcq_ctx->orig_st,
                   _mm256_castsi256_si128(orig_bytes));
}

static AVM_FORCE_INLINE void init_tcq_ctx_avx2(tcq_ctx_t *tcq_ctx,
                                               const tcq_param_t *p, int si) {
  const int blk_pos = p->scan[si];
  const int bwl = p->bwl;
  const int row = blk_pos >> bwl;
  const int col = blk_pos - (row << bwl);
  const int diag = AVMMIN(row + col, MAX_DIAG) + 2;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i init_st = _mm256_loadu_si256((const __m256i *)kConst[2]);
  for (int i = 0; i < diag; i += 4) {
    _mm256_storeu_si256((__m256i *)tcq_ctx->lev_new[i], zero);
    _mm256_storeu_si256((__m256i *)tcq_ctx->mag_base[i], zero);
    _mm256_storeu_si256((__m256i *)tcq_ctx->mag_mid[i], zero);
    _mm256_storeu_si256((__m256i *)tcq_ctx->ctx[i], zero);
    _mm256_storeu_si256((__m256i *)tcq_ctx->prev_st[i], init_st);
  }
}

// Integrated L1D-resident AVX2 2D Luma TCQ pipeline (iqmatrix == NULL):
// - Keeps 32 KB bt_info[MAX_TRELLIS][8] instead of 128 KB tcq_node_t trellis[]
// - Evaluates DC directly in AVX2 registers and reduces the 8 final states
// - Fuses qcoeff/dqcoeff backtrack with txb_entropy_ctx and BOB computation
int av2_trellis_quant_luma_avx2(const tcq_param_t *p, struct macroblock *x,
                                int block, int first_scan_pos, int is_inter,
                                int is_fsc, int bob_code,
                                int reduced_tx_set_used, TX_TYPE tx_type,
                                int txb_skip_ctx, tran_low_t *qcoeff,
                                tran_low_t *dqcoeff, int *rate_cost) {
  MACROBLOCKD *xd = &x->e_mbd;
  const struct macroblock_plane *mbp = &x->plane[0];
  const LV_MAP_COEFF_COST *txb_costs = p->txb_costs;
  const int16_t *scan = p->scan;
  const tran_low_t *tcoeff = p->tcoeff;
  const int log_scale = p->log_scale;
  const int sharpness = p->sharpness;
  const int64_t rdmult = p->rdmult;
  const TX_SIZE tx_size = p->tx_size;

  DECLARE_ALIGNED(32, uint32_t, bt_info[MAX_TRELLIS][8]);
  tcq_node_t first_nodes[TCQ_MAX_STATES];

  int best_state = -2;
  int min_rate = INT32_MAX;
  int64_t min_path_cost = INT64_MAX;

  if (first_scan_pos == 0) {
    av2_trellis_first_pos_node(p, 0, NULL, first_nodes);
    // Only states 0 and 4 are populated at first_scan_pos.
    best_state = 0;
    min_path_cost = first_nodes[0].rdCost;
    min_rate = first_nodes[0].rate;
    if (first_nodes[4].rdCost < min_path_cost) {
      best_state = 4;
      min_path_cost = first_nodes[4].rdCost;
      min_rate = first_nodes[4].rate;
    }
    const int lane0 = kStateLane[0];
    const int lane4 = kStateLane[4];
    bt_info[0][lane0] = ((uint32_t)first_nodes[0].absLevel & 0xFFFFFFu) |
                        ((uint32_t)(uint8_t)first_nodes[0].prevId << 24);
    bt_info[0][lane4] = ((uint32_t)first_nodes[4].absLevel & 0xFFFFFFu) |
                        ((uint32_t)(uint8_t)first_nodes[4].prevId << 24);
  } else {
    tcq_ctx_t tcq_ctx;
    init_tcq_ctx_avx2(&tcq_ctx, p, first_scan_pos);
    av2_trellis_first_pos_node(p, first_scan_pos, &tcq_ctx, first_nodes);

    TcqConsts c;
    tcq_init_consts(p, &c);
    TcqSegPos d;
    TcqRegs s;
    tcq_load_regs(first_nodes, &tcq_ctx, &s);
    const __m256i hi_first =
        _mm256_blend_epi32(_mm256_srli_epi64(s.info_a, 32), s.info_b, 0xAA);
    _mm256_store_si256((__m256i *)bt_info[first_scan_pos], hi_first);

    int scan_hi = first_scan_pos - 1;
    int seg_lo = scan_hi + 1;
    TcqDiagTables t;
    memset(&t, 0, sizeof(t));
    int t_key = -1;
    const int bwl = p->bwl;
    const int height = p->txb_height;
    while (scan_hi > 0) {
      const int blk_pos = scan[scan_hi];
      const int row = blk_pos >> bwl;
      const int col = blk_pos - (row << bwl);
      const int inc = AVMMIN(height - 1 - row, col);
      const int lo = scan_hi - inc;
      if (lo < seg_lo) {
        seg_lo = AVMMAX(scan_hi - TCQ_PREP_SEG + 1, 1);
        tcq_prepare_segment(p, seg_lo, scan_hi, &d);
      }
      const int lf = scan_hi < TCQ_LF_SCAN_END;
      const int diag_ctx = get_diag_ctx(lf, blk_pos, lo, bwl);
      const int bd = get_base_diag_ctx(diag_ctx);
      const int key = (lf << 16) | bd;
      if (key != t_key) {
        tcq_diag_tables_bd(txb_costs, lf, bd, &t);
        t_key = key;
      }
      if (!lf) {
        tcq_diagonal(p, &c, &d, &t, seg_lo, scan_hi, lo, blk_pos, col, diag_ctx,
                     0, &s, &tcq_ctx, NULL, bt_info);
      } else {
        tcq_diagonal(p, &c, &d, &t, seg_lo, scan_hi, lo, blk_pos, col, diag_ctx,
                     1, &s, &tcq_ctx, NULL, bt_info);
      }
      const __m256i orig_bytes = tcq_orig_by_state_bytes(&s);
      update_nbr_diagonal_core_avx2(&tcq_ctx, orig_bytes, row + inc, col - inc,
                                    bwl);
      s.orig = _mm256_load_si256((const __m256i *)kLaneState);
      scan_hi = lo - 1;
    }
    const __m256i hi_dc = tcq_dc_position_avx2(p, &c, &s, &tcq_ctx);
    _mm256_store_si256((__m256i *)bt_info[0], hi_dc);

    DECLARE_ALIGNED(32, int64_t, rd_arr[8]);
    DECLARE_ALIGNED(32, int32_t, rate_arr[8]);
    // States in a: {0, 4, 6, 2}, states in b: {1, 5, 7, 3}.
    _mm256_store_si256((__m256i *)&rd_arr[0], s.rd_a);
    _mm256_store_si256((__m256i *)&rd_arr[4], s.rd_b);
    const __m256i rates_ab =
        _mm256_blend_epi32(s.info_a, _mm256_slli_epi64(s.info_b, 32), 0xAA);
    _mm256_store_si256((__m256i *)rate_arr, rates_ab);
    static const uint8_t kStateRdIdx[8] = { 0, 4, 3, 7, 1, 5, 2, 6 };
    static const uint8_t kStateRateIdx[8] = { 0, 1, 6, 7, 2, 3, 4, 5 };
    for (int st = 0; st < TCQ_N_STATES; ++st) {
      const int64_t rdc = rd_arr[kStateRdIdx[st]];
      if (rdc < min_path_cost) {
        min_path_cost = rdc;
        min_rate = rate_arr[kStateRateIdx[st]];
        best_state = st;
      }
    }
  }

  // Backtrack in GPRs while accumulating cul_level and first non-zero (bob).
  int scan_pos = 0;
  int prev_id = best_state;
  int cul_level = 0;
  int first_nz = -1;
  uint32_t dqv = (uint32_t)p->dequant[0];
  const uint32_t dqv_ac = (uint32_t)p->dequant[1];
  const uint64_t round = 1ULL << (QUANT_TABLE_BITS - 1);
  const int shift = QUANT_TABLE_BITS + log_scale;
  for (; prev_id >= 0; ++scan_pos) {
    const int32_t info = (int32_t)bt_info[scan_pos][kStateLane[prev_id]];
    const int next_id = info >> 24;
    const int abs_lev = (int)((uint32_t)info & 0xFFFFFFu);
    const int blk_pos = scan[scan_pos];
    const int sign = tcoeff[blk_pos] >> 31;
    const int dq = (next_id >= 0) ? ((next_id >> 1) & 1) : 0;
    const uint32_t qc = abs_lev ? (uint32_t)(2 * abs_lev - dq) : 0u;
    const int dqc = (int)(((uint64_t)qc * dqv + round) >> shift);
    qcoeff[blk_pos] = (abs_lev ^ sign) - sign;
    dqcoeff[blk_pos] = (dqc ^ sign) - sign;
    cul_level += abs_lev;
    if (first_nz < 0 && abs_lev != 0) first_nz = scan_pos;
    dqv = dqv_ac;
    prev_id = next_id;
  }
  int eob = scan_pos;
  for (; scan_pos <= first_scan_pos; ++scan_pos) {
    const int blk_pos = scan[scan_pos];
    qcoeff[blk_pos] = 0;
    dqcoeff[blk_pos] = 0;
  }

  // Matches set_bob() called before skip-block check in av2_trellis_quant().
  mbp->bobs[block] = av2_get_max_eob(tx_size) -
                     (first_nz >= 0 ? first_nz : (first_scan_pos + 1));

  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int pred_mode_ctx =
      (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
  const int non_skip_cost =
      txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
  const int skip_cost =
      txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
  const int tx_type_cost = get_tx_type_cost(
      x, xd, 0, tx_size, tx_type, reduced_tx_set_used, eob, bob_code, is_fsc);
  const int64_t rd_cost_skip = RDCOST(rdmult, skip_cost, 0);
  int accu_rate = non_skip_cost + tx_type_cost + min_rate;
  const int64_t rd_cost_coded =
      min_path_cost + (int64_t)RDCOST(rdmult, non_skip_cost + tx_type_cost, 0);

  if ((rd_cost_coded > rd_cost_skip) && sharpness == 0) {
    for (int scan_idx = 0; scan_idx < eob; ++scan_idx) {
      const int blk_idx = scan[scan_idx];
      qcoeff[blk_idx] = 0;
      dqcoeff[blk_idx] = 0;
    }
    accu_rate = skip_cost;
    eob = 0;
    cul_level = 0;
  } else {
    cul_level = AVMMIN(COEFF_CONTEXT_MASK, cul_level);
    set_dc_sign(&cul_level, qcoeff[0]);
  }

  mbp->eobs[block] = (uint16_t)eob;
  mbp->txb_entropy_ctx[block] = (uint8_t)cul_level;
  *rate_cost = accu_rate;
  return eob;
}
