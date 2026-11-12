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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"
#include "test/register_state_check.h"
#include "test/function_equivalence_test.h"

#include "config/avm_config.h"
#include "config/avm_dsp_rtcd.h"
#include "config/av2_rtcd.h"

#include "avm/avm_integer.h"
#include "avm_ports/avm_timer.h"
#include "av2/common/enums.h"

using libavm_test::FunctionEquivalenceTest;

namespace {

template <typename F, typename T>
class FilterEdgeTest : public FunctionEquivalenceTest<F> {
 protected:
  static const int kIterations = 1000000;
  static const int kMaxEdge = 2 * 64;
  static const int kBufSize = kMaxEdge + 64;
  static const int kOffset = 15;

  virtual ~FilterEdgeTest() {}

  virtual void Execute(T *edge_tst) = 0;

  void Common() {
    edge_ref_ = &edge_ref_data_[kOffset];
    edge_tst_ = &edge_tst_data_[kOffset];

    Execute(edge_tst_);

    for (int r = 0; r < size_; ++r) {
      ASSERT_EQ(edge_ref_[r], edge_tst_[r]);
    }
  }

  T edge_ref_data_[kBufSize];
  T edge_tst_data_[kBufSize];

  T *edge_ref_;
  T *edge_tst_;

  int size_;
  int strength_;
};

//////////////////////////////////////////////////////////////////////////////
// High bit-depth version
//////////////////////////////////////////////////////////////////////////////

typedef void (*FEHB)(uint16_t *p, int size, int strength);
typedef libavm_test::FuncParam<FEHB> FilterEdgeTestFuncsHBD;

class FilterEdgeTestHB : public FilterEdgeTest<FEHB, uint16_t> {
 protected:
  void Execute(uint16_t *edge_tst) {
    params_.ref_func(edge_ref_, size_, strength_);
    ASM_REGISTER_STATE_CHECK(params_.tst_func(edge_tst, size_, strength_));
  }
  int bit_depth_;
};

TEST_P(FilterEdgeTestHB, RandomValues) {
  for (int iter = 0; iter < kIterations && !HasFatalFailure(); ++iter) {
    switch (rng_(3)) {
      case 0: bit_depth_ = 8; break;
      case 1: bit_depth_ = 10; break;
      default: bit_depth_ = 12; break;
    }
    const int hi = 1 << bit_depth_;
    strength_ = this->rng_(4);
    size_ = 4 * (this->rng_(128 / 4) + 1) + 1;

    int i, pix = 0;
    for (i = 0; i < kOffset + size_; ++i) {
      pix = rng_(hi);
      edge_ref_data_[i] = pix;
      edge_tst_data_[i] = pix;
    }

    Common();
  }
}

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE4_1, FilterEdgeTestHB,
                         ::testing::Values(FilterEdgeTestFuncsHBD(
                             av2_filter_intra_edge_high_c,
                             av2_filter_intra_edge_high_sse4_1)));
#endif  // HAVE_SSE4_1

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, FilterEdgeTestHB,
    ::testing::Values(FilterEdgeTestFuncsHBD(av2_filter_intra_edge_high_c,
                                             av2_filter_intra_edge_high_avx2)));
#endif  // HAVE_AVX2

// Speed tests

TEST_P(FilterEdgeTestHB, DISABLED_Speed) {
  const int num_iters = 1000000;
  bit_depth_ = 12;
  const int sample_range = 1 << bit_depth_;
  for (int i = 0; i < kOffset + kMaxEdge + 1; ++i) {
    const uint16_t sample_val = rng_(sample_range);
    edge_ref_data_[i] = sample_val;
    edge_tst_data_[i] = sample_val;
  }
  edge_ref_ = &edge_ref_data_[kOffset];
  edge_tst_ = &edge_tst_data_[kOffset];

  const int kEdgeLengths[] = { 5, 9, 17, 33, 65, 129 };
  double log_speedup_sum = 0.0;
  for (int size_idx = 0; size_idx < 6; ++size_idx) {
    const int edge_len = kEdgeLengths[size_idx];
    avm_usec_timer ref_timer;
    avm_usec_timer_start(&ref_timer);
    int strength = 1;
    for (int iter = 0; iter < num_iters; ++iter) {
      params_.ref_func(edge_ref_, edge_len, strength);
      strength = (strength == 3) ? 1 : strength + 1;
    }
    avm_usec_timer_mark(&ref_timer);
    const double ref_elapsed_us =
        static_cast<double>(avm_usec_timer_elapsed(&ref_timer));

    avm_usec_timer tst_timer;
    avm_usec_timer_start(&tst_timer);
    strength = 1;
    for (int iter = 0; iter < num_iters; ++iter) {
      params_.tst_func(edge_tst_, edge_len, strength);
      strength = (strength == 3) ? 1 : strength + 1;
    }
    avm_usec_timer_mark(&tst_timer);
    const double tst_elapsed_us =
        static_cast<double>(avm_usec_timer_elapsed(&tst_timer));
    const double speedup = ref_elapsed_us / tst_elapsed_us;
    log_speedup_sum += log(speedup);
    printf("FilterEdgeHB len=%3d: C=%7.1fms SIMD=%7.1fms speedup=%.2fx\n",
           edge_len, ref_elapsed_us / 1000.0, tst_elapsed_us / 1000.0, speedup);
  }
  printf("FilterEdgeHB geomean: %.2fx\n", exp(log_speedup_sum / 6.0));
}

}  // namespace
