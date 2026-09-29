// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <cmath>

#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestQ80BlockLayout() {
  MINI_LLAMA_ASSERT_EQ(kQ80BlockSize, 32);
  MINI_LLAMA_ASSERT_EQ(sizeof(BlockQ80), 34);
  return true;
}

static bool TestQ80RoundtripIdentity() {
  Tensor src({32}, 0.0f);
  for (int i = 0; i < 32; ++i) {
    src.data[i] = static_cast<float>(i) * 0.1f;
  }

  auto blocks = QuantizeToQ80(src);
  MINI_LLAMA_ASSERT_EQ(blocks.size(), 1);

  Tensor dst = DequantizeFromQ80(blocks, src.shape);
  MINI_LLAMA_ASSERT_EQ(dst.shape.size(), 1);
  MINI_LLAMA_ASSERT_EQ(dst.shape[0], 32);

  float max_err = 0.0f;
  for (size_t i = 0; i < src.size(); ++i) {
    float err = std::abs(src.data[i] - dst.data[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  MINI_LLAMA_ASSERT_TRUE(max_err < 6e-2f);
  return true;
}

static bool TestQ80AllZeros() {
  Tensor src({64}, 0.0f);
  auto blocks = QuantizeToQ80(src);
  MINI_LLAMA_ASSERT_EQ(blocks.size(), 2);

  Tensor dst = DequantizeFromQ80(blocks, src.shape);
  for (size_t i = 0; i < dst.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(dst.data[i], 0.0f, 1e-6f);
  }
  return true;
}

static bool TestQ80MultiBlock() {
  Tensor src({100}, 0.0f);
  for (int i = 0; i < 100; ++i) {
    src.data[i] = static_cast<float>(i) * 0.05f;
  }

  auto blocks = QuantizeToQ80(src);
  MINI_LLAMA_ASSERT_EQ(blocks.size(), 4);

  Tensor dst = DequantizeFromQ80(blocks, src.shape);
  MINI_LLAMA_ASSERT_EQ(dst.shape[0], 100);

  float max_err = 0.0f;
  for (size_t i = 0; i < src.size(); ++i) {
    float err = std::abs(src.data[i] - dst.data[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  MINI_LLAMA_ASSERT_TRUE(max_err < 6e-2f);
  return true;
}

static bool TestQ80EmptyTensor() {
  try {
    Tensor src({0}, 0.0f);
    (void)src;
    MINI_LLAMA_ASSERT_FAIL("expected exception for zero-dimension tensor");
  } catch (const std::runtime_error&) {
  }
  return true;
}

static bool TestMatmulQ80MatchesF32() {
  Tensor a({2, 3}, 0.0f);
  a.data[0] = 1.0f;
  a.data[1] = 2.0f;
  a.data[2] = 3.0f;
  a.data[3] = 4.0f;
  a.data[4] = 5.0f;
  a.data[5] = 6.0f;

  Tensor b({3, 2}, 0.0f);
  b.data[0] = 1.0f;
  b.data[1] = 2.0f;
  b.data[2] = 3.0f;
  b.data[3] = 4.0f;
  b.data[4] = 5.0f;
  b.data[5] = 6.0f;

  auto qA = QuantizeToQ80(a);
  Tensor c_f32 = Matmul(a, b);
  Tensor c_q8 = MatmulQ80(qA, b, a.shape);

  MINI_LLAMA_ASSERT_EQ(c_f32.shape.size(), 2);
  MINI_LLAMA_ASSERT_EQ(c_q8.shape.size(), 2);
  MINI_LLAMA_ASSERT_EQ(c_f32.shape[0], c_q8.shape[0]);
  MINI_LLAMA_ASSERT_EQ(c_f32.shape[1], c_q8.shape[1]);

  for (size_t i = 0; i < c_f32.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(c_f32.data[i], c_q8.data[i], 2e-1f);
  }
  return true;
}

static bool TestCompareMatmulError() {
  Tensor a({16, 16}, 0.0f);
  Tensor b({16, 16}, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    a.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
  }

  float err = CompareMatmulError(a, b);
  MINI_LLAMA_ASSERT_TRUE(err >= 0.0f);
  MINI_LLAMA_ASSERT_TRUE(err < 1e-1f);
  return true;
}

static bool TestDequantizeRejectsBadShape() {
  Tensor src({32}, 1.0f);
  auto blocks = QuantizeToQ80(src);

  try {
    (void)DequantizeFromQ80(blocks, {-1});
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative shape");
  } catch (const std::runtime_error&) {
  }

  try {
    (void)DequantizeFromQ80(blocks, {0});
    MINI_LLAMA_ASSERT_FAIL("expected exception for zero shape");
  } catch (const std::runtime_error&) {
  }

  return true;
}

static bool TestDequantizeRejectsBlockCountMismatch() {
  Tensor src({64}, 1.0f);
  auto blocks = QuantizeToQ80(src);
  blocks.pop_back();

  try {
    (void)DequantizeFromQ80(blocks, src.shape);
    MINI_LLAMA_ASSERT_FAIL("expected exception for block count mismatch");
  } catch (const std::runtime_error&) {
  }
  return true;
}

static struct QuantTestRegistrar {
  QuantTestRegistrar() {
    RegisterTest("q8_0_block_layout", TestQ80BlockLayout);
    RegisterTest("q8_0_roundtrip_identity", TestQ80RoundtripIdentity);
    RegisterTest("q8_0_all_zeros", TestQ80AllZeros);
    RegisterTest("q8_0_multi_block", TestQ80MultiBlock);
    RegisterTest("q8_0_empty_tensor", TestQ80EmptyTensor);
    RegisterTest("matmul_q8_0_matches_f32", TestMatmulQ80MatchesF32);
    RegisterTest("CompareMatmulError", TestCompareMatmulError);
    RegisterTest("dequantize_rejects_bad_shape", TestDequantizeRejectsBadShape);
    RegisterTest("dequantize_rejects_block_count_mismatch",
                 TestDequantizeRejectsBlockCountMismatch);
  }
} quant_test_registrar;
