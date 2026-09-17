// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestBatchSingle() {
  MiniBatch b = MiniBatch::Single(42, 5);
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 1);
  MINI_LLAMA_ASSERT_EQ(b.tokens[0], 42);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 5);
  return true;
}

static bool TestBatchFromTokens() {
  std::vector<int> toks = {10, 20, 30};
  MiniBatch b = MiniBatch::FromTokens(toks, 0);
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 3);
  MINI_LLAMA_ASSERT_EQ(b.tokens[0], 10);
  MINI_LLAMA_ASSERT_EQ(b.tokens[1], 20);
  MINI_LLAMA_ASSERT_EQ(b.tokens[2], 30);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 0);
  MINI_LLAMA_ASSERT_EQ(b.positions[1], 1);
  MINI_LLAMA_ASSERT_EQ(b.positions[2], 2);
  return true;
}

static bool TestBatchFromTokensWithOffset() {
  std::vector<int> toks = {5, 6};
  MiniBatch b = MiniBatch::FromTokens(toks, 10);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 10);
  MINI_LLAMA_ASSERT_EQ(b.positions[1], 11);
  return true;
}

static bool TestBatchEmpty() {
  MiniBatch b;
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 0);
  return true;
}

static bool TestForwardBatchPrefillMatchesIndividual() {
  MiniLlamaModel model = MakeCpuTestModel();
  const std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

  MiniLlamaContext ctx1(&model);
  Tensor logits1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    ctx1.pos = static_cast<int>(i);
    logits1 = ForwardToken(ctx1, model, tokens[i]);
  }

  MiniLlamaContext ctx2(&model);
  const MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  MINI_LLAMA_ASSERT_EQ(logits1.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits2.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-5f);
  }
  return true;
}

static bool TestForwardBatchSingleMatchesIndividual() {
  MiniLlamaModel model = MakeCpuTestModel();

  MiniLlamaContext ctx1(&model);
  ctx1.pos = 3;
  Tensor logits1 = ForwardToken(ctx1, model, 42);

  MiniLlamaContext ctx2(&model);
  const MiniBatch batch = MiniBatch::Single(42, 3);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-5f);
  }
  return true;
}

static bool TestForwardBatchUpdatesContext() {
  MiniLlamaModel model = MakeCpuTestModel();
  const std::vector<int> tokens = {1, 104, 101};
  MiniLlamaContext ctx(&model);
  ForwardBatch(ctx, model, MiniBatch::FromTokens(tokens, 0));

  MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[i], tokens[i]);
  }
  MINI_LLAMA_ASSERT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
  return true;
}

static bool TestForwardBatchAppendsDecodeHistory() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  ForwardBatch(ctx, model, MiniBatch::FromTokens({1, 104}, 0));
  ForwardBatch(ctx, model, MiniBatch::Single(101, 2));

  MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), 3);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[0], 1);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[1], 104);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[2], 101);
  MINI_LLAMA_ASSERT_EQ(ctx.pos, 2);
  return true;
}

static bool TestForwardBatchEmptyRejected() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  try {
    ForwardBatch(ctx, model, batch);
    MINI_LLAMA_ASSERT_FAIL("expected exception for empty batch");
  } catch (const std::runtime_error&) {
    return true;
  }
}

static bool TestForwardBatchMismatchedSizesRejected() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  batch.tokens = {1, 2};
  batch.positions = {0};
  try {
    ForwardBatch(ctx, model, batch);
    MINI_LLAMA_ASSERT_FAIL("expected exception for size mismatch");
  } catch (const std::runtime_error&) {
    return true;
  }
}

static struct BatchTestRegistrar {
  BatchTestRegistrar() {
    RegisterTest("batch_single", TestBatchSingle);
    RegisterTest("batch_from_tokens", TestBatchFromTokens);
    RegisterTest("batch_from_tokens_with_offset", TestBatchFromTokensWithOffset);
    RegisterTest("batch_empty", TestBatchEmpty);
    RegisterTest("forward_batch_prefill_matches_individual",
                 TestForwardBatchPrefillMatchesIndividual);
    RegisterTest("forward_batch_single_matches_individual",
                 TestForwardBatchSingleMatchesIndividual);
    RegisterTest("forward_batch_updates_context",
                 TestForwardBatchUpdatesContext);
    RegisterTest("forward_batch_appends_decode_history",
                 TestForwardBatchAppendsDecodeHistory);
    RegisterTest("forward_batch_empty_rejected", TestForwardBatchEmptyRejected);
    RegisterTest("forward_batch_mismatched_sizes_rejected",
                 TestForwardBatchMismatchedSizesRejected);
  }
} batch_test_registrar;
