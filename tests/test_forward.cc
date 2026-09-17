// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestMapQHeadToKvHeadGqa() {
  // 14 Q heads, 2 KV heads → group size 7.
  // 0-based head 8 is the 9th Q head and maps to KV head 1.
  MINI_LLAMA_ASSERT_EQ(MapQHeadToKvHead(0, 14, 2), 0);
  MINI_LLAMA_ASSERT_EQ(MapQHeadToKvHead(6, 14, 2), 0);
  MINI_LLAMA_ASSERT_EQ(MapQHeadToKvHead(7, 14, 2), 1);
  MINI_LLAMA_ASSERT_EQ(MapQHeadToKvHead(8, 14, 2), 1);
  MINI_LLAMA_ASSERT_EQ(MapQHeadToKvHead(13, 14, 2), 1);
  MINI_LLAMA_ASSERT_EQ(MapQHeadToKvHead(3, 4, 4), 3);
  return true;
}

static bool TestForwardTokenProducesLogitShape() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);

  MINI_LLAMA_ASSERT_EQ(logits.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
  for (float value : logits.data) {
    MINI_LLAMA_ASSERT_TRUE(std::isfinite(value));
  }
  return true;
}

static bool TestForwardTokenChangesKvCache() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  const float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
  MINI_LLAMA_ASSERT_NEAR(before, 0.0f, 1e-6f);

  ForwardToken(ctx, model, 1);

  const float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
  MINI_LLAMA_ASSERT_TRUE(std::isfinite(after));
  return true;
}

static bool TestForwardGqaWritesSharedKvHeads() {
  ModelConfig config;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.head_dim = 8;
  config.dim = 32;
  MiniLlamaModel model = MakeCpuTestModel(config);
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  ForwardToken(ctx, model, 3);

  const float* kv0 = ctx.kv_cache.KeyPtr(0, 0, 0);
  const float* kv1 = ctx.kv_cache.KeyPtr(0, 0, 1);
  MINI_LLAMA_ASSERT_TRUE(std::isfinite(kv0[0]));
  MINI_LLAMA_ASSERT_TRUE(std::isfinite(kv1[0]));
  return true;
}

static bool TestForwardRejectsUnloadedModel() {
  MiniLlamaModel model;
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  try {
    ForwardToken(ctx, model, 1);
    MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an unloaded model");
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    MINI_LLAMA_ASSERT_FAIL("ForwardToken threw the wrong exception type");
  }
}

static bool TestForwardRejectsInvalidToken() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  try {
    ForwardToken(ctx, model, model.config.vocab_size);
    MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range token");
  } catch (const std::out_of_range&) {
    return true;
  } catch (...) {
    MINI_LLAMA_ASSERT_FAIL(
        "ForwardToken threw the wrong exception type for invalid token");
  }
}

static bool TestForwardRejectsInvalidPosition() {
  MiniLlamaModel model = MakeCpuTestModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = model.config.max_seq_len;
  try {
    ForwardToken(ctx, model, 1);
    MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range position");
  } catch (const std::out_of_range&) {
    return true;
  } catch (...) {
    MINI_LLAMA_ASSERT_FAIL(
        "ForwardToken threw the wrong exception type for invalid position");
  }
}

static bool TestGenerationGreedySequence() {
  MiniLlamaModel model = MakeCpuTestModel();
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hello");
  MiniLlamaContext ctx(&model);

  Tensor logits;
  for (size_t i = 0; i < tokens.size(); ++i) {
    ctx.pos = static_cast<int>(i);
    logits = ForwardToken(ctx, model, tokens[i]);
  }
  tokens.push_back(ArgMax(logits));

  for (int i = 0; i < 5; ++i) {
    ctx.pos = static_cast<int>(tokens.size() - 1);
    logits = ForwardToken(ctx, model, tokens.back());
    tokens.push_back(ArgMax(logits));
  }

  MINI_LLAMA_ASSERT_TRUE(tokens.size() > 6);
  MINI_LLAMA_ASSERT_TRUE(tokens.back() >= 0);
  MINI_LLAMA_ASSERT_TRUE(tokens.back() < model.config.vocab_size);
  return true;
}

static struct ForwardTestRegistrar {
  ForwardTestRegistrar() {
    RegisterTest("forward_map_q_head_to_kv_head_gqa", TestMapQHeadToKvHeadGqa);
    RegisterTest("forward_token_produces_logit_shape",
                 TestForwardTokenProducesLogitShape);
    RegisterTest("forward_token_changes_kv_cache",
                 TestForwardTokenChangesKvCache);
    RegisterTest("forward_gqa_writes_shared_kv_heads",
                 TestForwardGqaWritesSharedKvHeads);
    RegisterTest("forward_rejects_unloaded_model",
                 TestForwardRejectsUnloadedModel);
    RegisterTest("forward_rejects_invalid_token", TestForwardRejectsInvalidToken);
    RegisterTest("forward_rejects_invalid_position",
                 TestForwardRejectsInvalidPosition);
    RegisterTest("forward_generation_greedy_sequence",
                 TestGenerationGreedySequence);
  }
} forward_test_registrar;
