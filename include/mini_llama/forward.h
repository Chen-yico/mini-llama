// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// GQA: map a query head onto the shared KV head.
int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads);

// Single-token forward: write KV at ctx.pos and return logits [vocab_size].
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token);

// Sequential scheduler over MiniBatch. Prefill and Decode share this path.
// Returns logits of the last token and appends tokens to ctx.token_history.
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace mini_llama
