// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Session state: KV cache, position cursor, token history, and counters.
struct MiniLlamaContext {
  const MiniLlamaModel* model = nullptr;
  KvCache kv_cache;
  int pos = 0;
  std::vector<int> token_history;
  int n_prefill_tokens = 0;
  int n_decode_tokens = 0;

  MiniLlamaContext() = default;
  MiniLlamaContext(const MiniLlamaContext&) = delete;
  MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
  MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
  MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
  explicit MiniLlamaContext(const MiniLlamaModel* model);
};

// Longest shared prefix of two token sequences. Used to align a new prompt
// with ctx.token_history before reusing the linear KV cache.
size_t CommonPrefixLength(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace mini_llama
