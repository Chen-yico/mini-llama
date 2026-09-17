// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/context.h"

#include <algorithm>

namespace mini_llama {

MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model_ptr)
    : model(model_ptr), pos(0) {
  if (model != nullptr) {
    const ModelConfig& config = model->config;
    kv_cache = KvCache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                       config.head_dim);
  }
}

size_t CommonPrefixLength(const std::vector<int>& a,
                          const std::vector<int>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) {
    ++i;
  }
  return i;
}

}  // namespace mini_llama
