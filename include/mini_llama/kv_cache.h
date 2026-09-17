// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include "mini_llama/tensor.h"

namespace mini_llama {

// KV cache stores keys and values for all layers and positions.
// keys:   [n_layers, max_seq_len, n_kv_heads, head_dim]
// values: [n_layers, max_seq_len, n_kv_heads, head_dim]
struct KvCache {
  Tensor keys;
  Tensor values;

  KvCache() = default;
  KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

  // Write k and v for a specific layer and position.
  // k: [n_kv_heads, head_dim]
  // v: [n_kv_heads, head_dim]
  void Write(int layer, int pos, const Tensor& k, const Tensor& v);

  // Zero-copy view of keys[layer, pos, kv_head, :] as head_dim floats.
  const float* KeyPtr(int layer, int pos, int kv_head) const;
  const float* ValuePtr(int layer, int pos, int kv_head) const;
};

}  // namespace mini_llama
