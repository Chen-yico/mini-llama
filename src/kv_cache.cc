// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mini_llama {

static void Require4DCacheShape(const Tensor& t, const char* name) {
  if (t.num_dims() != 4) {
    throw std::runtime_error(std::string(name) + " must be a 4D tensor");
  }
}

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
  keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
  values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
  Require4DCacheShape(keys, "keys");
  Require4DCacheShape(values, "values");
  if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
    throw std::out_of_range(
        "KvCache::Write expected matching [n_kv_heads, head_dim] tensors");
  }
  const int n_kv_heads = k.shape[0];
  const int head_dim = k.shape[1];
  if (layer < 0 || layer >= keys.shape[0]) {
    throw std::out_of_range("KvCache::Write layer out of range");
  }
  if (pos < 0 || pos >= keys.shape[1]) {
    throw std::out_of_range("KvCache::Write position out of range");
  }
  if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
    throw std::out_of_range(
        "KvCache::Write tensor shape does not match cache shape");
  }

  const size_t layer_stride =
      static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
  const size_t pos_stride =
      static_cast<size_t>(keys.shape[2] * keys.shape[3]);
  const size_t head_stride = static_cast<size_t>(head_dim);
  const size_t base = static_cast<size_t>(layer) * layer_stride +
                      static_cast<size_t>(pos) * pos_stride;

  for (int h = 0; h < n_kv_heads; ++h) {
    const size_t offset = base + static_cast<size_t>(h) * head_stride;
    std::memcpy(&keys.data[offset], &k.data[static_cast<size_t>(h) * head_dim],
                static_cast<size_t>(head_dim) * sizeof(float));
    std::memcpy(&values.data[offset],
                &v.data[static_cast<size_t>(h) * head_dim],
                static_cast<size_t>(head_dim) * sizeof(float));
  }
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
  Require4DCacheShape(keys, "keys");
  if (layer < 0 || layer >= keys.shape[0] || pos < 0 || pos >= keys.shape[1] ||
      kv_head < 0 || kv_head >= keys.shape[2]) {
    throw std::out_of_range("KvCache::KeyPtr index out of range");
  }
  const size_t layer_stride =
      static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
  const size_t pos_stride =
      static_cast<size_t>(keys.shape[2] * keys.shape[3]);
  const size_t head_stride = static_cast<size_t>(keys.shape[3]);
  return &keys.data[static_cast<size_t>(layer) * layer_stride +
                    static_cast<size_t>(pos) * pos_stride +
                    static_cast<size_t>(kv_head) * head_stride];
}

const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
  Require4DCacheShape(values, "values");
  if (layer < 0 || layer >= values.shape[0] || pos < 0 ||
      pos >= values.shape[1] || kv_head < 0 || kv_head >= values.shape[2]) {
    throw std::out_of_range("KvCache::ValuePtr index out of range");
  }
  const size_t layer_stride =
      static_cast<size_t>(values.shape[1] * values.shape[2] * values.shape[3]);
  const size_t pos_stride =
      static_cast<size_t>(values.shape[2] * values.shape[3]);
  const size_t head_stride = static_cast<size_t>(values.shape[3]);
  return &values.data[static_cast<size_t>(layer) * layer_stride +
                      static_cast<size_t>(pos) * pos_stride +
                      static_cast<size_t>(kv_head) * head_stride];
}

}  // namespace mini_llama
