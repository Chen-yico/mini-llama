// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/model.h"

#include <stdexcept>

#include "mini_llama/quantized_tensor.h"

namespace mini_llama {
namespace {

void FillDeterministic(Tensor& tensor, float scale) {
  for (size_t i = 0; i < tensor.data.size(); ++i) {
    const float unit = static_cast<float>((i * 17 + 13) % 100) / 100.0f;
    tensor.data[i] = scale * (unit - 0.5f);
  }
}

void FillOnes(Tensor& tensor) {
  for (float& value : tensor.data) {
    value = 1.0f;
  }
}

void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.head_dim <= 0 || config.max_seq_len <= 0) {
    throw std::runtime_error("MakeCpuTestModel: invalid model config");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error(
        "MakeCpuTestModel: n_heads must be divisible by n_kv_heads");
  }
  if (config.dim != config.n_heads * config.head_dim) {
    throw std::runtime_error(
        "MakeCpuTestModel: dim must equal n_heads * head_dim");
  }
  if (config.head_dim % 2 != 0) {
    throw std::runtime_error("MakeCpuTestModel: head_dim must be even for RoPE");
  }
}

}  // namespace

MiniLlamaModel MakeCpuTestModel(ModelConfig config) {
  ValidateConfig(config);

  MiniLlamaModel model;
  model.config = config;
  model.token_embedding = Tensor({config.vocab_size, config.dim}, 0.0f);
  model.final_norm = Tensor({config.dim}, 0.0f);
  Tensor lm_head({config.vocab_size, config.dim}, 0.0f);
  FillDeterministic(model.token_embedding, 0.05f);
  FillOnes(model.final_norm);
  FillDeterministic(lm_head, 0.02f);
  model.lm_head = ToQuantizedTensor(lm_head);

  model.layers.resize(static_cast<size_t>(config.n_layers));
  const int q_out = config.n_heads * config.head_dim;
  const int kv_out = config.n_kv_heads * config.head_dim;
  for (LayerWeights& layer : model.layers) {
    layer.attention_norm = Tensor({config.dim}, 0.0f);
    layer.ffn_norm = Tensor({config.dim}, 0.0f);
    Tensor wq({q_out, config.dim}, 0.0f);
    Tensor wk({kv_out, config.dim}, 0.0f);
    Tensor wv({kv_out, config.dim}, 0.0f);
    Tensor wo({config.dim, q_out}, 0.0f);
    Tensor w_gate({config.hidden_dim, config.dim}, 0.0f);
    Tensor w_up({config.hidden_dim, config.dim}, 0.0f);
    Tensor w_down({config.dim, config.hidden_dim}, 0.0f);

    FillOnes(layer.attention_norm);
    FillOnes(layer.ffn_norm);
    FillDeterministic(wq, 0.03f);
    FillDeterministic(wk, 0.03f);
    FillDeterministic(wv, 0.03f);
    FillDeterministic(wo, 0.03f);
    FillDeterministic(w_gate, 0.03f);
    FillDeterministic(w_up, 0.03f);
    FillDeterministic(w_down, 0.03f);
    layer.wq = ToQuantizedTensor(wq);
    layer.wk = ToQuantizedTensor(wk);
    layer.wv = ToQuantizedTensor(wv);
    layer.wo = ToQuantizedTensor(wo);
    layer.w_gate = ToQuantizedTensor(w_gate);
    layer.w_up = ToQuantizedTensor(w_up);
    layer.w_down = ToQuantizedTensor(w_down);
  }

  model.loaded = true;
  return model;
}

}  // namespace mini_llama
