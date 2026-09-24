// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/model.h"

#include <stdexcept>

#include "mini_llama/quant.h"
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

namespace {

size_t TensorBytes(const Tensor& t) {
  return t.data.size() * sizeof(float);
}

size_t QuantizedTensorBytes(const QuantizedTensor& qt) {
  switch (qt.type) {
    case QuantType::kF32:
      return qt.f32_data.size() * sizeof(float);
    case QuantType::kQ80:
      return qt.q8_0_data.size() * sizeof(BlockQ80);
    case QuantType::kQ40:
      return qt.q4_0_data.size() * sizeof(BlockQ40);
    case QuantType::kQ41:
      return qt.q4_1_data.size() * sizeof(BlockQ41);
  }
  return 0;
}

size_t QuantizedTensorBytesF32(const QuantizedTensor& qt) {
  return qt.num_elements() * sizeof(float);
}

Tensor QuantizedTensorToF32(const QuantizedTensor& qt) {
  switch (qt.type) {
    case QuantType::kF32:
      return ToTensor(qt);
    case QuantType::kQ80:
      return DequantizeFromQ80(qt.q8_0_data, qt.shape);
    case QuantType::kQ40:
      return DequantizeFromQ40(qt.q4_0_data, qt.shape);
    case QuantType::kQ41:
      return DequantizeFromQ41(qt.q4_1_data, qt.shape);
  }
  throw std::runtime_error("quantized_tensor_to_f32: unknown quant type");
}

void QuantizeQtToQ80(QuantizedTensor& qt) {
  if (qt.type == QuantType::kQ80) {
    return;
  }
  Tensor t = QuantizedTensorToF32(qt);
  qt.q8_0_data = QuantizeToQ80(t);
  qt.type = QuantType::kQ80;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q4_0_data.clear();
  qt.q4_0_data.shrink_to_fit();
  qt.q4_1_data.clear();
  qt.q4_1_data.shrink_to_fit();
}

void QuantizeQtToQ40(QuantizedTensor& qt) {
  if (qt.type == QuantType::kQ40) {
    return;
  }
  Tensor t = QuantizedTensorToF32(qt);
  qt.q4_0_data = QuantizeToQ40(t);
  qt.type = QuantType::kQ40;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q8_0_data.clear();
  qt.q8_0_data.shrink_to_fit();
  qt.q4_1_data.clear();
  qt.q4_1_data.shrink_to_fit();
}

}  // namespace

size_t ModelWeightBytes(const MiniLlamaModel& model) {
  size_t bytes = 0;
  bytes += TensorBytes(model.token_embedding);
  bytes += TensorBytes(model.final_norm);
  bytes += QuantizedTensorBytes(model.lm_head);
  for (const auto& lw : model.layers) {
    bytes += TensorBytes(lw.attention_norm);
    bytes += QuantizedTensorBytes(lw.wq);
    bytes += QuantizedTensorBytes(lw.wk);
    bytes += QuantizedTensorBytes(lw.wv);
    bytes += TensorBytes(lw.bq);
    bytes += TensorBytes(lw.bk);
    bytes += TensorBytes(lw.bv);
    bytes += QuantizedTensorBytes(lw.wo);
    bytes += TensorBytes(lw.ffn_norm);
    bytes += QuantizedTensorBytes(lw.w_gate);
    bytes += QuantizedTensorBytes(lw.w_up);
    bytes += QuantizedTensorBytes(lw.w_down);
  }
  return bytes;
}

size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
  size_t bytes = 0;
  bytes += TensorBytes(model.token_embedding);
  bytes += TensorBytes(model.final_norm);
  bytes += QuantizedTensorBytesF32(model.lm_head);
  for (const auto& lw : model.layers) {
    bytes += TensorBytes(lw.attention_norm);
    bytes += QuantizedTensorBytesF32(lw.wq);
    bytes += QuantizedTensorBytesF32(lw.wk);
    bytes += QuantizedTensorBytesF32(lw.wv);
    bytes += TensorBytes(lw.bq);
    bytes += TensorBytes(lw.bk);
    bytes += TensorBytes(lw.bv);
    bytes += QuantizedTensorBytesF32(lw.wo);
    bytes += TensorBytes(lw.ffn_norm);
    bytes += QuantizedTensorBytesF32(lw.w_gate);
    bytes += QuantizedTensorBytesF32(lw.w_up);
    bytes += QuantizedTensorBytesF32(lw.w_down);
  }
  return bytes;
}

void QuantizeModelToQ80(MiniLlamaModel& model) {
  QuantizeQtToQ80(model.lm_head);
  for (auto& lw : model.layers) {
    QuantizeQtToQ80(lw.wq);
    QuantizeQtToQ80(lw.wk);
    QuantizeQtToQ80(lw.wv);
    QuantizeQtToQ80(lw.wo);
    QuantizeQtToQ80(lw.w_gate);
    QuantizeQtToQ80(lw.w_up);
    QuantizeQtToQ80(lw.w_down);
  }
}

void QuantizeModelToQ40(MiniLlamaModel& model) {
  QuantizeQtToQ40(model.lm_head);
  for (auto& lw : model.layers) {
    QuantizeQtToQ40(lw.wq);
    QuantizeQtToQ40(lw.wk);
    QuantizeQtToQ40(lw.wv);
    QuantizeQtToQ40(lw.wo);
    QuantizeQtToQ40(lw.w_gate);
    QuantizeQtToQ40(lw.w_up);
    QuantizeQtToQ40(lw.w_down);
  }
}

}  // namespace mini_llama
