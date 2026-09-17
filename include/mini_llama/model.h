// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// Hyperparameters needed to size KV Cache / Context and Forward.
struct ModelConfig {
  int vocab_size = 128;
  int dim = 32;
  int hidden_dim = 86;
  int n_layers = 2;
  int n_heads = 4;
  int n_kv_heads = 4;
  int head_dim = 8;
  int max_seq_len = 128;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  RopeType rope_type = RopeType::kNormal;
};

// Norm / bias stay F32. Linear weights keep their GGUF storage type so Q4/Q8
// models do not expand to F32 at load time.
struct LayerWeights {
  Tensor attention_norm;   // [dim]
  QuantizedTensor wq;      // [n_heads * head_dim, dim]
  QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]
  QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]
  Tensor bq;               // optional [n_heads * head_dim]
  Tensor bk;               // optional [n_kv_heads * head_dim]
  Tensor bv;               // optional [n_kv_heads * head_dim]
  QuantizedTensor wo;      // [dim, n_heads * head_dim]
  Tensor ffn_norm;         // [dim]
  QuantizedTensor w_gate;  // [hidden_dim, dim]
  QuantizedTensor w_up;    // [hidden_dim, dim]
  QuantizedTensor w_down;  // [dim, hidden_dim]
};

struct MiniLlamaModel {
  ModelConfig config;
  bool loaded = false;
  std::string load_error;
  Tensor token_embedding;            // [vocab_size, dim]
  std::vector<LayerWeights> layers;  // [n_layers]
  Tensor final_norm;                 // [dim]
  QuantizedTensor lm_head;           // [vocab_size, dim]
};

// Allocate F32 weights and fill them with deterministic values for tests / CLI.
MiniLlamaModel MakeCpuTestModel(ModelConfig config = {});

}  // namespace mini_llama
