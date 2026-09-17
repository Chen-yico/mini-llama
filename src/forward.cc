// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {
namespace {

void ValidateForwardInputs(const MiniLlamaContext& ctx,
                           const MiniLlamaModel& model, int token) {
  const ModelConfig& config = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }
  if (token < 0 || token >= config.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= config.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
  if (model.layers.size() != static_cast<size_t>(config.n_layers)) {
    throw std::runtime_error(
        "ForwardToken layer count does not match model config");
  }
  if (model.token_embedding.data.size() <
      static_cast<size_t>(config.vocab_size) *
          static_cast<size_t>(config.dim)) {
    throw std::runtime_error(
        "ForwardToken token embedding tensor is smaller than config");
  }
}

void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                  const char* caller) {
  tensor.AssertShape(expected, caller);
}

Tensor ForwardAdd(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("forward_add: shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    y.data[i] = a.data[i] + b.data[i];
  }
  return y;
}

// Embedding lookup: token_id -> x[dim]
Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  const int dim = model.config.dim;
  Tensor x({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) {
    x.data[i] = model.token_embedding.data[token_id * dim + i];
  }
  return x;
}

// Attention: write current K/V, then Q @ history K, softmax, weighted V.
Tensor AttentionForward(const Tensor& q, const Tensor& k, const Tensor& v,
                        int pos, int layer, KvCache& kv_cache, int n_heads,
                        int n_kv_heads, int head_dim) {
  kv_cache.Write(layer, pos, k, v);

  Tensor attn_out({n_heads, head_dim}, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ParallelFor(n_heads, [&](int begin, int end) {
    for (int h = begin; h < end; ++h) {
      const int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

      std::vector<float> scores_data(static_cast<size_t>(pos + 1), 0.0f);
      for (int t = 0; t <= pos; ++t) {
        const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += q.data[h * head_dim + d] * k_ptr[d];
        }
        scores_data[static_cast<size_t>(t)] = dot * scale;
      }

      Tensor scores({pos + 1}, 0.0f);
      scores.data = std::move(scores_data);
      const Tensor probs = Softmax(scores);

      for (int d = 0; d < head_dim; ++d) {
        float val = 0.0f;
        for (int t = 0; t <= pos; ++t) {
          const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
          val += probs.data[t] * v_ptr[d];
        }
        attn_out.data[h * head_dim + d] = val;
      }
    }
  });

  return attn_out;
}

Tensor AddOptionalBias(const Tensor& x, const Tensor& bias,
                       const char* caller) {
  if (bias.data.empty()) {
    return x;
  }
  if (x.shape != bias.shape) {
    throw std::runtime_error(std::string(caller) +
                             ": bias shape mismatch " + x.ShapeStringShort() +
                             " vs " + bias.ShapeStringShort());
  }
  Tensor y(x.shape, 0.0f);
  for (size_t i = 0; i < x.size(); ++i) {
    y.data[i] = x.data[i] + bias.data[i];
  }
  return y;
}

Tensor FfnForward(const Tensor& h, const LayerWeights& layer) {
  const Tensor gate = Linear(h, layer.w_gate);
  const Tensor up = Linear(h, layer.w_up);
  return Linear(SwiGlu(gate, up), layer.w_down);
}

Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const Tensor& x, int layer, const LayerWeights& weights,
                    const ModelConfig& config) {
  const int dim = config.dim;
  const int n_heads = config.n_heads;
  const int n_kv_heads = config.n_kv_heads;
  const int head_dim = config.head_dim;
  const int pos = ctx.pos;

  Tensor h = RmsNorm(x, weights.attention_norm, config.rms_norm_eps);
  Tensor q_flat =
      AddOptionalBias(Linear(h, weights.wq), weights.bq, "forward_layer q");
  Tensor k_flat =
      AddOptionalBias(Linear(h, weights.wk), weights.bk, "forward_layer k");
  Tensor v_flat =
      AddOptionalBias(Linear(h, weights.wv), weights.bv, "forward_layer v");

  Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
  Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
  Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");

  Rope(q, k, pos, config.rope_theta, config.rope_type);

  Tensor attn_out = AttentionForward(q, k, v, pos, layer, ctx.kv_cache, n_heads,
                                     n_kv_heads, head_dim);
  Tensor attn_out_flat = attn_out.ReshapeChecked({n_heads * head_dim},
                                                 "forward_layer attn_out");
  Tensor attn_proj = Linear(attn_out_flat, weights.wo);
  RequireShape(attn_proj, {dim}, "forward_layer attn_proj");

  Tensor x_attn = ForwardAdd(x, attn_proj);

  Tensor h2 = RmsNorm(x_attn, weights.ffn_norm, config.rms_norm_eps);
  Tensor ff = FfnForward(h2, weights);
  RequireShape(ff, {dim}, "forward_layer ffn");
  return ForwardAdd(x_attn, ff);
}

Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                     const ModelConfig& config) {
  const Tensor normed = RmsNorm(x, model.final_norm, config.rms_norm_eps);
  const Tensor logits_flat = Linear(normed, model.lm_head);
  return logits_flat.ReshapeChecked({config.vocab_size}, "compute_logits");
}

}  // namespace

int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  if (n_kv_heads <= 0 || n_heads < n_kv_heads || n_heads % n_kv_heads != 0) {
    throw std::runtime_error("MapQHeadToKvHead: invalid GQA head counts");
  }
  if (q_head < 0 || q_head >= n_heads) {
    throw std::out_of_range("MapQHeadToKvHead: q_head out of range");
  }
  return q_head / (n_heads / n_kv_heads);
}

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
  ValidateForwardInputs(ctx, model, token);

  const ModelConfig& config = model.config;
  Tensor x = EmbedToken(model, token);
  for (int layer = 0; layer < config.n_layers; ++layer) {
    x = ForwardLayer(ctx, model, x, layer, model.layers[layer], config);
  }
  return ComputeLogits(x, model, config);
}

}  // namespace mini_llama
