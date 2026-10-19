// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {
namespace {

static void RequireShape(const Tensor& t, const std::vector<int>& expected,
                         const char* caller) {
  t.AssertShape(expected, caller);
}

static Tensor AddOptionalBias(const Tensor& x, const Tensor& bias,
                              const char* caller) {
  if (bias.data.empty()) {
    return x;
  }
  if (x.num_dims() != 1 || bias.num_dims() != 1 || x.shape[0] != bias.shape[0]) {
    throw std::runtime_error(std::string(caller) +
                             ": bias shape mismatch x=" + x.ShapeStringShort() +
                             " bias=" + bias.ShapeStringShort());
  }
  Tensor y = x;
  for (int i = 0; i < x.shape[0]; ++i) {
    y.data[i] += bias.data[i];
  }
  return y;
}

static const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model,
                                                const std::string& name,
                                                QuantType type) {
  if (!model.cuda_weights) {
    return nullptr;
  }
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.name == name && weight.type == type) {
      return &weight;
    }
  }
  return nullptr;
}

static void RecordCudaLinearCopy(const MiniLlamaModel& model, const Tensor& x,
                                 const Tensor& y) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls += 1;
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->host_to_device_bytes += x.size() * sizeof(float);
  model.cuda_weights->device_to_host_bytes += y.size() * sizeof(float);
}

static void RecordCudaLinearCall(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls += 1;
}

static void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model,
                                       size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->host_to_device_bytes += bytes;
}

static void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model,
                                       size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->device_to_host_bytes += bytes;
}

static void RecordCudaActivationCall(const MiniLlamaModel& model,
                                     size_t calls = 1) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->activation_calls += calls;
}

static void RecordCudaAttentionCall(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->attention_calls += 1;
}

static void RecordCudaAttentionCpuFallback(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->attention_cpu_fallbacks += 1;
}

static void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->kv_cache_write_bytes += bytes;
}

static void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->kv_cache_read_bytes += bytes;
}

static int CudaDeviceId(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

static bool CudaLinearSupported(const QuantizedTensor& weight) {
  return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
         weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

static bool CudaLinearReady(const MiniLlamaModel& model,
                            const std::string& name,
                            const QuantizedTensor& weight) {
  return model.cuda_weights && CudaLinearSupported(weight) &&
         FindCudaWeight(model, name, weight.type) != nullptr;
}

static CudaTensor UploadCudaTensor(const MiniLlamaModel& model,
                                   const Tensor& x) {
  CudaTensor x_dev = CudaTensorFromHost(x, CudaDeviceId(model));
  RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
  return x_dev;
}

static Tensor DownloadCudaTensor(const MiniLlamaModel& model,
                                 const CudaTensor& x) {
  Tensor y = x.Download();
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
  return y;
}

static CudaTensor ForwardLinearDevice(const MiniLlamaModel& model,
                                      const std::string& name,
                                      const CudaTensor& x,
                                      const QuantizedTensor& weight) {
  if (!CudaLinearSupported(weight)) {
    throw std::runtime_error("CUDA forward Linear unsupported weight type: " +
                             name);
  }

  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, weight.type);
  if (cuda_weight == nullptr) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                             name);
  }

  CudaTensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceInput(x, cuda_weight->buffer.data(),
                                cuda_weight->shape, CudaDeviceId(model));
      break;
    case QuantType::kQ80:
      y = CudaQ80LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
  }
  RecordCudaLinearCall(model);
  return y;
}

static Tensor ForwardLinear(const MiniLlamaModel& model,
                            const std::string& name, const Tensor& x,
                            const QuantizedTensor& weight) {
  if (!model.cuda_weights) {
    return Linear(x, weight);
  }

  if (!CudaLinearSupported(weight)) {
    return Linear(x, weight);
  }

  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, weight.type);
  if (cuda_weight == nullptr) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                             name);
  }
  Tensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceWeight(x, cuda_weight->buffer.data(),
                                 cuda_weight->shape, nullptr,
                                 model.cuda_weights->device_id);
      break;
    case QuantType::kQ80:
      y = CudaQ80LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
  }
  RecordCudaLinearCopy(model, x, y);
  return y;
}

static void ForwardQkvProjection(const MiniLlamaModel& model,
                                 const std::string& layer_prefix,
                                 const Tensor& h, const LayerWeights& lw,
                                 Tensor& q_flat, Tensor& k_flat,
                                 Tensor& v_flat) {
  const std::string q_name = layer_prefix + "wq";
  const std::string k_name = layer_prefix + "wk";
  const std::string v_name = layer_prefix + "wv";
  if (CudaLinearReady(model, q_name, lw.wq) &&
      CudaLinearReady(model, k_name, lw.wk) &&
      CudaLinearReady(model, v_name, lw.wv)) {
    CudaTensor h_dev = UploadCudaTensor(model, h);
    CudaTensor q_dev = ForwardLinearDevice(model, q_name, h_dev, lw.wq);
    CudaTensor k_dev = ForwardLinearDevice(model, k_name, h_dev, lw.wk);
    CudaTensor v_dev = ForwardLinearDevice(model, v_name, h_dev, lw.wv);
    q_flat = AddOptionalBias(DownloadCudaTensor(model, q_dev), lw.bq,
                             "forward_layer q");
    k_flat = AddOptionalBias(DownloadCudaTensor(model, k_dev), lw.bk,
                             "forward_layer k");
    v_flat = AddOptionalBias(DownloadCudaTensor(model, v_dev), lw.bv,
                             "forward_layer v");
    return;
  }

  q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq,
                           "forward_layer q");
  k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, lw.wk), lw.bk,
                           "forward_layer k");
  v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, lw.wv), lw.bv,
                           "forward_layer v");
}

static CudaTensor AddOptionalBiasDevice(const MiniLlamaModel& model,
                                        const std::string& name, CudaTensor x,
                                        const Tensor& bias) {
  if (bias.data.empty()) {
    return x;
  }
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error("CUDA forward bias missing uploaded weight: " +
                             name);
  }
  if (cuda_weight->shape != bias.shape) {
    throw std::runtime_error(
        "CUDA forward bias uploaded weight shape mismatch: " + name);
  }
  CudaTensor y = CudaElementwiseAddDeviceWeight(
      x, cuda_weight->buffer.data(), cuda_weight->shape, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static void ForwardQkvProjectionDevice(const MiniLlamaModel& model,
                                       const std::string& layer_prefix,
                                       const CudaTensor& h,
                                       const LayerWeights& lw, CudaTensor& q,
                                       CudaTensor& k, CudaTensor& v) {
  q = ForwardLinearDevice(model, layer_prefix + "wq", h, lw.wq);
  k = ForwardLinearDevice(model, layer_prefix + "wk", h, lw.wk);
  v = ForwardLinearDevice(model, layer_prefix + "wv", h, lw.wv);
  q = AddOptionalBiasDevice(model, layer_prefix + "bq", std::move(q), lw.bq);
  k = AddOptionalBiasDevice(model, layer_prefix + "bk", std::move(k), lw.bk);
  v = AddOptionalBiasDevice(model, layer_prefix + "bv", std::move(v), lw.bv);
}

static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                             const Tensor& weight, float eps) {
  if (!model.cuda_weights) {
    return RmsNorm(x, weight, eps);
  }
  Tensor y = CudaRmsNorm(x, weight, eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static CudaTensor ForwardRmsNormDevice(const MiniLlamaModel& model,
                                       const std::string& name,
                                       const CudaTensor& x,
                                       const Tensor& weight, float eps) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error("CUDA forward RMSNorm missing uploaded weight: " +
                             name);
  }
  if (cuda_weight->shape != weight.shape) {
    throw std::runtime_error(
        "CUDA forward RMSNorm uploaded weight shape mismatch: " + name);
  }
  CudaTensor y =
      CudaRmsNormDeviceWeight(x, cuda_weight->buffer.data(), cuda_weight->shape,
                              eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                            const Tensor& up) {
  if (!model.cuda_weights) {
    return SwiGlu(gate, up);
  }
  Tensor silu_gate = CudaSilu(gate, CudaDeviceId(model));
  Tensor y = CudaElementwiseMul(silu_gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return y;
}

static CudaTensor ForwardAddDevice(const MiniLlamaModel& model,
                                   const CudaTensor& a, const CudaTensor& b) {
  CudaTensor y = CudaElementwiseAddDeviceInput(a, b, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
  if (!model.cuda_weights) {
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
  Tensor y = CudaElementwiseAdd(a, b, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
  if (!model.cuda_weights) {
    return Softmax(x);
  }
  Tensor y = CudaSoftmax(x, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k,
                        int pos, float theta, RopeType rope_type) {
  if (!model.cuda_weights) {
    Rope(q, k, pos, theta, rope_type);
    return;
  }
  CudaRope(q, k, pos, theta, rope_type, CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

static void ForwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q,
                              CudaTensor& k, int n_heads, int n_kv_heads,
                              int head_dim, int pos, float theta,
                              RopeType rope_type) {
  CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                      rope_type, CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

// ---------------------------------------------------------------------------
// Embedding lookup: token_id -> x[dim]
// ---------------------------------------------------------------------------
static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  int dim = model.config.dim;
  Tensor x({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) {
    x.data[i] = model.token_embedding.data[token_id * dim + i];
  }
  return x;
}

static CudaTensor EmbedTokenDevice(const MiniLlamaModel& model, int token_id) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, "token_embedding", QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error(
        "CUDA forward embedding missing uploaded weight: token_embedding");
  }
  if (cuda_weight->shape != model.token_embedding.shape) {
    throw std::runtime_error(
        "CUDA forward embedding uploaded weight shape mismatch");
  }
  return CudaEmbeddingLookupDeviceWeight(cuda_weight->buffer.data(),
                                         cuda_weight->shape, token_id,
                                         CudaDeviceId(model));
}

// ---------------------------------------------------------------------------
// GQA head mapping: given a query head, return the corresponding kv head
// ---------------------------------------------------------------------------
static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  return q_head / (n_heads / n_kv_heads);
}

// ---------------------------------------------------------------------------
// Attention forward for a single layer
//   q: [n_heads, head_dim]
//   k: [n_kv_heads, head_dim]
//   v: [n_kv_heads, head_dim]
//   pos: current position
//   layer: layer FlatIndex
//   kv_cache: global kv cache
//   n_heads, n_kv_heads, head_dim
// Returns attention output: [n_heads, head_dim]
// ---------------------------------------------------------------------------
static Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q,
                               const Tensor& k, const Tensor& v, int pos,
                               int layer, KvCache& kv_cache,
                               CudaKvCache* cuda_kv_cache, int n_heads,
                               int n_kv_heads, int head_dim) {
  // Write current k, v into cache
  kv_cache.Write(layer, pos, k, v);
  if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
    cuda_kv_cache->Write(layer, pos, k, v);
    RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
    Tensor attn_out =
        CudaAttentionDecode(q, *cuda_kv_cache, layer, pos, n_heads, n_kv_heads,
                            head_dim, CudaDeviceId(model));
    RecordCudaAttentionCall(model);
    RecordCudaKvCacheRead(
        model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                   static_cast<size_t>(head_dim) * sizeof(float) * 2);
    RecordCudaHostToDeviceCopy(model, q.size() * sizeof(float));
    RecordCudaDeviceToHostCopy(model, attn_out.size() * sizeof(float));
    return attn_out;
  }
  if (model.cuda_weights) {
    RecordCudaAttentionCpuFallback(model);
  }

  Tensor attn_out({n_heads, head_dim}, 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto compute_head = [&](int h) {
    int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

    std::vector<float> scores_data(pos + 1, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q.data[h * head_dim + d] * k_ptr[d];
      }
      scores_data[t] = dot * scale;
    }

    Tensor scores({pos + 1}, 0.0f);
    scores.data = std::move(scores_data);
    Tensor probs = ForwardSoftmax(model, scores);

    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
        val += probs.data[t] * v_ptr[d];
      }
      attn_out.data[h * head_dim + d] = val;
    }
  };

  if (model.cuda_weights) {
    for (int h = 0; h < n_heads; ++h) {
      compute_head(h);
    }
    return attn_out;
  }

  ParallelFor(n_heads, [&](int begin, int end) {
    for (int h = begin; h < end; ++h) {
      compute_head(h);
    }
  });

  return attn_out;
}

static CudaTensor AttentionForwardDevice(
    const MiniLlamaModel& model, const CudaTensor& q, const CudaTensor& k,
    const CudaTensor& v, int pos, int layer, CudaKvCache& cuda_kv_cache,
    int n_heads, int n_kv_heads, int head_dim) {
  cuda_kv_cache.WriteDevice(layer, pos, k, v);
  RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
  CudaTensor attn_out =
      CudaAttentionDecodeDeviceInput(q, cuda_kv_cache, layer, pos, n_heads,
                                     n_kv_heads, head_dim, CudaDeviceId(model));
  RecordCudaAttentionCall(model);
  RecordCudaKvCacheRead(
      model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                 static_cast<size_t>(head_dim) * sizeof(float) * 2);
  return attn_out;
}

// ---------------------------------------------------------------------------
// FFN forward (SwiGLU) for a single layer
//   h: [dim]  (already normalized)
//   lw: layer weights
// Returns FFN output: [dim]
// ---------------------------------------------------------------------------
static Tensor FfnForward(const MiniLlamaModel& model,
                         const std::string& layer_prefix, const Tensor& h,
                         const LayerWeights& lw) {
  const std::string gate_name = layer_prefix + "w_gate";
  const std::string up_name = layer_prefix + "w_up";
  const std::string down_name = layer_prefix + "w_down";
  if (CudaLinearReady(model, gate_name, lw.w_gate) &&
      CudaLinearReady(model, up_name, lw.w_up) &&
      CudaLinearReady(model, down_name, lw.w_down)) {
    CudaTensor h_dev = UploadCudaTensor(model, h);
    CudaTensor gate_dev =
        ForwardLinearDevice(model, gate_name, h_dev, lw.w_gate);
    CudaTensor up_dev = ForwardLinearDevice(model, up_name, h_dev, lw.w_up);
    CudaTensor silu_gate_dev =
        CudaSiluDeviceInput(gate_dev, CudaDeviceId(model));
    CudaTensor ff_dev = CudaElementwiseMulDeviceInput(silu_gate_dev, up_dev,
                                                      CudaDeviceId(model));
    RecordCudaActivationCall(model, 2);
    CudaTensor out_dev =
        ForwardLinearDevice(model, down_name, ff_dev, lw.w_down);
    return DownloadCudaTensor(model, out_dev);
  }

  Tensor gate = ForwardLinear(model, layer_prefix + "w_gate", h,
                              lw.w_gate);  // [hidden_dim]
  Tensor up =
      ForwardLinear(model, layer_prefix + "w_up", h, lw.w_up);  // [hidden_dim]

  Tensor ff =
      ForwardLinear(model, layer_prefix + "w_down",
                    ForwardSwiGlu(model, gate, up), lw.w_down);  // [dim]

  return ff;
}

static CudaTensor FfnForwardDevice(const MiniLlamaModel& model,
                                   const std::string& layer_prefix,
                                   const CudaTensor& h,
                                   const LayerWeights& lw) {
  CudaTensor gate =
      ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
  CudaTensor up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
  CudaTensor silu_gate = CudaSiluDeviceInput(gate, CudaDeviceId(model));
  CudaTensor ff =
      CudaElementwiseMulDeviceInput(silu_gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
}

// ---------------------------------------------------------------------------
// Single layer forward
//   x: input hidden state [dim]
//   layer: layer FlatIndex
//   ctx: inference context (for pos and kv_cache)
//   lw: layer weights
// Returns updated hidden state [dim]
// ---------------------------------------------------------------------------
static Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           const Tensor& x, int layer,
                           const LayerWeights& lw, const ModelConfig& c) {
  int dim = c.dim;
  int n_heads = c.n_heads;
  int n_kv_heads = c.n_kv_heads;
  int head_dim = c.head_dim;
  int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  // ---- Attention sublayer ----
  Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, c.rms_norm_eps);

  Tensor q_flat;
  Tensor k_flat;
  Tensor v_flat;
  ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);

  Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
  Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
  Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");

  // Apply RoPE to q and k
  ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);

  // Attention: compute + read/write KV cache
  Tensor attn_out =
      AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                       model.cuda_weights ? &ctx.cuda_kv_cache : nullptr,
                       n_heads, n_kv_heads, head_dim);

  // Project and residual
  Tensor attn_out_flat = attn_out.ReshapeChecked({1, n_heads * head_dim},
                                                 "forward_layer attn_out");
  Tensor attn_proj =
      ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);  // [dim]
  RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj");
  Tensor attn_proj_1d =
      attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");

  Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

  // ---- FFN sublayer ----
  Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
  Tensor ff = FfnForward(model, layer_prefix, h2, lw);
  RequireShape(ff, {dim}, "forward_layer ffn");

  return ForwardAdd(model, x_attn, ff);
}

static CudaTensor ForwardLayerDevice(MiniLlamaContext& ctx,
                                     const MiniLlamaModel& model,
                                     const CudaTensor& x, int layer,
                                     const LayerWeights& lw,
                                     const ModelConfig& c) {
  int dim = c.dim;
  int n_heads = c.n_heads;
  int n_kv_heads = c.n_kv_heads;
  int head_dim = c.head_dim;
  int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  CudaTensor h = ForwardRmsNormDevice(model, layer_prefix + "attention_norm", x,
                                      lw.attention_norm, c.rms_norm_eps);

  CudaTensor q;
  CudaTensor k;
  CudaTensor v;
  ForwardQkvProjectionDevice(model, layer_prefix, h, lw, q, k, v);
  ForwardRopeDevice(model, q, k, n_heads, n_kv_heads, head_dim, pos,
                    c.rope_theta, c.rope_type);

  CudaTensor attn_out =
      AttentionForwardDevice(model, q, k, v, pos, layer, ctx.cuda_kv_cache,
                             n_heads, n_kv_heads, head_dim);

  CudaTensor attn_proj =
      ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
  if (attn_proj.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error(
        "forward_layer_device: attention projection shape mismatch");
  }
  CudaTensor x_attn = ForwardAddDevice(model, x, attn_proj);

  CudaTensor h2 = ForwardRmsNormDevice(model, layer_prefix + "ffn_norm", x_attn,
                                       lw.ffn_norm, c.rms_norm_eps);
  CudaTensor ff = FfnForwardDevice(model, layer_prefix, h2, lw);
  if (ff.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
  }
  return ForwardAddDevice(model, x_attn, ff);
}

// ---------------------------------------------------------------------------
// Compute logits from final hidden state
//   x: [dim]
// Returns logits: [vocab_size]
// ---------------------------------------------------------------------------
static Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                            const ModelConfig& c) {
  Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
  Tensor logits_flat =
      ForwardLinear(model, "lm_head", normed, model.lm_head);  // [vocab_size]
  return logits_flat.ReshapeChecked({c.vocab_size}, "compute_logits");
}

static Tensor ComputeLogitsDevice(const CudaTensor& x,
                                  const MiniLlamaModel& model,
                                  const ModelConfig& c) {
  CudaTensor normed = ForwardRmsNormDevice(model, "final_norm", x,
                                           model.final_norm, c.rms_norm_eps);
  CudaTensor logits_dev =
      ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
  Tensor logits = DownloadCudaTensor(model, logits_dev);
  return logits.ReshapeChecked({c.vocab_size}, "compute_logits_device");
}

}  // namespace

Tensor ForwardTokenDevice(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                          int token) {
  const ModelConfig& c = model.config;
  if (!model.cuda_weights) {
    throw std::runtime_error("ForwardTokenDevice requires uploaded CUDA weights");
  }
  if (ctx.cuda_kv_cache.empty()) {
    ctx.cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim,
                            model.cuda_weights->device_id);
  }
  CudaTensor x_dev = EmbedTokenDevice(model, token);
  for (int layer = 0; layer < c.n_layers; ++layer) {
    x_dev = ForwardLayerDevice(ctx, model, x_dev, layer, model.layers[layer], c);
  }
  return ComputeLogitsDevice(x_dev, model, c);
}

}  // namespace mini_llama
