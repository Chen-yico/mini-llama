// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
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

struct CudaUploadedWeight {
  std::string name;
  QuantType type = QuantType::kF32;
  bool linear = true;
  std::vector<int> shape;
  size_t block_count = 0;
  CudaDeviceBuffer buffer;
};

struct CudaModelWeights {
  int device_id = 0;
  size_t uploaded_weight_count = 0;
  size_t uploaded_bytes = 0;
  size_t linear_calls = 0;
  size_t activation_calls = 0;
  size_t attention_calls = 0;
  size_t attention_cpu_fallbacks = 0;
  size_t kv_cache_write_bytes = 0;
  size_t kv_cache_read_bytes = 0;
  size_t host_to_device_copies = 0;
  size_t device_to_host_copies = 0;
  size_t host_to_device_bytes = 0;
  size_t device_to_host_bytes = 0;
  std::vector<CudaUploadedWeight> weights;
};

struct MiniLlamaModel {
  ModelConfig config;
  bool loaded = false;
  std::string load_error;
  Tensor token_embedding;            // [vocab_size, dim]
  std::vector<LayerWeights> layers;  // [n_layers]
  Tensor final_norm;                 // [dim]
  QuantizedTensor lm_head;           // [vocab_size, dim]
  std::shared_ptr<CudaModelWeights> cuda_weights;
};

// Allocate F32 weights and fill them with deterministic values for tests / CLI.
MiniLlamaModel MakeCpuTestModel(ModelConfig config = {});

// Bytes consumed by model weights in their current storage format.
size_t ModelWeightBytes(const MiniLlamaModel& model);

// Bytes the same weights would consume if stored as F32.
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

// Upload CUDA-supported resident weights and keep the CPU copies intact.
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id = 0);
void ClearModelCudaWeights(MiniLlamaModel& model);
bool ModelHasCudaWeights(const MiniLlamaModel& model);
size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model);
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model);
void ResetModelCudaRuntimeStats(MiniLlamaModel& model);
size_t ModelCudaLinearCalls(const MiniLlamaModel& model);
size_t ModelCudaActivationCalls(const MiniLlamaModel& model);
size_t ModelCudaAttentionCalls(const MiniLlamaModel& model);
size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model);
size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model);
size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model);
size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model);
size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model);
size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model);
size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model);

// Convert Linear weight QuantizedTensors from F32 (or other) to Q8_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void QuantizeModelToQ80(MiniLlamaModel& model);

// Convert Linear weight QuantizedTensors from F32 (or other) to Q4_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void QuantizeModelToQ40(MiniLlamaModel& model);

}  // namespace mini_llama
