// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_names.h"

namespace {

constexpr float kAbsTol = 2e-3f;
constexpr float kRelTol = 2e-3f;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool CloseEnough(float actual, float expected) {
  float abs_err = std::abs(actual - expected);
  float scale = std::max(1.0f, std::abs(expected));
  return abs_err <= kAbsTol || abs_err / scale <= kRelTol;
}

void RequireCloseTensor(const Tensor& actual, const Tensor& expected,
                        const std::string& label) {
  Require(actual.shape == expected.shape, label + ": shape mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!CloseEnough(actual.data[i], expected.data[i])) {
      throw std::runtime_error(
          label + ": value mismatch at " + std::to_string(i) +
          ", actual=" + std::to_string(actual.data[i]) +
          ", expected=" + std::to_string(expected.data[i]));
    }
  }
}

void RequireCudaNotBuilt() {
  Require(!CudaRuntimeBuilt(), "CudaRuntimeBuilt should be false in CPU build");
  Require(!CudaOpsBuilt(), "CudaOpsBuilt should be false in CPU build");
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(model.loaded, "tiny model should load");
  Require(ModelCudaActivationCalls(model) == 0,
          "CPU model should report zero CUDA activation calls");
}

void TestCudaForwardFullTinyPath() {
  Require(CudaRuntimeBuilt(), "CudaRuntimeBuilt should be true in CUDA build");
  Require(CudaOpsBuilt(), "CudaOpsBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  MiniLlamaModel cpu_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(cpu_model.loaded, "CPU tiny model should load");
  Require(cuda_model.loaded, "CUDA tiny model should load");

  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext cuda_ctx(&cuda_model);
  MiniBatch batch = MiniBatch::FromTokens({1, 2, 3}, 0);

  Tensor cpu_logits = ForwardBatch(cpu_ctx, cpu_model, batch);

  UploadModelWeightsToCuda(cuda_model, 0);
  ResetModelCudaRuntimeStats(cuda_model);
  Tensor cuda_logits = ForwardBatch(cuda_ctx, cuda_model, batch);

  RequireCloseTensor(cuda_logits, cpu_logits, "cuda_forward logits");
  Require(ModelCudaLinearCalls(cuda_model) == 45,
          "three tiny tokens should run 45 CUDA Linear calls");
  Require(ModelCudaActivationCalls(cuda_model) == 45,
          "three tiny tokens should run 45 CUDA activation calls");
  Require(ModelCudaAttentionCalls(cuda_model) == 6,
          "three tiny tokens should run 6 CUDA attention calls");
  Require(!cuda_ctx.cuda_kv_cache.empty(),
          "CUDA forward should allocate GPU KV cache");
}

}  // namespace

int main() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaForwardFullTinyPath();
    std::cout << "PASS cuda_forward\n";
#else
    RequireCudaNotBuilt();
    std::cout << "PASS cuda_forward_cpu_build\n";
#endif
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL cuda_forward: " << e.what() << "\n";
    return 1;
  }
}
