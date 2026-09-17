// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <random>

#include "mini_llama/tensor.h"

namespace mini_llama {

// Sampling parameters.
struct SamplingParams {
  float temperature = 0.0f;
  int top_k = 0;          // 0 means disabled
  unsigned int seed = 0;  // 0 means use random device
};

// MiniSampler supports greedy, temperature, and top-k sampling.
class MiniSampler {
 public:
  explicit MiniSampler(unsigned int seed = 0);
  explicit MiniSampler(const SamplingParams& params);

  // Sample a token from logits using the given parameters.
  int Sample(const Tensor& logits, const SamplingParams& params);

  // Greedy sampling (always returns ArgMax).
  static int SampleGreedy(const Tensor& logits);

  // Temperature sampling.
  int SampleTemperature(const Tensor& logits, float temperature);

  // Top-k sampling (temperature applied within top-k).
  int SampleTopK(const Tensor& logits, float temperature, int top_k);

 private:
  std::mt19937 rng_;
};

// Legacy free function for backward compatibility.
int SampleGreedy(const Tensor& logits);

}  // namespace mini_llama
