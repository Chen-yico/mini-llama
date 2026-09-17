// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_llama/ops.h"

namespace mini_llama {

namespace {

constexpr float kGreedyTemperature = 1e-6f;

void ValidateLogits(const Tensor& logits) {
  if (logits.size() == 0) {
    throw std::runtime_error("sampler: cannot sample from empty logits");
  }
  for (float value : logits.data) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("sampler: logits must be finite");
    }
  }
}

void ValidateTemperature(float temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0f) {
    throw std::runtime_error(
        "sampler: temperature must be finite and non-negative");
  }
}

void ValidateTopK(int top_k) {
  if (top_k < 0) {
    throw std::runtime_error("sampler: top_k must be non-negative");
  }
}

int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates,
                         float temperature, std::mt19937& rng) {
  if (temperature < kGreedyTemperature) {
    int best = candidates[0].second;
    float best_val = candidates[0].first;
    for (size_t i = 1; i < candidates.size(); ++i) {
      if (candidates[i].first > best_val) {
        best = candidates[i].second;
        best_val = candidates[i].first;
      }
    }
    return best;
  }

  float max_val = candidates[0].first;
  for (size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i].first > max_val) {
      max_val = candidates[i].first;
    }
  }

  std::vector<float> probs(candidates.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < candidates.size(); ++i) {
    probs[i] = std::exp((candidates[i].first - max_val) / temperature);
    sum += probs[i];
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);
  float cumulative = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i] / sum;
    if (r <= cumulative) {
      return candidates[i].second;
    }
  }
  return candidates.back().second;
}

}  // namespace

MiniSampler::MiniSampler(unsigned int seed) {
  if (seed != 0) {
    rng_.seed(seed);
  } else {
    std::random_device rd;
    rng_.seed(rd());
  }
}

MiniSampler::MiniSampler(const SamplingParams& params)
    : MiniSampler(params.seed) {}

int MiniSampler::SampleGreedy(const Tensor& logits) {
  ValidateLogits(logits);
  return ArgMax(logits);
}

int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
  ValidateLogits(logits);
  ValidateTemperature(temperature);
  if (temperature < kGreedyTemperature) {
    return SampleGreedy(logits);
  }

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.data.size());
  for (size_t i = 0; i < logits.data.size(); ++i) {
    candidates.emplace_back(logits.data[i], static_cast<int>(i));
  }
  return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::SampleTopK(const Tensor& logits, float temperature,
                            int top_k) {
  ValidateLogits(logits);
  ValidateTemperature(temperature);
  ValidateTopK(top_k);
  if (top_k == 0) {
    throw std::runtime_error(
        "sampler: top_k must be positive when calling SampleTopK");
  }
  if (top_k == 1) {
    return SampleGreedy(logits);
  }
  if (top_k > static_cast<int>(logits.data.size())) {
    top_k = static_cast<int>(logits.data.size());
  }

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.data.size());
  for (size_t i = 0; i < logits.data.size(); ++i) {
    candidates.emplace_back(logits.data[i], static_cast<int>(i));
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
              if (a.first == b.first) {
                return a.second < b.second;
              }
              return a.first > b.first;
            });
  candidates.resize(static_cast<size_t>(top_k));
  return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::Sample(const Tensor& logits, const SamplingParams& params) {
  ValidateTemperature(params.temperature);
  ValidateTopK(params.top_k);
  if (params.temperature < kGreedyTemperature || params.top_k == 1) {
    return SampleGreedy(logits);
  }
  if (params.top_k > 1) {
    return SampleTopK(logits, params.temperature, params.top_k);
  }
  return SampleTemperature(logits, params.temperature);
}

int SampleGreedy(const Tensor& logits) {
  return MiniSampler::SampleGreedy(logits);
}

}  // namespace mini_llama
