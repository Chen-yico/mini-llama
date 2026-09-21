// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------
struct BenchmarkResult {
  int n_prompt_tokens = 0;
  int n_generated_tokens = 0;
  int n_decode_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;

  double tokens_per_sec() const {
    double total_ms = prefill_ms + decode_ms;
    if (total_ms <= 0.0) {
      return 0.0;
    }
    return n_generated_tokens * 1000.0 / total_ms;
  }

  double decode_tokens_per_sec() const {
    if (decode_ms <= 0.0 || n_decode_tokens <= 0) {
      return 0.0;
    }
    return n_decode_tokens * 1000.0 / decode_ms;
  }
};

// ---------------------------------------------------------------------------
// Debug dump helpers
// ---------------------------------------------------------------------------
void DumpTensorShape(const Tensor& t, const std::string& name);
void DumpLogitsTopK(const Tensor& logits, int k);
void DumpKvCacheInfo(const KvCache& cache, int current_pos = -1);

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------
// Runs prefill + Decode and returns timing stats.
// If verbose is true, prints debug dumps after each step.
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose);

}  // namespace mini_llama
