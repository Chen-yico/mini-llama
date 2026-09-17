// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

namespace mini_llama {

// MiniBatch mirrors llama_batch: tokens with absolute positions.
struct MiniBatch {
  std::vector<int> tokens;
  std::vector<int> positions;

  int num_tokens() const { return static_cast<int>(tokens.size()); }

  static MiniBatch Single(int token, int pos);
  static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);
};

}  // namespace mini_llama
