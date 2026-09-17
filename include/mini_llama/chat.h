// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

struct ChatMessage {
  std::string role;
  std::string content;
};

// Interactive session state, including the token prefix cache.
class ChatSession {
 public:
  std::vector<ChatMessage> messages;
  std::vector<int> token_history;
  RadixTree prefix_cache;
  SamplingParams sampling_params;

  int total_prompt_tokens = 0;
  int total_generated_tokens = 0;
  double total_time_ms = 0.0;

  ChatSession() = default;

  void Clear();
  void AddMessage(const std::string& role, const std::string& content);
  void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);
  void SetTokenHistory(const std::vector<int>& tokens);
  void AppendToken(int token);
  void RecordPrefix(const std::vector<int>& tokens);
  size_t LongestCachedPrefix(const std::vector<int>& tokens) const;

  // min(radix_hit, tokens still present in the linear KV / history).
  size_t ReusablePrefixLength(const std::vector<int>& context_tokens,
                              const std::vector<int>& prompt_tokens) const;
};

}  // namespace mini_llama
