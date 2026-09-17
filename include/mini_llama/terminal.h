// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

// Handles interactive terminal I/O for the chat mode.
class Terminal {
 public:
  Terminal() = default;

  void PrintUserPrompt() const;
  std::string ReadLine() const;
  void PrintAssistantPrefix() const;
  void PrintTokenText(const std::string& text) const;
  void Flush() const;
  void NewLine() const;
  void PrintHelp() const;
  void PrintStats(const ChatSession& session) const;
  void PrintParams(const SamplingParams& params) const;
  void PrintMessage(const std::string& msg) const;
};

}  // namespace mini_llama
