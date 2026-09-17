// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "mini_llama/chat.h"

namespace mini_llama {

// Builds a text prompt from a chat message history.
// PromptBuilder::Build() dispatches on chat_template_:
//   1. empty            -> BuildPlain (System:/User:/Assistant:)
//   2. Jinja2 ({% / {{) -> CompileTemplate + ExecuteTemplate(..., true)
//   3. "qwen2"          -> BuildQwen2 (ChatML, built-in fallback)
//   4. anything else    -> BuildPlain
//
// ExecuteTemplate's add_generation_prompt is always true so the rendered
// prompt keeps the trailing assistant prefix (e.g. <|im_start|>assistant\n).
class PromptBuilder {
 public:
  PromptBuilder() = default;

  // Set a chat template. If empty, uses plain text mode.
  void SetChatTemplate(const std::string& template_str);

  // Build prompt from messages.
  std::string Build(const std::vector<ChatMessage>& messages) const;

 private:
  std::string chat_template_;

  std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
  std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

// Load tokenizer.chat_template from a GGUF file.
// Returns the raw Jinja2 string when present; "qwen2" when the key is missing
// and general.architecture == "qwen2"; otherwise an empty string.
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);

}  // namespace mini_llama
