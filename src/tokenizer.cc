// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/tokenizer.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.reserve(text.size() + 1);
  tokens.push_back(bos_id());
  for (unsigned char c : text) {
    const int id = static_cast<int>(c);
    tokens.push_back(id >= kVocabSize ? unk_id() : id);
  }
  return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
  if (token == bos_id()) {
    return "<bos>";
  }
  if (token == eos_id()) {
    return "<eos>";
  }
  if (token == unk_id()) {
    return "<unk>";
  }
  if (token < 0 || token >= kVocabSize) {
    return "<unk>";
  }
  if (token < 32 || token == 127) {
    return "";
  }
  return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string result;
  result.reserve(tokens.size());
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
  if (!vocab_path.empty()) {
    std::ifstream file(vocab_path);
    if (file.good()) {
      try {
        return std::make_unique<JsonVocabTokenizer>(vocab_path);
      } catch (const std::exception&) {
        // Fall back to ASCII when vocab parsing fails.
      }
    }
  }
  return std::make_unique<AsciiTokenizer>();
}

}  // namespace mini_llama
