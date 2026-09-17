// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// GgufTokenizer: loads tokenizer data directly from GGUF metadata.
//
// Reads:
//   - tokenizer.ggml.tokens      (vocab pieces)
//   - tokenizer.ggml.merges      (BPE merge rules)
//   - tokenizer.ggml.token_type  (to identify special/control tokens)
//   - tokenizer.ggml.bos_token_id
//   - tokenizer.ggml.eos_token_id
//
// Implements GPT-2 style BPE with byte-level fallback.
// ---------------------------------------------------------------------------
class GgufTokenizer : public ITokenizer {
 public:
  GgufTokenizer() = default;
  ~GgufTokenizer() override = default;

  // Load tokenizer from a GGUF file. Returns false if the file lacks
  // the required tokenizer metadata.
  bool Load(const std::string& gguf_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override {
    return static_cast<int>(id_to_token_.size());
  }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;
  std::vector<std::pair<std::string, int>>
      special_tokens_;  // sorted by length desc

  // Byte <-> unicode mappings (GPT-2 style)
  std::vector<std::string> b2u_;
  std::unordered_map<std::string, uint8_t> u2b_;

  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;

  void BuildByteMappings();
};

// ---------------------------------------------------------------------------
// Factory: create a GgufTokenizer from a GGUF file.
// Returns nullptr if the GGUF lacks tokenizer metadata.
// ---------------------------------------------------------------------------
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
