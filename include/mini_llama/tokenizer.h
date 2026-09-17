// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_llama {

// Tokenizer interface: text <-> token ids.
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // Encode text to token ids. AsciiTokenizer / JsonVocabTokenizer prepend BOS.
  // BpeTokenizer and GgufTokenizer do not; prompt logic owns BOS placement.
  virtual std::vector<int> Encode(const std::string& text) const = 0;

  virtual std::string DecodeToken(int token) const = 0;
  virtual std::string Decode(const std::vector<int>& tokens) const = 0;

  virtual int vocab_size() const = 0;
  virtual int bos_id() const = 0;
  virtual int eos_id() const = 0;
  virtual int unk_id() const = 0;
};

// Character-level ASCII tokenizer for unit tests (no external files).
class AsciiTokenizer : public ITokenizer {
 public:
  static constexpr int kVocabSize = 128;

  AsciiTokenizer() = default;

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return kVocabSize; }
  int bos_id() const override { return 1; }
  int eos_id() const override { return 2; }
  int unk_id() const override { return 0; }
};

// JSON vocab tokenizer for tiny models.
// Format: [{"id": 0, "content": "<unk>", "special": true}, ...]
class JsonVocabTokenizer : public ITokenizer {
 public:
  explicit JsonVocabTokenizer(const std::string& vocab_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return vocab_size_; }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  struct VocabEntry {
    int id = 0;
    std::string content;
    bool special = false;
  };

  int vocab_size_ = 0;
  int bos_id_ = 1;
  int eos_id_ = 2;
  int unk_id_ = 0;

  std::vector<VocabEntry> id_to_entry_;
  std::vector<std::pair<std::string, int>> content_to_id_;
};

// GPT-2 style BPE tokenizer (vocab.json + merges.txt + special_tokens.json).
// Encode: special tokens first (longest match), then byte-to-unicode, then
// rank-min merge. Decode reverses unicode back to raw bytes.
class BpeTokenizer : public ITokenizer {
 public:
  BpeTokenizer() = default;
  ~BpeTokenizer() override = default;

  bool Load(const std::string& vocab_path, const std::string& merges_path,
            const std::string& special_path);

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
  std::vector<std::pair<std::string, int>> special_tokens_;

  std::vector<std::string> b2u_;
  std::unordered_map<std::string, uint8_t> u2b_;

  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;

  void BuildByteMappings();
};

// Loads JsonVocabTokenizer when vocab_path exists; otherwise AsciiTokenizer.
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TOKENIZER_H_
