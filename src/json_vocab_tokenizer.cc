// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the JSON vocabulary tokenizer implementation.
// clang-format off
#include "mini_llama/tokenizer.h"
// clang-format on

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

// Simple JSON array parser for the vocab format.
// Only supports: [{"id": n, "content": "...", "special": true/false}, ...]

namespace {

static std::string Trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

static std::string ParseJsonString(const std::string& content, size_t& pos) {
  while (pos < content.size() && content[pos] != '"') {
    ++pos;
  }
  if (pos >= content.size() || content[pos] != '"') {
    throw std::runtime_error("expected opening quote for string");
  }
  ++pos;  // skip opening quote
  std::string result;
  while (pos < content.size() && content[pos] != '"') {
    if (content[pos] == '\\' && pos + 1 < content.size()) {
      char next = content[pos + 1];
      if (next == 'n') {
        result += '\n';
      } else if (next == 't') {
        result += '\t';
      } else if (next == 'r') {
        result += '\r';
      } else if (next == '\\') {
        result += '\\';
      } else if (next == '"') {
        result += '"';
      } else if (next == 'u' && pos + 5 < content.size()) {
        // Simple \uXXXX escape (4 hex digits)
        std::string hex = content.substr(pos + 2, 4);
        unsigned int codepoint = std::stoul(hex, nullptr, 16);
        if (codepoint <= 0x7F) {
          result += static_cast<char>(codepoint);
        } else {
          // For codepoints outside ASCII, just append as-is or use '?'
          result += '?';
        }
        pos += 6;
        continue;
      } else {
        result += next;
      }
      pos += 2;
    } else {
      result += content[pos];
      ++pos;
    }
  }
  if (pos >= content.size()) {
    throw std::runtime_error("unterminated string");
  }
  ++pos;  // skip closing quote
  return result;
}

static int ParseJsonInt(const std::string& content, size_t& pos) {
  while (pos < content.size() &&
         !std::isdigit(static_cast<unsigned char>(content[pos])) &&
         content[pos] != '-') {
    ++pos;
  }
  size_t start = pos;
  if (content[pos] == '-') {
    ++pos;
  }
  while (pos < content.size() &&
         std::isdigit(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  return std::stoi(content.substr(start, pos - start));
}

static bool ParseJsonBool(const std::string& content, size_t& pos) {
  if (content.substr(pos, 4) == "true") {
    pos += 4;
    return true;
  }
  if (content.substr(pos, 5) == "false") {
    pos += 5;
    return false;
  }
  throw std::runtime_error("expected true or false");
}

}  // namespace

JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f.is_open()) {
    throw std::runtime_error("failed to open vocab: " + vocab_path);
  }
  std::stringstream buffer;
  buffer << f.rdbuf();
  if (f.bad()) {
    throw std::runtime_error("failed to read vocab: " + vocab_path);
  }

  std::string content = buffer.str();

  // Find opening bracket
  size_t pos = content.find('[');
  if (pos == std::string::npos) {
    throw std::runtime_error("vocab JSON must be an array");
  }
  ++pos;

  int max_id = -1;

  while (pos < content.size()) {
    // Find next object start
    while (pos < content.size() && content[pos] != '{') {
      if (content[pos] == ']') {
        break;
      }
      ++pos;
    }
    if (pos >= content.size() || content[pos] != '{') {
      break;
    }
    ++pos;

    VocabEntry entry;
    bool has_id = false;
    bool has_content = false;

    // Parse object fields
    while (pos < content.size() && content[pos] != '}') {
      // Skip whitespace and commas
      while (pos < content.size() &&
             (std::isspace(static_cast<unsigned char>(content[pos])) ||
              content[pos] == ',')) {
        ++pos;
      }
      if (pos >= content.size() || content[pos] == '}') {
        break;
      }

      // Parse key
      std::string key = ParseJsonString(content, pos);

      // Skip to colon
      while (pos < content.size() && content[pos] != ':') {
        ++pos;
      }
      if (pos < content.size()) {
        ++pos;
      }
      // Skip whitespace
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }

      if (key == "id") {
        entry.id = ParseJsonInt(content, pos);
        has_id = true;
      } else if (key == "content") {
        entry.content = ParseJsonString(content, pos);
        has_content = true;
      } else if (key == "special") {
        entry.special = ParseJsonBool(content, pos);
      } else {
        // Skip unknown value
        if (pos < content.size() && content[pos] == '"') {
          ParseJsonString(content, pos);
        } else {
          while (pos < content.size() && content[pos] != ',' &&
                 content[pos] != '}') {
            ++pos;
          }
        }
      }
    }

    if (pos < content.size() && content[pos] == '}') {
      ++pos;
    }

    if (!has_id || !has_content) {
      throw std::runtime_error("vocab entry missing id or content");
    }

    if (entry.id > max_id) {
      max_id = entry.id;
    }
    if (entry.id < 0) {
      throw std::runtime_error("vocab entry id must be non-negative");
    }

    if (entry.content == "<bos>") {
      bos_id_ = entry.id;
    } else if (entry.content == "<eos>") {
      eos_id_ = entry.id;
    } else if (entry.content == "<unk>") {
      unk_id_ = entry.id;
    }

    if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
      id_to_entry_.resize(entry.id + 1);
    }
    id_to_entry_[entry.id] = entry;
    content_to_id_.push_back({entry.content, entry.id});
  }

  if (max_id < 0) {
    throw std::runtime_error("vocab file contains no entries");
  }

  vocab_size_ = max_id + 1;

  // Sort content_to_id_ by content length descending for greedy longest match
  std::sort(content_to_id_.begin(), content_to_id_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
}

std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.push_back(bos_id_);

  size_t pos = 0;
  while (pos < text.size()) {
    bool matched = false;
    for (const auto& pair : content_to_id_) {
      const std::string& content = pair.first;
      if (content.empty()) {
        continue;
      }
      if (text.compare(pos, content.size(), content) == 0) {
        tokens.push_back(pair.second);
        pos += content.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      // No vocab entry matches this substring; consume one byte as UNK
      tokens.push_back(unk_id_);
      ++pos;
    }
  }

  return tokens;
}

std::string JsonVocabTokenizer::DecodeToken(int token) const {
  if (token < 0 || static_cast<size_t>(token) >= id_to_entry_.size()) {
    return "<unk>";
  }
  return id_to_entry_[token].content;
}

std::string JsonVocabTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

}  // namespace mini_llama
