// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/loader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Simple JSON parser helper for model config and tensor metadata
// Only supports the specific format we need

namespace mini_llama {

static MiniLlamaModel LoadFailure(const std::string& message) {
  MiniLlamaModel model;
  model.loaded = false;
  model.load_error = message;
  std::cerr << "Failed to load model: " << message << std::endl;
  return model;
}

static size_t FindConfigKey(const std::string& content,
                            const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing config key: " + key);
  }
  size_t colon_pos = content.find(":", key_pos + marker.size());
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("missing ':' after config key: " + key);
  }
  size_t value_pos = colon_pos + 1;
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }
  if (value_pos >= content.size()) {
    throw std::runtime_error("missing value for config key: " + key);
  }
  return value_pos;
}

static bool IsJsonValueTerminator(char c) {
  return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' ||
         c == ']';
}

static void ValidateNumberTerminator(const std::string& content,
                                     size_t value_pos, size_t parsed,
                                     const std::string& key) {
  size_t pos = value_pos + parsed;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  if (pos < content.size() && !IsJsonValueTerminator(content[pos])) {
    throw std::runtime_error("invalid numeric value for config key: " + key);
  }
}

static int ParseRequiredInt(const std::string& content,
                            const std::string& key) {
  size_t value_pos = FindConfigKey(content, key);
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for config key: " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid integer for config key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static std::string TrimWs(const std::string& text) {
  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = text.find_last_not_of(" \t\n\r");
  return text.substr(start, end - start + 1);
}

static int ParseStrictIntToken(const std::string& token,
                               const std::string& field) {
  std::string trimmed = TrimWs(token);
  if (trimmed.empty()) {
    throw std::runtime_error("missing integer value for " + field);
  }

  char* end = nullptr;
  errno = 0;
  int64_t value = std::strtoll(trimmed.c_str(), &end, 10);
  if (end == trimmed.c_str() || errno == ERANGE ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid integer value for " + field + ": " +
                             trimmed);
  }
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  if (*end != '\0') {
    throw std::runtime_error("invalid integer value for " + field + ": " +
                             trimmed);
  }
  return static_cast<int>(value);
}

static size_t ParseStrictSizeValue(const std::string& content, size_t value_pos,
                                   const std::string& field) {
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }
  if (value_pos >= content.size() || content[value_pos] == '-') {
    throw std::runtime_error("invalid non-negative integer for " + field);
  }

  size_t parsed = 0;
  uint64_t value = 0;
  try {
    value = std::stoull(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid non-negative integer for " + field);
  }
  if (parsed == 0 || value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("invalid non-negative integer for " + field);
  }
  ValidateNumberTerminator(content, value_pos, parsed, field);
  return static_cast<size_t>(value);
}

static float ParseRequiredFloat(const std::string& content,
                                const std::string& key) {
  size_t value_pos = FindConfigKey(content, key);
  size_t parsed = 0;
  float value = 0.0f;
  try {
    value = std::stof(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid float for config key: " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid float for config key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static size_t FindOptionalKey(const std::string& content,
                              const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    return std::string::npos;
  }
  size_t colon_pos = content.find(":", key_pos + marker.size());
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("missing ':' after key: " + key);
  }
  size_t value_pos = colon_pos + 1;
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }
  if (value_pos >= content.size()) {
    throw std::runtime_error("missing value for key: " + key);
  }
  return value_pos;
}

static std::string ParseOptionalStringField(const std::string& content,
                                            const std::string& key,
                                            const std::string& fallback) {
  size_t value_pos = FindOptionalKey(content, key);
  if (value_pos == std::string::npos) {
    return fallback;
  }
  if (content[value_pos] != '"') {
    throw std::runtime_error("expected string value for key: " + key);
  }
  size_t end = content.find('"', value_pos + 1);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated string value for key: " + key);
  }
  return content.substr(value_pos + 1, end - value_pos - 1);
}

static int ParseOptionalIntField(const std::string& content,
                                 const std::string& key, int fallback) {
  size_t value_pos = FindOptionalKey(content, key);
  if (value_pos == std::string::npos) {
    return fallback;
  }
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for key: " + key);
  }
  if (parsed == 0) {
    throw std::runtime_error("invalid integer for key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static std::string ParseObjectForKey(const std::string& content,
                                     const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    return "";
  }
  size_t obj_start = content.find("{", key_pos + marker.size());
  if (obj_start == std::string::npos) {
    throw std::runtime_error("missing object for key: " + key);
  }

  int depth = 0;
  for (size_t pos = obj_start; pos < content.size(); ++pos) {
    if (content[pos] == '{') {
      ++depth;
    } else if (content[pos] == '}') {
      --depth;
      if (depth == 0) {
        return content.substr(obj_start, pos - obj_start + 1);
      }
    }
  }
  throw std::runtime_error("unclosed object for key: " + key);
}

static TokenizerInfo ParseTokenizerJson(const std::string& content) {
  TokenizerInfo tokenizer;
  std::string obj = ParseObjectForKey(content, "tokenizer");
  if (obj.empty()) {
    return tokenizer;
  }

  tokenizer.type = ParseOptionalStringField(obj, "type", tokenizer.type);
  tokenizer.path = ParseOptionalStringField(obj, "path", tokenizer.path);
  tokenizer.bos_id = ParseOptionalIntField(obj, "bos_id", tokenizer.bos_id);
  tokenizer.eos_id = ParseOptionalIntField(obj, "eos_id", tokenizer.eos_id);
  tokenizer.unk_id = ParseOptionalIntField(obj, "unk_id", tokenizer.unk_id);

  if (tokenizer.type != "ascii" && tokenizer.type != "json_vocab") {
    throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
  }
  if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
    throw std::runtime_error("tokenizer special ids must be non-negative");
  }
  if (tokenizer.bos_id == tokenizer.eos_id ||
      tokenizer.bos_id == tokenizer.unk_id ||
      tokenizer.eos_id == tokenizer.unk_id) {
    throw std::runtime_error("tokenizer special ids must be distinct");
  }
  if (tokenizer.type == "json_vocab" && tokenizer.path.empty()) {
    throw std::runtime_error("json_vocab tokenizer requires a path");
  }

  return tokenizer;
}

static void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0) {
    throw std::runtime_error("all model dimensions must be positive");
  }
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("dim must be divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("n_heads must be divisible by n_kv_heads");
  }
  if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be a positive even number");
  }
  if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
    throw std::runtime_error("rope_theta must be positive");
  }
  if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
    throw std::runtime_error("rms_norm_eps must be positive");
  }
}

static size_t CheckedTensorByteSize(const TensorInfo& info) {
  size_t num_elements = 1;
  for (int dim : info.shape) {
    if (dim <= 0) {
      throw std::runtime_error("invalid tensor shape for " + info.name);
    }
    const size_t d = static_cast<size_t>(dim);
    if (num_elements > std::numeric_limits<size_t>::max() / d) {
      throw std::runtime_error("tensor size overflow for " + info.name);
    }
    num_elements *= d;
  }
  if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("tensor byte size overflow for " + info.name);
  }
  return num_elements * sizeof(float);
}

static std::vector<std::pair<std::string, std::vector<int>>>
ExpectedTensorShapes(const ModelConfig& config) {
  std::vector<std::pair<std::string, std::vector<int>>> expected = {
      {"token_embedding", {config.vocab_size, config.dim}},
      {"final_norm", {config.dim}},
      {"lm_head", {config.vocab_size, config.dim}},
  };

  for (int layer = 0; layer < config.n_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    expected.push_back({prefix + "attention_norm", {config.dim}});
    expected.push_back(
        {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
    expected.push_back({prefix + "ffn_norm", {config.dim}});
    expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
  }

  return expected;
}

static std::string ShapeToString(const std::vector<int>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

static ModelConfig ParseConfigJson(const std::string& content) {
  const std::string obj = ParseObjectForKey(content, "config");
  if (obj.empty()) {
    throw std::runtime_error("missing 'config' object in model.json");
  }

  ModelConfig config;
  config.vocab_size = ParseRequiredInt(obj, "vocab_size");
  config.dim = ParseRequiredInt(obj, "dim");
  config.hidden_dim = ParseRequiredInt(obj, "hidden_dim");
  config.n_layers = ParseRequiredInt(obj, "n_layers");
  config.n_heads = ParseRequiredInt(obj, "n_heads");
  config.n_kv_heads = ParseRequiredInt(obj, "n_kv_heads");
  config.max_seq_len = ParseRequiredInt(obj, "max_seq_len");
  config.rope_theta = ParseRequiredFloat(obj, "rope_theta");
  config.rms_norm_eps = ParseRequiredFloat(obj, "rms_norm_eps");
  config.head_dim = config.dim / config.n_heads;
  config.rope_type = RopeType::kNormal;

  ValidateConfig(config);
  return config;
}

// ---------------------------------------------------------------------------
// Parse tensor metadata array from JSON content
// ---------------------------------------------------------------------------
static std::vector<TensorInfo> ParseTensorMetadata(const std::string& content) {
  std::vector<TensorInfo> result;

  const std::string tensors_marker = "\"tensors\"";
  size_t tensors_pos = content.find(tensors_marker);
  if (tensors_pos == std::string::npos) {
    throw std::runtime_error("missing 'tensors' field in model.json");
  }

  size_t bracket_open = content.find("[", tensors_pos);
  if (bracket_open == std::string::npos) {
    throw std::runtime_error("missing '[' after 'tensors'");
  }

  size_t i = bracket_open + 1;
  int depth = 1;
  while (i < content.size() && depth > 0) {
    // Find next object start
    while (i < content.size() && content[i] != '{') {
      if (content[i] == '[') {
        ++depth;
      } else if (content[i] == ']') {
        --depth;
      }
      ++i;
    }
    if (depth <= 0 || i >= content.size()) {
      break;
    }

    // Parse one tensor object
    TensorInfo info;
    size_t obj_start = i;
    size_t obj_end = content.find("}", obj_start);
    if (obj_end == std::string::npos) {
      throw std::runtime_error("unclosed tensor object");
    }
    std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

    bool has_name = false;
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offset = false;
    bool has_byte_size = false;

    if (FindOptionalKey(obj, "name") != std::string::npos) {
      info.name = ParseOptionalStringField(obj, "name", "");
      has_name = true;
    }

    if (FindOptionalKey(obj, "dtype") != std::string::npos) {
      info.dtype = ParseOptionalStringField(obj, "dtype", "");
      has_dtype = true;
    }

    // Parse shape array
    size_t shape_pos = obj.find("\"shape\"");
    if (shape_pos != std::string::npos) {
      size_t shape_bracket = obj.find("[", shape_pos);
      size_t shape_end = obj.find("]", shape_bracket);
      if (shape_bracket != std::string::npos &&
          shape_end != std::string::npos) {
        std::string shape_string_short =
            obj.substr(shape_bracket + 1, shape_end - shape_bracket - 1);
        std::stringstream ss(shape_string_short);
        std::string token;
        while (std::getline(ss, token, ',')) {
          if (TrimWs(token).empty()) {
            continue;
          }
          info.shape.push_back(
              ParseStrictIntToken(token, info.name + " shape"));
        }
        has_shape = true;
      }
    }

    // Parse offset
    size_t offset_pos = obj.find("\"offset\"");
    if (offset_pos != std::string::npos) {
      size_t offset_colon = obj.find(":", offset_pos);
      if (offset_colon == std::string::npos) {
        throw std::runtime_error("tensor offset field is malformed");
      }
      info.offset = ParseStrictSizeValue(obj, offset_colon + 1,
                                         info.name + " offset");
      has_offset = true;
    }

    // Parse byte_size
    size_t bs_pos = obj.find("\"byte_size\"");
    if (bs_pos != std::string::npos) {
      size_t bs_colon = obj.find(":", bs_pos);
      if (bs_colon == std::string::npos) {
        throw std::runtime_error("tensor byte_size field is malformed");
      }
      info.byte_size = ParseStrictSizeValue(obj, bs_colon + 1,
                                           info.name + " byte_size");
      has_byte_size = true;
    }

    if (!has_name || info.name.empty()) {
      throw std::runtime_error("tensor missing name field");
    }
    if (!has_dtype) {
      throw std::runtime_error("tensor missing dtype field: " + info.name);
    }
    if (!has_shape) {
      throw std::runtime_error("tensor missing shape field: " + info.name);
    }
    if (!has_offset) {
      throw std::runtime_error("tensor missing offset field: " + info.name);
    }
    if (!has_byte_size) {
      throw std::runtime_error("tensor missing byte_size field: " + info.name);
    }

    result.push_back(info);
    i = obj_end + 1;
  }

  return result;
}

static void ValidateManifest(const ModelManifest& manifest) {
  if (manifest.tensors.empty()) {
    throw std::runtime_error("model.json contains no tensors");
  }

  std::map<std::string, TensorInfo> tensor_map;
  std::vector<TensorInfo> by_offset;
  by_offset.reserve(manifest.tensors.size());

  for (const auto& info : manifest.tensors) {
    if (info.dtype != "float32") {
      throw std::runtime_error("unsupported dtype '" + info.dtype +
                               "' for tensor: " + info.name);
    }
    if (info.shape.empty()) {
      throw std::runtime_error("tensor shape must not be empty: " + info.name);
    }
    size_t expected_bytes = CheckedTensorByteSize(info);
    if (expected_bytes != info.byte_size) {
      throw std::runtime_error("tensor byte_size mismatch for " + info.name +
                               ": expected " + std::to_string(expected_bytes) +
                               ", manifest says " +
                               std::to_string(info.byte_size));
    }
    if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
      throw std::runtime_error("tensor offset overflow for " + info.name);
    }
    auto inserted = tensor_map.emplace(info.name, info);
    if (!inserted.second) {
      throw std::runtime_error("duplicate tensor name: " + info.name);
    }
    by_offset.push_back(info);
  }

  std::sort(by_offset.begin(), by_offset.end(),
            [](const TensorInfo& a, const TensorInfo& b) {
              return a.offset < b.offset;
            });
  for (size_t i = 1; i < by_offset.size(); ++i) {
    size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;
    if (by_offset[i].offset < previous_end) {
      throw std::runtime_error(
          "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
          by_offset[i].name);
    }
  }

  for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
    auto it = tensor_map.find(expected.first);
    if (it == tensor_map.end()) {
      throw std::runtime_error("missing required tensor: " + expected.first);
    }
    if (it->second.shape != expected.second) {
      throw std::runtime_error("tensor shape mismatch for " + expected.first +
                               ": expected " + ShapeToString(expected.second) +
                               ", manifest says " +
                               ShapeToString(it->second.shape));
    }
  }
}

// ---------------------------------------------------------------------------
// ModelManifest
// ---------------------------------------------------------------------------
ModelManifest ParseManifest(const std::string& config_path) {
  ModelManifest manifest;

  std::ifstream cf(config_path);
  if (!cf.is_open()) {
    throw std::runtime_error("failed to open config: " + config_path);
  }
  std::stringstream buffer;
  buffer << cf.rdbuf();
  if (cf.bad()) {
    throw std::runtime_error("failed to read config: " + config_path);
  }

  std::string content = buffer.str();
  manifest.config = ParseConfigJson(content);
  manifest.tokenizer = ParseTokenizerJson(content);
  manifest.tensors = ParseTensorMetadata(content);
  ValidateManifest(manifest);

  return manifest;
}

// ---------------------------------------------------------------------------
// Load a single tensor from binary file using manifest metadata
// ---------------------------------------------------------------------------
static Tensor LoadTensorByName(
    std::ifstream& wf, const std::map<std::string, TensorInfo>& tensor_map,
    const std::string& name) {
  auto it = tensor_map.find(name);
  if (it == tensor_map.end()) {
    throw std::runtime_error("missing required tensor: " + name);
  }

  const TensorInfo& info = it->second;

  if (info.offset >
      static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error("tensor offset exceeds stream limit for " + name);
  }
  if (info.byte_size >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("tensor byte size exceeds stream limit for " +
                             name);
  }

  wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
  if (!wf.good()) {
    throw std::runtime_error("failed to seek to offset for tensor: " + name);
  }

  Tensor t(info.shape, 0.0f);
  wf.read(reinterpret_cast<char*>(t.data.data()),
          static_cast<std::streamsize>(info.byte_size));
  if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
    throw std::runtime_error("weights file ended while reading tensor: " +
                             name);
  }

  return t;
}

// ---------------------------------------------------------------------------
// LoadModel
// ---------------------------------------------------------------------------
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path) {
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& e) {
    return LoadFailure(e.what());
  }

  std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
            << " dim=" << manifest.config.dim
            << " layers=" << manifest.config.n_layers
            << " heads=" << manifest.config.n_heads
            << " kv_heads=" << manifest.config.n_kv_heads
            << " head_dim=" << manifest.config.head_dim
            << " hidden_dim=" << manifest.config.hidden_dim << std::endl;

  std::map<std::string, TensorInfo> tensor_map;
  for (const auto& info : manifest.tensors) {
    tensor_map[info.name] = info;
  }

  // Open weights binary
  std::ifstream wf(weights_path, std::ios::binary);
  if (!wf.is_open()) {
    return LoadFailure("failed to open weights: " + weights_path);
  }

  MiniLlamaModel model;
  model.config = manifest.config;

  try {
    model.token_embedding = LoadTensorByName(wf, tensor_map, "token_embedding");

    model.layers.resize(manifest.config.n_layers);
    for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
      LayerWeights& lw = model.layers[layer];
      const std::string prefix = "layers." + std::to_string(layer) + ".";
      lw.attention_norm =
          LoadTensorByName(wf, tensor_map, prefix + "attention_norm");
      lw.wq = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "wq"));
      lw.wk = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "wk"));
      lw.wv = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "wv"));
      lw.wo = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "wo"));
      lw.ffn_norm = LoadTensorByName(wf, tensor_map, prefix + "ffn_norm");
      lw.w_gate = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "w_gate"));
      lw.w_up = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "w_up"));
      lw.w_down = ToQuantizedTensor(
          LoadTensorByName(wf, tensor_map, prefix + "w_down"));
    }

    model.final_norm = LoadTensorByName(wf, tensor_map, "final_norm");
    model.lm_head =
        ToQuantizedTensor(LoadTensorByName(wf, tensor_map, "lm_head"));
  } catch (const std::exception& e) {
    return LoadFailure(e.what());
  }

  // Validate trailing bytes
  wf.seekg(0, std::ios::end);
  std::streamoff file_size = wf.tellg();
  if (file_size < 0) {
    return LoadFailure("failed to determine weights file size");
  }

  size_t expected_end = 0;
  for (const auto& info : manifest.tensors) {
    size_t tensor_end = info.offset + info.byte_size;
    if (tensor_end > expected_end) {
      expected_end = tensor_end;
    }
  }

  if (static_cast<size_t>(file_size) != expected_end) {
    if (static_cast<size_t>(file_size) > expected_end) {
      return LoadFailure("weights file has trailing bytes: file size=" +
                         std::to_string(file_size) +
                         ", expected=" + std::to_string(expected_end));
    } else {
      return LoadFailure(
          "weights file is too short: file size=" + std::to_string(file_size) +
          ", expected=" + std::to_string(expected_end));
    }
  }

  model.loaded = true;
  std::cout << "Model loaded successfully from " << weights_path << std::endl;
  return model;
}

// ---------------------------------------------------------------------------
// InspectModel
// ---------------------------------------------------------------------------
bool InspectModel(const std::string& config_path) {
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to inspect model: " << e.what() << std::endl;
    return false;
  }

  const ModelConfig& c = manifest.config;
  std::cout << "=== Model Config ===" << std::endl;
  std::cout << "  vocab_size:    " << c.vocab_size << std::endl;
  std::cout << "  dim:           " << c.dim << std::endl;
  std::cout << "  hidden_dim:    " << c.hidden_dim << std::endl;
  std::cout << "  n_layers:      " << c.n_layers << std::endl;
  std::cout << "  n_heads:       " << c.n_heads << std::endl;
  std::cout << "  n_kv_heads:    " << c.n_kv_heads << std::endl;
  std::cout << "  head_dim:      " << c.head_dim << std::endl;
  std::cout << "  max_seq_len:   " << c.max_seq_len << std::endl;
  std::cout << "  rope_theta:    " << c.rope_theta << std::endl;
  std::cout << "  rms_norm_eps:  " << c.rms_norm_eps << std::endl;

  std::cout << std::endl;
  std::cout << "=== Tensors (" << manifest.tensors.size()
            << ") ===" << std::endl;

  size_t total_bytes = 0;
  for (const auto& t : manifest.tensors) {
    std::cout << "  " << t.name << std::endl;
    std::cout << "    shape:     [";
    for (size_t i = 0; i < t.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << t.shape[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "    dtype:     " << t.dtype << std::endl;
    std::cout << "    offset:    " << t.offset << std::endl;
    std::cout << "    byte_size: " << t.byte_size << std::endl;
    total_bytes += t.byte_size;
  }

  std::cout << std::endl;
  std::cout << "=== Tokenizer ===" << std::endl;
  std::cout << "  type: " << manifest.tokenizer.type << std::endl;
  std::cout << "  path: "
            << (manifest.tokenizer.path.empty() ? "<none>"
                                                : manifest.tokenizer.path)
            << std::endl;
  std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
  std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
  std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;
  std::cout << std::endl;
  std::cout << "Total tensor bytes: " << total_bytes << std::endl;
  return true;
}

}  // namespace mini_llama
