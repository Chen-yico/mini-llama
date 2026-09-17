// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
  size_t offset = 0;
  size_t byte_size = 0;
};

struct TokenizerInfo {
  std::string type = "ascii";
  std::string path;
  int bos_id = 1;
  int eos_id = 2;
  int unk_id = 0;
};

struct ModelManifest {
  ModelConfig config;
  TokenizerInfo tokenizer;
  std::vector<TensorInfo> tensors;
};

// Parse model.json into config, tokenizer, and tensor metadata.
// Throws std::runtime_error when the manifest is incomplete or inconsistent.
ModelManifest ParseManifest(const std::string& config_path);

// Load JSON+BIN weights by name. On failure, loaded=false and load_error is set.
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);

// Print a metadata snapshot from model.json without reading model.bin.
bool InspectModel(const std::string& config_path);

}  // namespace mini_llama
