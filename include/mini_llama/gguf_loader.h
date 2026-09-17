// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "mini_llama/model.h"

namespace mini_llama {

// Linear / embedding / norm / bias: norms, embeddings and bias stay F32.
// Linear weights keep Q4/Q8 blocks in memory; Forward dequantizes per block.
MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}  // namespace mini_llama
