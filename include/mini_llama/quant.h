// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape);
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape);
Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape);

Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape);
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape);
Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape);

}  // namespace mini_llama
