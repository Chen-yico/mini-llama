// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/quantized_tensor.h"

#include <stdexcept>

namespace mini_llama {

Tensor ToTensor(const QuantizedTensor& q) {
  if (q.type != QuantType::kF32) {
    throw std::runtime_error(
        "ToTensor: expected F32 QuantizedTensor, got quantized type");
  }
  Tensor t(q.shape, 0.0f);
  if (q.f32_data.size() == t.data.size()) {
    t.data = q.f32_data;
  }
  return t;
}

QuantizedTensor ToQuantizedTensor(const Tensor& t) {
  QuantizedTensor q;
  q.type = QuantType::kF32;
  q.shape = t.shape;
  q.f32_data = t.data;
  return q;
}

}  // namespace mini_llama
