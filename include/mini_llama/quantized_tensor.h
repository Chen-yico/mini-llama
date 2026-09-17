// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class QuantType {
  kF32,
  kQ80,
  kQ40,
  kQ41,
};

constexpr int kQ80BlockSize = 32;

struct BlockQ80 {
  uint16_t d;
  int8_t qs[32];
};

static_assert(sizeof(BlockQ80) == sizeof(uint16_t) + kQ80BlockSize,
              "wrong Q8_0 block size");

constexpr int kQ40BlockSize = 32;

struct BlockQ40 {
  uint16_t d;
  uint8_t qs[16];
};

static_assert(sizeof(BlockQ40) == sizeof(uint16_t) + kQ40BlockSize / 2,
              "wrong Q4_0 block size");

constexpr int kQ41BlockSize = 32;

struct BlockQ41 {
  uint16_t d;
  uint16_t m;
  uint8_t qs[16];
};

static_assert(sizeof(BlockQ41) == sizeof(uint16_t) * 2 + kQ41BlockSize / 2,
              "wrong Q4_1 block size");

struct QuantizedTensor {
  QuantType type = QuantType::kF32;
  std::vector<int> shape;
  std::vector<float> f32_data;
  std::vector<BlockQ80> q8_0_data;
  std::vector<BlockQ40> q4_0_data;
  std::vector<BlockQ41> q4_1_data;

  size_t num_elements() const {
    size_t n = 1;
    for (int d : shape) {
      n *= static_cast<size_t>(d);
    }
    return n;
  }

  std::string ShapeStringShort() const {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      s += std::to_string(shape[i]);
      if (i + 1 < shape.size()) {
        s += ", ";
      }
    }
    s += "]";
    return s;
  }

  void AssertShape(const std::vector<int>& expected, const char* caller) const {
    if (shape != expected) {
      QuantizedTensor expected_view;
      expected_view.shape = expected;
      throw std::runtime_error(std::string(caller) +
                               ": shape mismatch, expected " +
                               expected_view.ShapeStringShort() + ", got " +
                               ShapeStringShort());
    }
  }
};

Tensor ToTensor(const QuantizedTensor& q);
QuantizedTensor ToQuantizedTensor(const Tensor& t);

}  // namespace mini_llama
