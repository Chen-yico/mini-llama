// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/quant.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/thread_pool.h"

namespace mini_llama {
namespace {

float Fp16ToFloat(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      uint32_t exp32 = exponent + (127 - 15);
      bits = sign | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    uint32_t exp32 = exponent + (127 - 15);
    bits = sign | (exp32 << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  size_t total = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    int dim = shape[axis];
    if (dim <= 0) {
      throw std::runtime_error(std::string(caller) + ": dimension at axis " +
                               std::to_string(axis) +
                               " must be positive, got " + std::to_string(dim));
    }
    size_t dim_size = static_cast<size_t>(dim);
    if (total > std::numeric_limits<size_t>::max() / dim_size) {
      throw std::runtime_error(std::string(caller) +
                               ": shape element count overflow");
    }
    total *= dim_size;
  }
  return total;
}

}  // namespace

Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeFromQ80");

  int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
  int row_blocks = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
  size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "DequantizeFromQ80: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor dst(shape, 0.0f);
  size_t block_idx = 0;
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQ80BlockSize;
      int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
      const BlockQ80& block = blocks[block_idx++];
      float d = Fp16ToFloat(block.d);
      for (int k = base; k < k_end; ++k) {
        dst.data[k] = d * static_cast<float>(block.qs[k - base]);
      }
    }
  }
  return dst;
}

Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeFromQ40");

  int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
  int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
  size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "DequantizeFromQ40: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor dst(shape, 0.0f);
  size_t block_idx = 0;
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQ40BlockSize;
      int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
      const BlockQ40& block = blocks[block_idx++];
      float d = Fp16ToFloat(block.d);

      for (int j = 0; j < kQ40BlockSize / 2; ++j) {
        int idx0 = base + j;
        int idx1 = base + j + kQ40BlockSize / 2;
        int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
        int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
        if (idx0 < k_end) {
          dst.data[idx0] = d * static_cast<float>(q0);
        }
        if (idx1 < k_end) {
          dst.data[idx1] = d * static_cast<float>(q1);
        }
      }
    }
  }
  return dst;
}

Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape) {
  size_t total = CheckedNumel(shape, "DequantizeFromQ41");

  int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
  int row_blocks = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
  size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "DequantizeFromQ41: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor dst(shape, 0.0f);
  size_t block_idx = 0;
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQ41BlockSize;
      int k_end = std::min(base + kQ41BlockSize, row_offset + row_size);
      const BlockQ41& block = blocks[block_idx++];
      float d = Fp16ToFloat(block.d);
      float m = Fp16ToFloat(block.m);

      for (int j = 0; j < kQ41BlockSize / 2; ++j) {
        int idx0 = base + j;
        int idx1 = base + j + kQ41BlockSize / 2;
        int q0 = static_cast<int>(block.qs[j] & 0x0F);
        int q1 = static_cast<int>(block.qs[j] >> 4);
        if (idx0 < k_end) {
          dst.data[idx0] = d * static_cast<float>(q0) + m;
        }
        if (idx1 < k_end) {
          dst.data[idx1] = d * static_cast<float>(q1) + m;
        }
      }
    }
  }
  return dst;
}

namespace {

template <typename BlockType, int kBlockSize>
Tensor LinearQuantizedImpl(const Tensor& x,
                           const std::vector<BlockType>& blocks,
                           const std::vector<int>& weight_shape,
                           float (*dequant_fn)(const BlockType&, int)) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error("linear_quantized: expected 2D weight shape");
  }

  int in_features = 0;
  int rows = 1;
  bool is_1d = false;
  if (x.num_dims() == 1) {
    in_features = x.shape[0];
    is_1d = true;
  } else if (x.num_dims() == 2) {
    rows = x.shape[0];
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "linear_quantized: expected x shape [in_features] or [batch, "
        "in_features], got " +
        x.ShapeStringShort());
  }

  int out_features = weight_shape[0];
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(
        "linear_quantized: dimension mismatch x=" + x.ShapeStringShort() +
        " weight=" +
        QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
  }

  int n_blocks_per_row = (in_features + kBlockSize - 1) / kBlockSize;
  size_t expected_blocks =
      static_cast<size_t>(out_features) * static_cast<size_t>(n_blocks_per_row);
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "linear_quantized: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{rows, out_features},
                0.0f);

  ParallelFor(rows * out_features, [&](int begin, int end) {
    for (int flat_index = begin; flat_index < end; ++flat_index) {
      int row = flat_index / out_features;
      int j = flat_index % out_features;
      const float* x_row =
          x.data.data() + static_cast<size_t>(row) * in_features;
      float sum = 0.0f;
      int block_base = j * n_blocks_per_row;
      for (int b = 0; b < n_blocks_per_row; ++b) {
        const BlockType& block = blocks[static_cast<size_t>(block_base + b)];
        int base_k = b * kBlockSize;
        int k_end = std::min(base_k + kBlockSize, in_features);
        for (int k = base_k; k < k_end; ++k) {
          sum += dequant_fn(block, k - base_k) * x_row[k];
        }
      }
      result.data[static_cast<size_t>(row) * out_features +
                  static_cast<size_t>(j)] = sum;
    }
  });

  return result;
}

float DequantQ80(const BlockQ80& block, int idx) {
  return Fp16ToFloat(block.d) * static_cast<float>(block.qs[idx]);
}

float DequantQ40(const BlockQ40& block, int idx) {
  int q = idx < 16 ? static_cast<int>(block.qs[idx] & 0x0F)
                   : static_cast<int>(block.qs[idx - 16] >> 4);
  return Fp16ToFloat(block.d) * static_cast<float>(q - 8);
}

float DequantQ41(const BlockQ41& block, int idx) {
  int q = idx < 16 ? static_cast<int>(block.qs[idx] & 0x0F)
                   : static_cast<int>(block.qs[idx - 16] >> 4);
  return Fp16ToFloat(block.d) * static_cast<float>(q) + Fp16ToFloat(block.m);
}

}  // namespace

Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape,
                                                      DequantQ80);
}

Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ40, kQ40BlockSize>(x, weight, weight_shape,
                                                      DequantQ40);
}

Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ41, kQ41BlockSize>(x, weight, weight_shape,
                                                      DequantQ41);
}

}  // namespace mini_llama
