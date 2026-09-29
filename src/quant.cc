// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

namespace mini_llama {
namespace {

uint16_t FloatToFp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    uint32_t shifted = mantissa >> (1 - exponent);
    if ((shifted & 0x00001000u) != 0) {
      shifted += 0x00002000u;
    }
    return static_cast<uint16_t>(sign | (shifted >> 13));
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if ((mantissa & 0x00001000u) != 0) {
    mantissa += 0x00002000u;
    if ((mantissa & 0x00800000u) != 0) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

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

std::vector<BlockQ80> QuantizeToQ80(const Tensor& src) {
  if (src.size() == 0) {
    return {};
  }

  int row_size =
      src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
  int n_rows =
      src.num_dims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
  int row_blocks = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
  size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

  std::vector<BlockQ80> blocks;
  blocks.reserve(total_blocks);

  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQ80BlockSize;
      int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
      int block_len = k_end - base;

      float max_abs = 0.0f;
      for (int k = base; k < k_end; ++k) {
        float abs_val = std::abs(src.data[k]);
        if (abs_val > max_abs) {
          max_abs = abs_val;
        }
      }

      BlockQ80 block;
      std::memset(&block, 0, sizeof(block));
      if (max_abs > 0.0f) {
        float d = max_abs / 127.0f;
        block.d = FloatToFp16(d);
        float stored_d = Fp16ToFloat(block.d);
        float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
        for (int i = 0; i < block_len; ++i) {
          float q = src.data[base + i] * id;
          int qi = static_cast<int>(std::round(q));
          if (qi > 127) {
            qi = 127;
          } else if (qi < -127) {
            qi = -127;
          }
          block.qs[i] = static_cast<int8_t>(qi);
        }
      } else {
        block.d = 0;
      }
      for (int i = block_len; i < kQ80BlockSize; ++i) {
        block.qs[i] = 0;
      }

      blocks.push_back(block);
    }
  }

  return blocks;
}

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

std::vector<BlockQ40> QuantizeToQ40(const Tensor& src) {
  if (src.size() == 0) {
    return {};
  }

  int row_size =
      src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
  int n_rows =
      src.num_dims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
  int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
  size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

  std::vector<BlockQ40> blocks;
  blocks.reserve(total_blocks);

  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQ40BlockSize;
      int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);

      float max_abs = 0.0f;
      for (int k = base; k < k_end; ++k) {
        float abs_val = std::abs(src.data[k]);
        if (abs_val > max_abs) {
          max_abs = abs_val;
        }
      }

      BlockQ40 block;
      std::memset(&block, 0, sizeof(block));
      if (max_abs > 0.0f) {
        float d = max_abs / 7.0f;
        block.d = FloatToFp16(d);
        float stored_d = Fp16ToFloat(block.d);
        float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;

        for (int j = 0; j < kQ40BlockSize / 2; ++j) {
          float x0 = 0.0f;
          float x1 = 0.0f;
          int idx0 = base + j;
          int idx1 = base + j + kQ40BlockSize / 2;
          if (idx0 < k_end) {
            x0 = src.data[idx0] * id;
          }
          if (idx1 < k_end) {
            x1 = src.data[idx1] * id;
          }
          int qi0 = static_cast<int>(std::round(x0 + 8.0f));
          int qi1 = static_cast<int>(std::round(x1 + 8.0f));
          if (qi0 < 0) {
            qi0 = 0;
          }
          if (qi0 > 15) {
            qi0 = 15;
          }
          if (qi1 < 0) {
            qi1 = 0;
          }
          if (qi1 > 15) {
            qi1 = 15;
          }
          block.qs[j] = static_cast<uint8_t>(qi0 | (qi1 << 4));
        }
      } else {
        block.d = 0;
      }

      blocks.push_back(block);
    }
  }

  return blocks;
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

#ifdef MINI_LLAMA_USE_NEON
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error("LinearQ80: expected 2D weight shape");
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
        "LinearQ80: expected x shape [in_features] or [batch, in_features], "
        "got " +
        x.ShapeStringShort());
  }

  int out_features = weight_shape[0];
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(
        "LinearQ80: dimension mismatch x=" + x.ShapeStringShort() +
        " weight=" +
        QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
  }

  int n_blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
  size_t expected_blocks =
      static_cast<size_t>(out_features) * static_cast<size_t>(n_blocks_per_row);
  if (weight.size() != expected_blocks) {
    throw std::runtime_error("LinearQ80: block count mismatch: expected " +
                             std::to_string(expected_blocks) + ", got " +
                             std::to_string(weight.size()));
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
      int block_base = j * n_blocks_per_row;
      float32x4_t sum_vec = vdupq_n_f32(0.0f);
      float sum_scalar = 0.0f;

      for (int b = 0; b < n_blocks_per_row; ++b) {
        const BlockQ80& block = weight[static_cast<size_t>(block_base + b)];
        float d = Fp16ToFloat(block.d);
        int base_k = b * kQ80BlockSize;
        int k_end = std::min(base_k + kQ80BlockSize, in_features);
        int k = base_k;
        for (; k + 8 <= k_end; k += 8) {
          int8x8_t q8 = vld1_s8(block.qs + (k - base_k));
          int16x8_t q16 = vmovl_s8(q8);
          int32x4_t q32_lo = vmovl_s16(vget_low_s16(q16));
          int32x4_t q32_hi = vmovl_s16(vget_high_s16(q16));
          float32x4_t qf_lo = vmulq_n_f32(vcvtq_f32_s32(q32_lo), d);
          float32x4_t qf_hi = vmulq_n_f32(vcvtq_f32_s32(q32_hi), d);
          float32x4_t xf_lo = vld1q_f32(x_row + k);
          float32x4_t xf_hi = vld1q_f32(x_row + k + 4);
          sum_vec = vfmaq_f32(sum_vec, qf_lo, xf_lo);
          sum_vec = vfmaq_f32(sum_vec, qf_hi, xf_hi);
        }

        float32x2_t sum_lo = vget_low_f32(sum_vec);
        float32x2_t sum_hi = vget_high_f32(sum_vec);
        sum_lo = vadd_f32(sum_lo, sum_hi);
        float32x2_t sum_fp = vpadd_f32(sum_lo, sum_lo);
        sum_scalar += vget_lane_f32(sum_fp, 0);
        sum_vec = vdupq_n_f32(0.0f);

        for (; k < k_end; ++k) {
          float w = d * static_cast<float>(block.qs[k - base_k]);
          sum_scalar += w * x_row[k];
        }
      }

      result.data[static_cast<size_t>(row) * out_features +
                  static_cast<size_t>(j)] = sum_scalar;
    }
  });

  return result;
}
#else
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape,
                                                      DequantQ80);
}
#endif

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

Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
  Tensor weight_f32 = DequantizeFromQ80(weight, weight_shape);
  return Matmul(weight_f32, input);
}

float CompareMatmulError(const Tensor& weight, const Tensor& input) {
  Tensor f32_result = Matmul(weight, input);
  std::vector<BlockQ80> qweight = QuantizeToQ80(weight);
  Tensor q8_result = MatmulQ80(qweight, input, weight.shape);
  if (f32_result.shape != q8_result.shape) {
    throw std::runtime_error("CompareMatmulError: shape mismatch");
  }

  float max_err = 0.0f;
  for (size_t i = 0; i < f32_result.size(); ++i) {
    float err = std::abs(f32_result.data[i] - q8_result.data[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  return max_err;
}

}  // namespace mini_llama
