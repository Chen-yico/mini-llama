#include "mini_llama/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/quant.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

Tensor Matmul(const Tensor& a, const Tensor& b) {
  return MatmulDispatch(a, b, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
  return LinearDispatch(x, weight, DefaultMatmulMode());
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
  if (x.num_dims() != 1 || weight.num_dims() != 1) {
    throw std::runtime_error("RmsNorm: expected 1D tensors");
  }
  int dim = x.shape[0];
  if (weight.shape[0] != dim) {
    throw std::runtime_error("RmsNorm: dimension mismatch");
  }

  float ss = 0.0f;
  for (int i = 0; i < dim; ++i) {
    ss += x.data[i] * x.data[i];
  }
  ss /= static_cast<float>(dim);
  float scale = 1.0f / std::sqrt(ss + eps);

  Tensor y({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) {
    y.data[i] = x.data[i] * scale * weight.data[i];
  }
  return y;
}

Tensor Softmax(const Tensor& x) {
  if (x.num_dims() != 1) {
    throw std::runtime_error("Softmax: expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error("Softmax: cannot compute Softmax of empty tensor");
  }
  int n = x.shape[0];
  float max_val = x.data[0];
  for (int i = 1; i < n; ++i) {
    if (x.data[i] > max_val) {
      max_val = x.data[i];
    }
  }

  float sum = 0.0f;
  Tensor y({n}, 0.0f);
  for (int i = 0; i < n; ++i) {
    y.data[i] = std::exp(x.data[i] - max_val);
    sum += y.data[i];
  }
  for (int i = 0; i < n; ++i) {
    y.data[i] /= sum;
  }
  return y;
}

Tensor Silu(const Tensor& x) {
  Tensor y(x.shape, 0.0f);
  for (size_t i = 0; i < x.data.size(); ++i) {
    float val = x.data[i];
    y.data[i] = val / (1.0f + std::exp(-val));
  }
  return y;
}

Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("ElementwiseMul: shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.data.size(); ++i) {
    y.data[i] = a.data[i] * b.data[i];
  }
  return y;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
  if (gate.shape != up.shape) {
    throw std::runtime_error("SwiGlu: shape mismatch " +
                             gate.ShapeStringShort() + " vs " +
                             up.ShapeStringShort());
  }
  Tensor y(gate.shape, 0.0f);
  for (size_t i = 0; i < gate.data.size(); ++i) {
    float val = gate.data[i];
    float sigmoid = 1.0f / (1.0f + std::exp(-val));
    y.data[i] = val * sigmoid * up.data[i];
  }
  return y;
}

static void ApplyRopeNormalToHeads(Tensor& x, int pos, float theta) {
  int n_heads = x.shape[0];
  int head_dim = x.shape[1];
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < head_dim; i += 2) {
      float freq = 1.0f / std::pow(theta, static_cast<float>(i) /
                                              static_cast<float>(head_dim));
      float cos_val = std::cos(pos * freq);
      float sin_val = std::sin(pos * freq);
      float x0 = x.data[h * head_dim + i];
      float x1 = x.data[h * head_dim + i + 1];
      x.data[h * head_dim + i] = x0 * cos_val - x1 * sin_val;
      x.data[h * head_dim + i + 1] = x0 * sin_val + x1 * cos_val;
    }
  }
}

static void ApplyRopeNeoXToHeads(Tensor& x, int pos, float theta) {
  int n_heads = x.shape[0];
  int head_dim = x.shape[1];
  int half_dim = head_dim / 2;
  for (int h = 0; h < n_heads; ++h) {
    int base = h * head_dim;
    for (int i = 0; i < half_dim; ++i) {
      float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) /
                                              static_cast<float>(head_dim));
      float cos_val = std::cos(pos * freq);
      float sin_val = std::sin(pos * freq);
      float x0 = x.data[base + i];
      float x1 = x.data[base + half_dim + i];
      x.data[base + i] = x0 * cos_val - x1 * sin_val;
      x.data[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
    }
  }
}

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
  if (q.num_dims() != 2 || k.num_dims() != 2) {
    throw std::runtime_error("Rope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("Rope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("Rope: theta must be finite and positive");
  }

  int head_dim = q.shape[1];
  int k_head_dim = k.shape[1];
  if (head_dim != k_head_dim) {
    throw std::runtime_error("Rope: q and k head_dim mismatch " +
                             q.ShapeStringShort() + " vs " +
                             k.ShapeStringShort());
  }
  if (head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("Rope: head_dim must be positive and even, got " +
                             std::to_string(head_dim));
  }

  if (rope_type == RopeType::kNeoX) {
    ApplyRopeNeoXToHeads(q, pos, theta);
    ApplyRopeNeoXToHeads(k, pos, theta);
  } else {
    ApplyRopeNormalToHeads(q, pos, theta);
    ApplyRopeNormalToHeads(k, pos, theta);
  }
}

Tensor Linear(const Tensor& x, const QuantizedTensor& weight) {
  switch (weight.type) {
    case QuantType::kF32:
      return Linear(x, ToTensor(weight));
    case QuantType::kQ80:
      return LinearQ80(x, weight.q8_0_data, weight.shape);
    case QuantType::kQ40:
      return LinearQ40(x, weight.q4_0_data, weight.shape);
    case QuantType::kQ41:
      return LinearQ41(x, weight.q4_1_data, weight.shape);
  }
  throw std::runtime_error("Linear: unknown quant type");
}

int ArgMax(const Tensor& x) {
  if (x.size() == 0) {
    throw std::runtime_error("ArgMax: cannot compute ArgMax of empty tensor");
  }
  int best = 0;
  float best_val = x.data[0];
  for (size_t i = 1; i < x.data.size(); ++i) {
    if (x.data[i] > best_val) {
      best_val = x.data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

}  // namespace mini_llama
