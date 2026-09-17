#pragma once

#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

enum class RopeType {
  kNormal,
  kNeoX,
};

// Matrix multiplication: c = a * b
// a: [m, k], b: [k, n], c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// Linear projection: y = x * weight^T
// x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// Linear with quantized weight. F32 dispatches to the Tensor path; Q8/Q4
// dequantize one block at a time and never allocate a full F32 weight copy.
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over a 1D tensor
Tensor Softmax(const Tensor& x);

// SiLU activation: Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

// Element-wise multiplication
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU: SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE applied to Q and K
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// Argmax: return index of maximum value
int ArgMax(const Tensor& x);

}  // namespace mini_llama
