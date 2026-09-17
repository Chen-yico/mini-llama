// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class MatmulMode {
  kNaive,
  kThreaded,
  kSimd,
  kThreadedSimd,
};

// Default mode used by the main inference path.
MatmulMode DefaultMatmulMode();

// F32 matrix multiplication [m,k] x [k,n] -> [m,n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear: x @ weight^T
// Supports x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result rank matches input rank.
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama
