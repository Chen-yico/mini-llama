// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the optimized dispatch implementation.
// clang-format off
#include "mini_llama/matmul_dispatch.h"
// clang-format on

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/thread_pool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace mini_llama {

// ---------------------------------------------------------------------------
// Naive implementations (kept for readability and fallback)
// ---------------------------------------------------------------------------

static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int inner_dim = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < inner_dim; ++k) {
        sum += a.data[i * inner_dim + k] * b.data[k * n + j];
      }
      c.data[i * n + j] = sum;
    }
  }
  return c;
}

static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  for (int j = 0; j < out_features; ++j) {
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
      sum += x.data[k] * weight.data[j * in_features + k];
    }
    result.data[j] = sum;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Threaded implementations
// ---------------------------------------------------------------------------

static Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int inner_dim = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  ParallelFor(m, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < inner_dim; ++k) {
          sum += a.data[i * inner_dim + k] * b.data[k * n + j];
        }
        c.data[i * n + j] = sum;
      }
    }
  });
  return c;
}

static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  ParallelFor(out_features, [&](int begin, int end) {
    for (int j = begin; j < end; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < in_features; ++k) {
        sum += x.data[k] * weight.data[j * in_features + k];
      }
      result.data[j] = sum;
    }
  });
  return result;
}

// ---------------------------------------------------------------------------
// SIMD dot-product helpers (platform-specific)
// ---------------------------------------------------------------------------

#ifdef MINI_LLAMA_USE_NEON
static float DotSimdNeon(const float* a, const float* b, int n) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum = vfmaq_f32(sum, va, vb);
  }
  float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
  r = vpadd_f32(r, r);
  float result = vget_lane_f32(r, 0);
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}
#endif

#ifdef MINI_LLAMA_USE_AVX2
static float DotSimdAvx2(const float* a, const float* b, int n) {
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  // Horizontal reduce
  __m128 sum_lo = _mm256_castps256_ps128(sum);
  __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
  sum_lo = _mm_add_ps(sum_lo, sum_hi);
  sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
  sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
  float result = _mm_cvtss_f32(sum_lo);
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}
#endif

static float DotSimd(const float* a, const float* b, int n) {
#if defined(MINI_LLAMA_USE_AVX2)
  return DotSimdAvx2(a, b, n);
#elif defined(MINI_LLAMA_USE_NEON)
  return DotSimdNeon(a, b, n);
#else
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

// ---------------------------------------------------------------------------
// SIMD implementations
// ---------------------------------------------------------------------------

static Tensor LinearSimd(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  for (int j = 0; j < out_features; ++j) {
    result.data[j] = DotSimd(x.data.data(),
                             weight.data.data() + j * in_features, in_features);
  }
  return result;
}

static Tensor LinearThreadedSimd(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  ParallelFor(out_features, [&](int begin, int end) {
    for (int j = begin; j < end; ++j) {
      result.data[j] = DotSimd(
          x.data.data(), weight.data.data() + j * in_features, in_features);
    }
  });
  return result;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

MatmulMode DefaultMatmulMode() { return MatmulMode::kThreadedSimd; }

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  if (a.num_dims() != 2 || b.num_dims() != 2) {
    throw std::runtime_error("MatmulDispatch: expected 2D tensors");
  }
  if (a.shape[1] != b.shape[0]) {
    throw std::runtime_error("MatmulDispatch: dimension mismatch");
  }
  switch (mode) {
    case MatmulMode::kNaive:
      return MatmulNaive(a, b);
    case MatmulMode::kThreaded:
      return MatmulThreaded(a, b);
    case MatmulMode::kSimd:
    case MatmulMode::kThreadedSimd:
      // For general Matmul, b column access is not contiguous,
      // so SIMD is complex. Fall back to threaded for now.
      return MatmulThreaded(a, b);
  }
  return MatmulThreaded(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  if (weight.num_dims() != 2) {
    throw std::runtime_error(
        "LinearDispatch: expected weight shape [out_features, in_features]");
  }
  int in_features;
  if (x.num_dims() == 1) {
    in_features = x.shape[0];
  } else if (x.num_dims() == 2 && x.shape[0] == 1) {
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "LinearDispatch: expected x shape [in_features] or [1, in_features]");
  }
  if (weight.shape[1] != in_features) {
    throw std::runtime_error("LinearDispatch: dimension mismatch");
  }
  switch (mode) {
    case MatmulMode::kNaive:
      return LinearNaive(x, weight);
    case MatmulMode::kThreaded:
      return LinearThreaded(x, weight);
    case MatmulMode::kSimd:
      return LinearSimd(x, weight);
    case MatmulMode::kThreadedSimd:
      return LinearThreadedSimd(x, weight);
  }
  return LinearThreadedSimd(x, weight);
}

}  // namespace mini_llama
