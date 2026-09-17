// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

// Minimal dense tensor for F32 data
struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  Tensor() = default;
  explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

  // Total number of elements
  size_t size() const { return data.size(); }
  size_t num_elements() const { return data.size(); }

  // Number of dimensions
  int num_dims() const { return static_cast<int>(shape.size()); }

  // Compute flat FlatIndex from multi-dimensional indices
  size_t FlatIndex(const std::vector<int>& indices) const;

  // Get element at multi-dimensional indices
  float& At(const std::vector<int>& indices);
  float At(const std::vector<int>& indices) const;

  // Convenience 1D/2D/3D/4D accessors
  float& At1(int i);
  float At1(int i) const;
  float& At2(int i, int j);
  float At2(int i, int j) const;
  float& At3(int i, int j, int k);
  float At3(int i, int j, int k) const;
  float& At4(int i, int j, int k, int l);
  float At4(int i, int j, int k, int l) const;

  // Get a pointer to a row in a 2D tensor
  // t: [rows, cols], RowPtr(r) -> &data[r * cols]
  float* RowPtr(int row);
  const float* RowPtr(int row) const;

  // Assert shape matches expected; throws on mismatch
  void AssertShape(const std::vector<int>& expected, const char* caller) const;

  // Reshape after verifying element count stays unchanged
  Tensor ReshapeChecked(const std::vector<int>& new_shape,
                        const char* caller) const;

  // Helper for 1D access
  float& operator[](size_t i) { return data[i]; }
  float operator[](size_t i) const { return data[i]; }

  // Print shape and optionally data
  std::string ShapeStringShort() const;
  std::string ShapeString() const { return ShapeStringShort(); }
  void Print(const std::string& name = "", bool print_data = false) const;
};

// Create a 1D tensor
Tensor MakeTensor1D(int d0, float fill = 0.0f);

// Create a 2D tensor
Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);

// Create a 3D tensor
Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);

// Create a 4D tensor
Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);

}  // namespace mini_llama
