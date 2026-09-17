// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

// Minimal c++ test runner header (declarations only)

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "tests/test_names.h"

// Test registry
struct TestCase {
  std::string name;
  std::function<bool()> fn;
};

std::vector<TestCase>& TestRegistry();
void RegisterTest(const std::string& name, std::function<bool()> fn);

// Assertion helpers (inlined to avoid duplicate symbols)
inline bool AssertTrue(bool cond, const char* expr, const char* file,
                       int line) {
  if (!cond) {
    std::cerr << "  ASSERTION FAILED: " << expr << " at " << file << ":" << line
              << std::endl;
    return false;
  }
  return true;
}

inline bool AssertEqFloat(float a, float b, float tol, const char* expr_a,
                          const char* expr_b, const char* file, int line) {
  if (std::fabs(a - b) > tol) {
    std::cerr << "  ASSERTION FAILED: |" << expr_a << " - " << expr_b << "| = |"
              << a << " - " << b << "| = " << std::fabs(a - b) << " > " << tol
              << " at " << file << ":" << line << std::endl;
    return false;
  }
  return true;
}

#define MINI_LLAMA_ASSERT_TRUE(cond)                      \
  do {                                                    \
    if (!AssertTrue((cond), #cond, __FILE__, __LINE__)) { \
      return false;                                       \
    }                                                     \
  } while (0)
#define MINI_LLAMA_ASSERT_EQ(a, b)                                   \
  do {                                                               \
    if (!AssertTrue((a) == (b), #a " == " #b, __FILE__, __LINE__)) { \
      return false;                                                  \
    }                                                                \
  } while (0)
#define MINI_LLAMA_ASSERT_NEAR(a, b, tol)                              \
  do {                                                                 \
    if (!AssertEqFloat((a), (b), (tol), #a, #b, __FILE__, __LINE__)) { \
      return false;                                                    \
    }                                                                  \
  } while (0)
#define MINI_LLAMA_ASSERT_FAIL(msg)                                         \
  do {                                                                      \
    std::cerr << "  ASSERTION FAILED: " << msg << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                     \
    return false;                                                           \
  } while (0)
