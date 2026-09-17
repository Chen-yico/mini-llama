// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mini_llama {

// User-selected ParallelFor override; zero uses the hardware concurrency
// default.
static int g_thread_count = 0;

int GetThreadCount() {
  if (g_thread_count > 0) {
    return g_thread_count;
  }
  unsigned int hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(hw) : 4;
}

void SetThreadCount(int n) { g_thread_count = n > 0 ? n : 0; }

void ParallelFor(int n, const std::function<void(int begin, int end)>& fn) {
  if (n <= 0) {
    return;
  }

  int n_threads = GetThreadCount();

  // Only parallelize if each thread gets a meaningful chunk of work.
  constexpr int kMinChunk = 16;
  if (n < n_threads * kMinChunk) {
    fn(0, n);
    return;
  }

  if (n_threads > n) {
    n_threads = n;
  }

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  std::exception_ptr first_exception = nullptr;
  std::mutex exception_mutex;

  int chunk = n / n_threads;
  int remainder = n % n_threads;

  int start = 0;
  for (int t = 0; t < n_threads; ++t) {
    int count = chunk + (t < remainder ? 1 : 0);
    int end = start + count;
    if (count > 0) {
      threads.emplace_back(
          [start, end, &fn, &first_exception, &exception_mutex]() {
            try {
              fn(start, end);
            } catch (...) {
              std::lock_guard<std::mutex> lock(exception_mutex);
              if (!first_exception) {
                first_exception = std::current_exception();
              }
            }
          });
    }
    start = end;
  }

  for (auto& t : threads) {
    t.join();
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

}  // namespace mini_llama
