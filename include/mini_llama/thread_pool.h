// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <thread>
#include <vector>

namespace mini_llama {

// Get the configured number of threads for parallel operations.
// Defaults to std::thread::hardware_concurrency().
int GetThreadCount();

// Set the number of threads for parallel operations.
// Pass 0 to Reset to hardware_concurrency().
void SetThreadCount(int n);

// Simple parallel for: split [0, n) into roughly equal chunks.
// Uses GetThreadCount() to decide how many threads to spawn.
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}  // namespace mini_llama
