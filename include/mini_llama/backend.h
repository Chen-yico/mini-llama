// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_BACKEND_H_
#define INCLUDE_MINI_LLAMA_BACKEND_H_

#include <string>

namespace mini_llama {

enum class BackendKind {
  kCpu,
  kCuda,
};

struct BackendConfig {
  BackendKind kind = BackendKind::kCpu;
  int device_id = 0;
  bool device_id_set = false;
};

bool ParseBackendKind(const std::string& text, BackendKind& kind);
std::string BackendKindName(BackendKind kind);

bool CudaBackendBuilt();
bool CudaBackendAvailable(int device_id = 0);
std::string CudaDeviceSummary(int device_id = 0);

// Throws when the requested backend cannot run in this build or on this host.
void ValidateBackend(const BackendConfig& config);

// Explains which execution path is active.
std::string BackendExecutionNote(const BackendConfig& config);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_BACKEND_H_
