// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// GGML types (subset)
// ---------------------------------------------------------------------------
enum GgmlType : uint32_t {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ40 = 2,
  kGgmlTypeQ41 = 3,
  kGgmlTypeQ80 = 8,
};

const char* GgmlTypeName(uint32_t type);

// ---------------------------------------------------------------------------
// GGUF value types
// ---------------------------------------------------------------------------
enum GgufValueType : uint32_t {
  kGgufTypeUint8 = 0,
  kGgufTypeInt8 = 1,
  kGgufTypeUint16 = 2,
  kGgufTypeInt16 = 3,
  kGgufTypeUint32 = 4,
  kGgufTypeInt32 = 5,
  kGgufTypeFloat32 = 6,
  kGgufTypeBool = 7,
  kGgufTypeString = 8,
  kGgufTypeArray = 9,
  kGgufTypeUint64 = 10,
  kGgufTypeInt64 = 11,
  kGgufTypeFloat64 = 12,
};

const char* GgufValueTypeName(uint32_t type);

// ---------------------------------------------------------------------------
// GGUF tensor info
// ---------------------------------------------------------------------------
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  uint32_t type = 0;    // GGML type
  uint64_t offset = 0;  // offset from start of data section
};

// ---------------------------------------------------------------------------
// GGUF metadata KV
// ---------------------------------------------------------------------------
struct GgufMetadataKv {
  std::string key;
  uint32_t type = 0;      // GGUF value type
  std::string value_str;  // human-readable representation
};

// ---------------------------------------------------------------------------
// Minimal GGUF reader (v3)
// ---------------------------------------------------------------------------
struct GgufReader {
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  std::vector<GgufMetadataKv> metadata;
  std::vector<GgufTensorInfo> tensors;
  uint64_t data_offset = 0;  // absolute file offset where tensor data begins

  // Load and parse a GGUF file. Returns false on error (message in load_error).
  bool Load(const std::string& path);

  std::string load_error;
};

// Print GGUF contents to stdout (for inspect-gguf).
void InspectGguf(const GgufReader& reader);

// ---------------------------------------------------------------------------
// Tensor data reading
// ---------------------------------------------------------------------------

// Read raw tensor bytes from the GGUF file data section.
// Returns empty vector on error.
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset);

// Compute the byte size of a tensor's data given its GGML type and shape.
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size);

// ---------------------------------------------------------------------------
// Typed metadata reading (re-opens file and seeks to the key)
// Returns true if found and type matches.
// ---------------------------------------------------------------------------
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out);
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out);

}  // namespace mini_llama
