// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/gguf.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Type name helpers
// ---------------------------------------------------------------------------
const char* GgmlTypeName(uint32_t type) {
  switch (type) {
    case kGgmlTypeF32:
      return "F32";
    case kGgmlTypeF16:
      return "F16";
    case kGgmlTypeQ40:
      return "Q4_0";
    case kGgmlTypeQ41:
      return "Q4_1";
    case kGgmlTypeQ80:
      return "Q8_0";
    default:
      return "UNKNOWN";
  }
}

const char* GgufValueTypeName(uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
      return "uint8";
    case kGgufTypeInt8:
      return "int8";
    case kGgufTypeUint16:
      return "uint16";
    case kGgufTypeInt16:
      return "int16";
    case kGgufTypeUint32:
      return "uint32";
    case kGgufTypeInt32:
      return "int32";
    case kGgufTypeFloat32:
      return "float32";
    case kGgufTypeBool:
      return "bool";
    case kGgufTypeString:
      return "string";
    case kGgufTypeArray:
      return "array";
    case kGgufTypeUint64:
      return "uint64";
    case kGgufTypeInt64:
      return "int64";
    case kGgufTypeFloat64:
      return "float64";
    default:
      return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Internal little-endian readers
// ---------------------------------------------------------------------------
namespace {

constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "GGUF reader currently supports little-endian hosts only."
#endif

bool ReadBytes(std::ifstream& f, char* buf, std::size_t n) {
  if (n > static_cast<std::size_t>(
              std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  return static_cast<bool>(
      f.read(buf, static_cast<std::streamsize>(n)));
}

template <typename T>
bool ReadLe(std::ifstream& f, T& out) {
  char buf[sizeof(T)];
  if (!ReadBytes(f, buf, sizeof(T))) {
    return false;
  }
  std::memcpy(&out, buf, sizeof(T));
  return true;
}

bool ReadU32(std::ifstream& f, uint32_t& out) { return ReadLe(f, out); }

bool ReadU64(std::ifstream& f, uint64_t& out) { return ReadLe(f, out); }

bool ReadString(std::ifstream& f, std::string& out) {
  uint64_t len = 0;
  if (!ReadU64(f, len)) {
    return false;
  }
  if (len > kMaxStringBytes ||
      len >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  out.resize(len);
  if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
    return false;
  }
  return true;
}

bool SkipValue(std::ifstream& f, uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
    case kGgufTypeInt8: {
      char dummy;
      return ReadBytes(f, &dummy, 1);
    }
    case kGgufTypeUint16:
    case kGgufTypeInt16: {
      char dummy[2];
      return ReadBytes(f, dummy, 2);
    }
    case kGgufTypeUint32:
    case kGgufTypeInt32:
    case kGgufTypeFloat32: {
      char dummy[4];
      return ReadBytes(f, dummy, 4);
    }
    case kGgufTypeBool: {
      char dummy;
      return ReadBytes(f, &dummy, 1);
    }
    case kGgufTypeString: {
      std::string dummy;
      return ReadString(f, dummy);
    }
    case kGgufTypeUint64:
    case kGgufTypeInt64:
    case kGgufTypeFloat64: {
      char dummy[8];
      return ReadBytes(f, dummy, 8);
    }
    case kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
  out = 1;
  for (uint64_t dim : tensor.shape) {
    if (dim == 0) {
      return false;
    }
    if (!MultiplyChecked(out, dim, out)) {
      return false;
    }
  }
  return true;
}

bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
  uint64_t num_elements = 0;
  if (!TensorNumel(tensor, num_elements)) {
    return false;
  }

  switch (tensor.type) {
    case kGgmlTypeF32:
      return MultiplyChecked(num_elements, 4, out);
    case kGgmlTypeF16:
      return MultiplyChecked(num_elements, 2, out);
    case kGgmlTypeQ40: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 18, out);
    }
    case kGgmlTypeQ41: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 20, out);
    }
    case kGgmlTypeQ80: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 34, out);
    }
    default:
      out = 0;
      return true;
  }
}

bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    return false;
  }
  out = a + b;
  return true;
}

// Read a scalar value and convert to string representation.
bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
  std::ostringstream oss;
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << static_cast<unsigned>(v);
      break;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << static_cast<int>(v);
      break;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeBool: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << (v ? "true" : "false");
      break;
    }
    case kGgufTypeString: {
      if (!ReadString(f, out)) {
        return false;
      }
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) {
          return false;
        }
      }
      break;
    }
    default:
      return false;
  }
  out = oss.str();
  return true;
}

uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
  uint64_t alignment = kDefaultAlignment;
  for (const auto& kv : metadata) {
    if (kv.key != "general.alignment") {
      continue;
    }
    const char* begin = kv.value_str.c_str();
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(begin, &end, 10);
    // strtoull stops at the first '\0'. Compare against value_str.size() so
    // values like "123\0abc" are rejected instead of being treated as 123.
    if (end == begin ||
        static_cast<std::size_t>(end - begin) != kv.value_str.size()) {
      return 0;
    }
    alignment = static_cast<uint64_t>(parsed);
  }
  if (alignment == 0 || alignment > kMaxAlignment) {
    return 0;
  }
  if ((alignment & (alignment - 1)) != 0) {
    return 0;
  }
  return alignment;
}

}  // namespace

// ---------------------------------------------------------------------------
// GgufReader::Load
// ---------------------------------------------------------------------------
bool GgufReader::Load(const std::string& path) {
  version = 0;
  n_tensors = 0;
  n_metadata = 0;
  metadata.clear();
  tensors.clear();
  data_offset = 0;
  load_error.clear();

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    load_error = "failed to open: " + path;
    return false;
  }

  // Magic
  char magic[4];
  if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    load_error = "invalid GGUF magic";
    return false;
  }

  // Version
  if (!ReadU32(f, version)) {
    load_error = "failed to read version";
    return false;
  }
  if (version != 3) {
    load_error = "unsupported GGUF version: " + std::to_string(version);
    return false;
  }

  // n_tensors, n_metadata
  if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
    load_error = "failed to read header counts";
    return false;
  }
  if (n_tensors > kMaxTensorEntries) {
    load_error = "too many tensors: " + std::to_string(n_tensors);
    return false;
  }
  if (n_metadata > kMaxMetadataEntries) {
    load_error = "too many metadata entries: " + std::to_string(n_metadata);
    return false;
  }

  // Metadata
  metadata.reserve(n_metadata);
  std::unordered_set<std::string> metadata_keys;
  for (uint64_t i = 0; i < n_metadata; ++i) {
    GgufMetadataKv kv;
    if (!ReadString(f, kv.key)) {
      load_error =
          "failed to read metadata key at index " + std::to_string(i);
      return false;
    }
    if (!metadata_keys.insert(kv.key).second) {
      load_error = "duplicate metadata key: " + kv.key;
      return false;
    }
    if (!ReadU32(f, kv.type)) {
      load_error = "failed to read metadata type for key: " + kv.key;
      return false;
    }
    if (!ReadValueAsString(f, kv.type, kv.value_str)) {
      load_error = "failed to read metadata value for key: " + kv.key;
      return false;
    }
    metadata.push_back(std::move(kv));
  }

  // Tensor info
  tensors.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo info;
    if (!ReadString(f, info.name)) {
      load_error =
          "failed to read tensor name at index " + std::to_string(i);
      return false;
    }
    uint32_t n_dims = 0;
    if (!ReadU32(f, n_dims)) {
      load_error = "failed to read tensor n_dims for: " + info.name;
      return false;
    }
    if (n_dims == 0 || n_dims > kMaxTensorDims) {
      load_error = "invalid tensor n_dims for: " + info.name;
      return false;
    }
    info.shape.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadU64(f, info.shape[d])) {
        load_error = "failed to read tensor shape for: " + info.name;
        return false;
      }
      if (info.shape[d] == 0) {
        load_error = "invalid zero tensor dimension for: " + info.name;
        return false;
      }
    }
    if (!ReadU32(f, info.type)) {
      load_error = "failed to read tensor type for: " + info.name;
      return false;
    }
    if (!ReadU64(f, info.offset)) {
      load_error = "failed to read tensor offset for: " + info.name;
      return false;
    }
    tensors.push_back(std::move(info));
  }

  std::unordered_set<std::string> names;
  for (const auto& tensor : tensors) {
    if (!names.insert(tensor.name).second) {
      load_error = "duplicate tensor name: " + tensor.name;
      return false;
    }
  }

  // Compute data offset (aligned to general.alignment, default 32).
  std::streampos current = f.tellg();
  if (current < 0) {
    load_error = "failed to determine GGUF data offset";
    return false;
  }
  data_offset = static_cast<uint64_t>(current);
  uint64_t alignment = ParseAlignment(metadata);
  if (alignment == 0) {
    load_error = "invalid general.alignment";
    return false;
  }
  uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
  data_offset += padding;

  f.seekg(0, std::ios::end);
  const std::streampos end = f.tellg();
  if (end < 0) {
    load_error = "failed to determine GGUF file size";
    return false;
  }
  const uint64_t file_size = static_cast<uint64_t>(end);
  if (n_tensors > 0 && file_size < data_offset) {
    load_error = "GGUF file ended before tensor data section";
    return false;
  }

  for (const auto& tensor : tensors) {
    uint64_t tensor_size = 0;
    if (!TensorDataSize(tensor, tensor_size)) {
      load_error = "invalid tensor data size for: " + tensor.name;
      return false;
    }
    if (tensor_size == 0) {
      continue;
    }

    uint64_t tensor_start = 0;
    if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
      load_error = "tensor offset overflow for: " + tensor.name;
      return false;
    }

    uint64_t tensor_end = 0;
    if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
      load_error = "tensor data range overflow for: " + tensor.name;
      return false;
    }

    if (tensor_end > file_size) {
      load_error = "tensor data outside file bounds: " + tensor.name;
      return false;
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// InspectGguf
// ---------------------------------------------------------------------------
void InspectGguf(const GgufReader& reader) {
  std::cout << "=== GGUF Info ===\n"
            << "  version:      " << reader.version << "\n"
            << "  n_tensors:    " << reader.n_tensors << "\n"
            << "  n_metadata:   " << reader.n_metadata << "\n"
            << "  data_offset:  " << reader.data_offset << "\n";

  for (const GgufMetadataKv& kv : reader.metadata) {
    std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
              << "): " << kv.value_str << "\n";
  }

  for (const GgufTensorInfo& tensor : reader.tensors) {
    std::cout << "  " << tensor.name << "\n"
              << "    shape:  [";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << tensor.shape[i];
    }
    std::cout << "]\n"
              << "    dtype:  " << GgmlTypeName(tensor.type) << "\n"
              << "    offset: " << tensor.offset << "\n";
  }
}

// ---------------------------------------------------------------------------
// Tensor data reading (inside anonymous namespace for helper access)
// ---------------------------------------------------------------------------

bool GgufTensorDataSize(const GgufTensorInfo& tensor, uint64_t& out_size) {
  return TensorDataSize(tensor, out_size);
}

std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset) {
  std::vector<uint8_t> result;

  uint64_t tensor_size = 0;
  if (!TensorDataSize(info, tensor_size)) {
    return result;
  }
  if (tensor_size == 0) {
    return result;
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return result;
  }

  uint64_t tensor_start = 0;
  if (!AddChecked(data_offset, info.offset, tensor_start)) {
    return result;
  }

  if (tensor_start >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    return result;
  }

  f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
  if (!f.good()) {
    return result;
  }

  result.resize(tensor_size);
  f.read(reinterpret_cast<char*>(result.data()),
         static_cast<std::streamsize>(tensor_size));
  if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
    result.clear();
    return result;
  }

  return result;
}

namespace {

// Open GGUF, skip to metadata, find key. On success, `f` is positioned
// after the key string and before the type uint32.
bool OpenAndFindMetadataKey(std::ifstream& f, const std::string& path,
                            const std::string& key, uint32_t& out_type) {
  f.open(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }

  // Skip magic (4) + version (4)
  f.seekg(8, std::ios::beg);
  if (!f.good()) {
    return false;
  }

  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
    return false;
  }

  for (uint64_t i = 0; i < n_metadata; ++i) {
    std::string read_key;
    if (!ReadString(f, read_key)) {
      return false;
    }
    uint32_t type = 0;
    if (!ReadU32(f, type)) {
      return false;
    }
    if (read_key == key) {
      out_type = type;
      return true;
    }
    // Skip value
    if (!SkipValue(f, type)) {
      return false;
    }
  }
  return false;  // key not found
}

}  // namespace

// ---------------------------------------------------------------------------
// Typed metadata reading
// ---------------------------------------------------------------------------

bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    default:
      return false;
  }
}

bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  switch (type) {
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<double>(v);
      return true;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    default:
      return false;
  }
}

bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeString) {
    return false;
  }
  return ReadString(f, out);
}

bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeArray) {
    return false;
  }
  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
    return false;
  }
  if (elem_type != kGgufTypeString) {
    return false;
  }
  if (len > kMaxArrayElements) {
    return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; ++i) {
    std::string s;
    if (!ReadString(f, s)) {
      return false;
    }
    out.push_back(std::move(s));
  }
  return true;
}

bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeArray) {
    return false;
  }
  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
    return false;
  }
  if (elem_type != kGgufTypeInt32) {
    return false;
  }
  if (len > kMaxArrayElements) {
    return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; ++i) {
    int32_t v = 0;
    if (!ReadLe(f, v)) {
      return false;
    }
    out.push_back(v);
  }
  return true;
}

}  // namespace mini_llama
