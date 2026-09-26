#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/debug.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/request_context.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/tokenizer.h"

namespace {

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program << " --help\n"
      << "  " << program
      << " generate [--model path|dir] [--tokenizer vocab.json] [-p prompt] "
         "[-n tokens] [--temperature T] [--top-k k] [--seed S] [--threads N] "
         "[--quant q8_0|q4_0]\n"
      << "  " << program << " inspect <model-path|dir>\n"
      << "  " << program << " inspect-gguf <path>\n"
      << "  " << program
      << " run [model-path|dir] [--tokenizer vocab.json] [-n tokens] "
         "[--temperature T] [--top-k k] [--seed S] [--synthetic]\n"
      << "  " << program
      << " bench <model-path|dir> [-p prompt] [-n tokens] [--seed S] "
         "[--tokenizer vocab.json] [--quant q8_0|q4_0] [--threads N] "
         "[--verbose]\n\n"
      << "Commands:\n"
      << "  generate     Tokenize, run CPU forward, and sample tokens.\n"
      << "  inspect      Print model.json metadata and load JSON+BIN weights.\n"
      << "  inspect-gguf Print GGUF header, metadata, and tensor index.\n"
      << "  run          Interactive multi-turn chat.\n"
      << "  bench        Measure prefill / decode timing and weight memory.\n";
}

bool ParsePositiveInt(const std::string& text, int& value) {
  try {
    std::size_t parsed = 0;
    int number = std::stoi(text, &parsed);
    if (parsed != text.size() || number <= 0) {
      return false;
    }
    value = number;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseNonNegativeInt(const std::string& text, int& value) {
  try {
    std::size_t parsed = 0;
    int number = std::stoi(text, &parsed);
    if (parsed != text.size() || number < 0) {
      return false;
    }
    value = number;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseNonNegativeFloat(const std::string& text, float& value) {
  try {
    std::size_t parsed = 0;
    float number = std::stof(text, &parsed);
    if (parsed != text.size() || !std::isfinite(number) || number < 0.0f) {
      return false;
    }
    value = number;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseUnsigned(const std::string& text, unsigned int& value) {
  if (text.empty() || text[0] == '-') {
    return false;
  }
  try {
    std::size_t parsed = 0;
    unsigned long number = std::stoul(text, &parsed);
    if (parsed != text.size() ||
        number > std::numeric_limits<unsigned int>::max()) {
      return false;
    }
    value = static_cast<unsigned int>(number);
    return true;
  } catch (...) {
    return false;
  }
}

std::string FindGgufInDirectory(const std::string& dir,
                                bool skip_inspect_fixture = false) {
  std::vector<std::filesystem::path> candidates;
  std::error_code error_code;
  for (const auto& entry :
       std::filesystem::directory_iterator(dir, error_code)) {
    if (error_code) {
      break;
    }
    std::error_code file_error;
    if (entry.is_regular_file(file_error) && !file_error &&
        entry.path().extension() == ".gguf") {
      if (skip_inspect_fixture &&
          entry.path().filename() == "test.gguf") {
        continue;
      }
      candidates.push_back(entry.path());
    }
  }
  if (candidates.empty()) {
    return "";
  }
  std::sort(candidates.begin(), candidates.end());
  if (candidates.size() > 1) {
    std::cerr << "warning: multiple .gguf files in " << dir << ", using "
              << candidates.front().filename().string() << "\n";
  }
  return candidates.front().string();
}

bool EndsWithGguf(const std::string& path) {
  if (path.size() < 5) {
    return false;
  }
  std::string ext = path.substr(path.size() - 5);
  for (char& ch : ext) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return ext == ".gguf";
}

std::unique_ptr<mini_llama::ITokenizer> CreateTokenizerFromVocabHint(
    const std::string& vocab_path) {
  if (!std::filesystem::exists(vocab_path)) {
    return nullptr;
  }
  const std::filesystem::path vocab(vocab_path);
  const std::filesystem::path dir = vocab.parent_path();
  const std::filesystem::path merges = dir / "merges.txt";
  const std::filesystem::path special = dir / "special_tokens.json";
  if (std::filesystem::exists(merges)) {
    return mini_llama::CreateBpeTokenizer(vocab.string(), merges.string(),
                                          special.string());
  }
  try {
    return std::make_unique<mini_llama::JsonVocabTokenizer>(vocab.string());
  } catch (const std::exception&) {
    return nullptr;
  }
}

std::unique_ptr<mini_llama::ITokenizer> LoadTokenizerForGguf(
    const std::string& gguf_path, const std::string& explicit_tokenizer_path) {
  if (!explicit_tokenizer_path.empty()) {
    return CreateTokenizerFromVocabHint(explicit_tokenizer_path);
  }

  std::unique_ptr<mini_llama::ITokenizer> tokenizer =
      mini_llama::CreateGgufTokenizer(gguf_path);
  if (tokenizer) {
    return tokenizer;
  }

  const std::filesystem::path gguf_dir =
      std::filesystem::path(gguf_path).parent_path();
  const std::string vocab_path = (gguf_dir / "vocab.json").string();
  const std::string merges_path = (gguf_dir / "merges.txt").string();
  const std::string special_path = (gguf_dir / "special_tokens.json").string();
  if (std::filesystem::exists(vocab_path) &&
      std::filesystem::exists(merges_path)) {
    return mini_llama::CreateBpeTokenizer(vocab_path, merges_path, special_path);
  }
  if (std::filesystem::exists(vocab_path)) {
    try {
      return std::make_unique<mini_llama::JsonVocabTokenizer>(vocab_path);
    } catch (const std::exception&) {
      return nullptr;
    }
  }
  return nullptr;
}

mini_llama::MiniLlamaModel LoadWeightsFromPath(const std::string& path) {
  std::string model_path = path;
  if (std::filesystem::is_directory(path)) {
    const std::filesystem::path json_path =
        std::filesystem::path(path) / "model.json";
    const std::filesystem::path bin_path =
        std::filesystem::path(path) / "model.bin";
    if (std::filesystem::exists(json_path) &&
        std::filesystem::exists(bin_path)) {
      return mini_llama::LoadModel(json_path.string(), bin_path.string());
    }
    const std::string gguf_path = FindGgufInDirectory(path);
    if (!gguf_path.empty()) {
      model_path = gguf_path;
    }
  }

  if (EndsWithGguf(model_path)) {
    return mini_llama::LoadGgufModel(model_path);
  }

  std::filesystem::path file(model_path);
  std::filesystem::path json_path;
  std::filesystem::path bin_path;
  if (file.extension() == ".json") {
    json_path = file;
    bin_path = file.parent_path() / "model.bin";
  } else {
    json_path = file.parent_path() / "model.json";
    bin_path = file;
  }
  return mini_llama::LoadModel(json_path.string(), bin_path.string());
}

std::string ResolveModelPath(const std::string& path) {
  if (!std::filesystem::is_directory(path)) {
    return path;
  }
  const std::filesystem::path json_path =
      std::filesystem::path(path) / "model.json";
  const std::filesystem::path bin_path =
      std::filesystem::path(path) / "model.bin";
  if (std::filesystem::exists(json_path) &&
      std::filesystem::exists(bin_path)) {
    return path;
  }
  const std::string gguf_path = FindGgufInDirectory(path);
  if (!gguf_path.empty()) {
    return gguf_path;
  }
  return path;
}

std::unique_ptr<mini_llama::ITokenizer> LoadTokenizerForResolved(
    const std::string& model_path, const std::string& resolved,
    const std::string& explicit_tokenizer_path) {
  if (EndsWithGguf(resolved)) {
    return LoadTokenizerForGguf(resolved, explicit_tokenizer_path);
  }
  if (!explicit_tokenizer_path.empty()) {
    return CreateTokenizerFromVocabHint(explicit_tokenizer_path);
  }

  std::filesystem::path json_path;
  if (std::filesystem::is_directory(model_path)) {
    json_path = std::filesystem::path(model_path) / "model.json";
  } else {
    const std::filesystem::path file(model_path);
    json_path = file.extension() == ".json"
                    ? file
                    : file.parent_path() / "model.json";
  }
  if (!std::filesystem::exists(json_path)) {
    return mini_llama::CreateTokenizer("");
  }

  try {
    mini_llama::ModelManifest manifest =
        mini_llama::ParseManifest(json_path.string());
    if (manifest.tokenizer.type == "json_vocab") {
      std::filesystem::path tokenizer_path = manifest.tokenizer.path;
      if (tokenizer_path.empty()) {
        return nullptr;
      }
      if (tokenizer_path.is_relative()) {
        tokenizer_path = json_path.parent_path() / tokenizer_path;
      }
      if (!std::filesystem::exists(tokenizer_path)) {
        return nullptr;
      }
      return std::make_unique<mini_llama::JsonVocabTokenizer>(
          tokenizer_path.string());
    }
  } catch (const std::exception&) {
    return nullptr;
  }
  return mini_llama::CreateTokenizer("");
}

void ApplyQuantOverride(mini_llama::MiniLlamaModel& model,
                        const std::string& quant_type) {
  if (quant_type.empty()) {
    return;
  }
  if (quant_type == "q8_0") {
    mini_llama::QuantizeModelToQ80(model);
    return;
  }
  if (quant_type == "q4_0") {
    mini_llama::QuantizeModelToQ40(model);
    return;
  }
  throw std::runtime_error("unsupported quant type: " + quant_type);
}

mini_llama::Tensor RunLogitsForTokens(const mini_llama::MiniLlamaModel& model,
                                      const std::vector<int>& tokens) {
  mini_llama::MiniLlamaContext ctx(&model);
  mini_llama::MiniBatch batch = mini_llama::MiniBatch::FromTokens(tokens, 0);
  return mini_llama::ForwardBatch(ctx, model, batch);
}

std::pair<float, float> LogitsError(const mini_llama::Tensor& baseline,
                                    const mini_llama::Tensor& candidate) {
  if (baseline.shape != candidate.shape) {
    throw std::runtime_error(
        "LogitsError: shape mismatch, baseline=" + baseline.ShapeStringShort() +
        ", candidate=" + candidate.ShapeStringShort());
  }
  float max_err = 0.0f;
  float sum_err = 0.0f;
  for (size_t i = 0; i < baseline.size(); ++i) {
    float err = std::abs(baseline.data[i] - candidate.data[i]);
    max_err = std::max(max_err, err);
    sum_err += err;
  }
  return {max_err, sum_err / static_cast<float>(baseline.size())};
}

void PrintCpuBanner() {
  std::cout << "mini-llama\n==============\n\nbackend: cpu\ncompute: cpu\n";
}

void PrintIntList(const char* label, const std::vector<int>& values) {
  std::cout << label << " [";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << "]\n";
}

int FailRequest(mini_llama::RequestContext& request, const std::string& message) {
  request.SetError(message);
  request.Finish();
  mini_llama::PrintRequestTrace(request, std::cerr);
  std::cerr << request.error << "\n";
  return 1;
}

int RunGenerate(int argc, char** argv) {
  std::string prompt = "hello";
  int n_predict = 8;
  int n_threads = 0;
  mini_llama::SamplingParams sampling_params;
  std::string model_path = "models/tiny";
  std::string explicit_tokenizer_path;
  std::string quant_type;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-p" || arg == "--prompt") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      prompt = argv[++i];
    } else if (arg == "-n" || arg == "--n-predict") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParsePositiveInt(argv[++i], n_predict)) {
        std::cerr << "n-predict must be a positive integer\n";
        return 1;
      }
    } else if (arg == "--temperature") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseNonNegativeFloat(argv[++i], sampling_params.temperature)) {
        std::cerr << "temperature must be a non-negative float\n";
        return 1;
      }
    } else if (arg == "--top-k") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseNonNegativeInt(argv[++i], sampling_params.top_k)) {
        std::cerr << "top-k must be a non-negative integer\n";
        return 1;
      }
    } else if (arg == "--seed") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseUnsigned(argv[++i], sampling_params.seed)) {
        std::cerr << "seed must be a non-negative integer\n";
        return 1;
      }
    } else if (arg == "--threads") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseNonNegativeInt(argv[++i], n_threads)) {
        std::cerr << "threads must be a non-negative integer\n";
        return 1;
      }
    } else if (arg == "--model") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      model_path = argv[++i];
    } else if (arg == "--tokenizer") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      explicit_tokenizer_path = argv[++i];
    } else if (arg == "--quant") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      quant_type = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown option: " << arg << "\n";
      return 1;
    }
  }

  if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
    std::cerr << "Invalid --quant value: " << quant_type
              << ". Supported values: q8_0, q4_0.\n";
    return 1;
  }

  PrintCpuBanner();

  const std::string resolved = ResolveModelPath(model_path);
  mini_llama::RequestContext request =
      mini_llama::StartRequest("generate", "cpu", resolved);

  auto stage_start = mini_llama::RequestClock::now();
  mini_llama::MiniLlamaModel model = LoadWeightsFromPath(model_path);
  request.model_load_ms = mini_llama::ElapsedMs(stage_start);
  request.RecordEvent("model_load", request.model_load_ms, 0, resolved);
  if (!model.loaded) {
    return FailRequest(request, "Failed to load model: " + model.load_error);
  }
  const std::string quant_label = quant_type.empty() ? "model-native" : quant_type;
  try {
    ApplyQuantOverride(model, quant_type);
  } catch (const std::exception& e) {
    return FailRequest(request, "Quantization failed: " + std::string(e.what()));
  }
  request.RecordEvent("quantize", 0.0, 0, quant_label);

  std::unique_ptr<mini_llama::ITokenizer> tokenizer;
  if (EndsWithGguf(resolved)) {
    tokenizer = LoadTokenizerForGguf(resolved, explicit_tokenizer_path);
  } else if (!explicit_tokenizer_path.empty()) {
    tokenizer = CreateTokenizerFromVocabHint(explicit_tokenizer_path);
  } else {
    std::filesystem::path json_path;
    if (std::filesystem::is_directory(model_path)) {
      json_path = std::filesystem::path(model_path) / "model.json";
    } else {
      const std::filesystem::path file(model_path);
      json_path = file.extension() == ".json"
                      ? file
                      : file.parent_path() / "model.json";
    }
    mini_llama::ModelManifest manifest;
    try {
      manifest = mini_llama::ParseManifest(json_path.string());
    } catch (const std::exception& e) {
      return FailRequest(request,
                         "Failed to parse model.json: " + std::string(e.what()));
    }
    if (manifest.tokenizer.type == "json_vocab") {
      std::filesystem::path tokenizer_path = manifest.tokenizer.path;
      if (tokenizer_path.empty()) {
        return FailRequest(request, "Failed to load tokenizer.");
      }
      if (tokenizer_path.is_relative()) {
        tokenizer_path = json_path.parent_path() / tokenizer_path;
      }
      if (!std::filesystem::exists(tokenizer_path)) {
        return FailRequest(request, "Failed to load tokenizer.");
      }
      try {
        tokenizer = std::make_unique<mini_llama::JsonVocabTokenizer>(
            tokenizer_path.string());
      } catch (const std::exception&) {
        return FailRequest(request, "Failed to load tokenizer.");
      }
    } else {
      tokenizer = mini_llama::CreateTokenizer("");
    }
  }

  if (!tokenizer) {
    return FailRequest(request, "Failed to load tokenizer.");
  }
  if (model.config.vocab_size < tokenizer->vocab_size()) {
    return FailRequest(request, "Model vocab_size must be at least " +
                                    std::to_string(tokenizer->vocab_size()) +
                                    " for the tokenizer.");
  }

  stage_start = mini_llama::RequestClock::now();
  const std::vector<int> tokens = tokenizer->Encode(prompt);
  request.tokenize_ms = mini_llama::ElapsedMs(stage_start);
  request.prompt_tokens = static_cast<int>(tokens.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                      "prompt");

  std::cout << "prompt: " << prompt << "\n";
  PrintIntList("tokens:", tokens);
  mini_llama::SetThreadCount(n_threads);
  std::cout << "sampling: temperature=" << sampling_params.temperature
            << ", top_k=" << sampling_params.top_k
            << ", seed=" << sampling_params.seed << "\n"
            << "quant: " << quant_label << "\n"
            << "threads: " << mini_llama::GetThreadCount() << "\n\n";

  if (tokens.empty()) {
    return FailRequest(request, "Prompt produced no tokens.");
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    return FailRequest(request, "Requested tokens exceed context window.");
  }

  mini_llama::MiniSampler sampler(sampling_params);
  mini_llama::MiniLlamaContext ctx(&model);
  std::vector<int> generated;
  try {
    std::cout << "prefill...\n";
    stage_start = mini_llama::RequestClock::now();
    mini_llama::Tensor logits = mini_llama::ForwardBatch(
        ctx, model, mini_llama::MiniBatch::FromTokens(tokens, 0));
    request.prefill_ms = mini_llama::ElapsedMs(stage_start);
    request.prefill_tokens = static_cast<int>(tokens.size());
    ctx.n_prefill_tokens += request.prefill_tokens;
    request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                        "batch");

    std::cout << "decode loop...\n";
    generated.reserve(static_cast<size_t>(n_predict));
    for (int i = 0; i < n_predict; ++i) {
      stage_start = mini_llama::RequestClock::now();
      const int next = sampler.Sample(logits, sampling_params);
      request.sample_ms += mini_llama::ElapsedMs(stage_start);
      generated.push_back(next);
      if (next == tokenizer->eos_id() || i + 1 == n_predict) {
        break;
      }

      stage_start = mini_llama::RequestClock::now();
      logits = mini_llama::ForwardBatch(
          ctx, model, mini_llama::MiniBatch::Single(next, ctx.pos + 1));
      const double decode_ms = mini_llama::ElapsedMs(stage_start);
      request.decode_ms += decode_ms;
      ++request.decode_tokens;
      ++ctx.n_decode_tokens;
      request.RecordEvent("decode", decode_ms, 1,
                          "pos=" + std::to_string(ctx.pos));
    }
  } catch (const std::exception& e) {
    return FailRequest(request, "Inference failed: " + std::string(e.what()));
  }

  request.generated_tokens = static_cast<int>(generated.size());
  request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                      "generated_tokens");
  request.Finish();
  std::cout << "\n";
  PrintIntList("generated tokens:", generated);
  std::cout << "generated text: \"" << tokenizer->Decode(generated) << "\"\n";
  mini_llama::PrintRequestTrace(request, std::cout);
  return 0;
}

int RunInspect(int argc, char** argv) {
  if (argc >= 3 && (std::string(argv[2]) == "-h" ||
                    std::string(argv[2]) == "--help")) {
    PrintUsage(argv[0]);
    return 0;
  }
  if (argc < 3) {
    std::cerr << "Missing model path.\n";
    PrintUsage(argv[0]);
    return 1;
  }
  if (argc > 3) {
    std::cerr << "unknown option: " << argv[3] << "\n";
    return 1;
  }

  const std::string model_path = argv[2];
  std::filesystem::path json_path;
  std::filesystem::path bin_path;
  if (std::filesystem::is_directory(model_path)) {
    json_path = std::filesystem::path(model_path) / "model.json";
    bin_path = std::filesystem::path(model_path) / "model.bin";
  } else {
    const std::filesystem::path file(model_path);
    if (file.extension() == ".json") {
      json_path = file;
      bin_path = file.parent_path() / "model.bin";
    } else {
      json_path = file.parent_path() / "model.json";
      bin_path = file;
    }
  }

  if (!std::filesystem::exists(json_path)) {
    std::cerr << "model.json not found: " << json_path.string() << "\n";
    return 1;
  }
  if (!mini_llama::InspectModel(json_path.string())) {
    return 1;
  }
  if (!std::filesystem::exists(bin_path)) {
    std::cerr << "model.bin not found: " << bin_path.string() << "\n";
    return 1;
  }

  mini_llama::MiniLlamaModel model =
      mini_llama::LoadModel(json_path.string(), bin_path.string());
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }
  std::cout << "backend: cpu\n";
  return 0;
}

int RunInspectGguf(int argc, char** argv) {
  if (argc >= 3 && (std::string(argv[2]) == "-h" ||
                    std::string(argv[2]) == "--help")) {
    std::cout << "Usage: " << argv[0] << " inspect-gguf <path>\n";
    return 0;
  }
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
    return 1;
  }
  if (argc > 3) {
    std::cerr << "unknown option: " << argv[3] << "\n";
    return 1;
  }

  mini_llama::GgufReader reader;
  if (!reader.Load(argv[2])) {
    std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
    return 1;
  }
  mini_llama::InspectGguf(reader);
  return 0;
}

void PrintBenchUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " bench <model-path|dir> [options]\n"
      << "Options:\n"
      << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>   Number of tokens to generate (default: 64)\n"
      << "  --seed <S>            Random seed (default: 0)\n"
      << "  --tokenizer <path>    Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0     Quantize loaded Linear weights before "
         "benchmark\n"
      << "  --threads <n>         Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --verbose             Print debug dumps after each step\n"
      << "  -h, --help            Show this help\n";
}

int RunBench(int argc, char** argv) {
  if (argc >= 3 && (std::string(argv[2]) == "-h" ||
                    std::string(argv[2]) == "--help")) {
    PrintBenchUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintBenchUsage(argv[0]);
    return 1;
  }

  std::string model_dir = argv[2];
  std::string prompt = "hello";
  int n_predict = 64;
  unsigned int seed = 0;
  std::string tokenizer_path;
  bool verbose = false;
  std::string quant_type;
  int n_threads = 0;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
      prompt = argv[++i];
    } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
      if (!ParseNonNegativeInt(argv[++i], n_predict)) {
        std::cerr << "Invalid --n-predict value.\n";
        return 1;
      }
    } else if (arg == "--seed" && i + 1 < argc) {
      if (!ParseUnsigned(argv[++i], seed)) {
        std::cerr << "Invalid --seed value.\n";
        return 1;
      }
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (arg == "--quant" && i + 1 < argc) {
      quant_type = argv[++i];
    } else if (arg == "--threads" && i + 1 < argc) {
      if (!ParseNonNegativeInt(argv[++i], n_threads)) {
        std::cerr << "Invalid --threads value.\n";
        return 1;
      }
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "-h" || arg == "--help") {
      PrintBenchUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintBenchUsage(argv[0]);
      return 1;
    }
  }

  if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
    std::cerr << "Invalid --quant value: " << quant_type
              << ". Supported values: q8_0, q4_0.\n";
    return 1;
  }

  mini_llama::MiniLlamaModel model = LoadWeightsFromPath(model_dir);
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  const std::string resolved = ResolveModelPath(model_dir);
  std::unique_ptr<mini_llama::ITokenizer> tokenizer =
      LoadTokenizerForResolved(model_dir, resolved, tokenizer_path);
  if (!tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (model.config.vocab_size < tokenizer->vocab_size()) {
    std::cerr << "Model vocab_size must be at least "
              << tokenizer->vocab_size() << " for the tokenizer.\n";
    return 1;
  }

  std::vector<int> tokens = tokenizer->Encode(prompt);
  if (tokens.empty()) {
    std::cerr << "Prompt produced no tokens.\n";
    return 1;
  }
  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    std::cerr << "Prompt too long.\n";
    return 1;
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    n_predict = model.config.max_seq_len - static_cast<int>(tokens.size());
  }

  mini_llama::MiniLlamaModel baseline_model;
  if (!quant_type.empty()) {
    baseline_model = model;
  }
  try {
    ApplyQuantOverride(model, quant_type);
  } catch (const std::exception& e) {
    std::cerr << "Quantization failed: " << e.what() << "\n";
    return 1;
  }

  std::cout << "Benchmark: " << model_dir << "\n";
  std::cout << "  backend: cpu\n";
  std::cout << "  compute: cpu\n";
  std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size()
            << " tokens)\n";
  mini_llama::SetThreadCount(n_threads);
  std::cout << "  n_predict: " << n_predict << "\n";
  std::cout << "  seed: " << seed << "\n";
  std::cout << "  quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n";
  std::cout << "  threads: " << mini_llama::GetThreadCount() << "\n";
  std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";

  mini_llama::BenchmarkResult result =
      mini_llama::RunBenchmark(model, tokens, n_predict, seed, verbose);

  std::cout << "Results:\n";
  std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
  std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
  std::cout << "  Decode tokens:     " << result.n_decode_tokens << "\n";
  std::cout << "  prefill time:      " << std::fixed << std::setprecision(2)
            << result.prefill_ms << " ms\n";
  std::cout << "  Decode time:       " << std::fixed << std::setprecision(2)
            << result.decode_ms << " ms\n";
  std::cout << "  total time:        " << std::fixed << std::setprecision(2)
            << (result.prefill_ms + result.decode_ms) << " ms\n";
  std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2)
            << result.tokens_per_sec() << "\n";
  std::cout << "  tokens/s (Decode): " << std::fixed << std::setprecision(2)
            << result.decode_tokens_per_sec() << "\n";

  size_t actual_bytes = mini_llama::ModelWeightBytes(model);
  size_t f32_bytes = mini_llama::ModelWeightBytesF32(model);
  std::cout << "\n  weight memory:\n";
  std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << (actual_bytes / (1024.0 * 1024.0))
            << " MB)\n";
  std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << (f32_bytes / (1024.0 * 1024.0))
            << " MB)\n";
  double savings = actual_bytes == 0
                       ? 0.0
                       : static_cast<double>(f32_bytes) / actual_bytes;
  std::cout << "    savings:   " << std::fixed << std::setprecision(2)
            << savings << "x compression\n";

  if (!quant_type.empty()) {
    try {
      mini_llama::Tensor baseline_logits =
          RunLogitsForTokens(baseline_model, tokens);
      mini_llama::Tensor quant_logits = RunLogitsForTokens(model, tokens);
      auto err = LogitsError(baseline_logits, quant_logits);

      std::cout << "\n  logits error vs model-native:\n";
      std::cout << "    max:  " << std::scientific << std::setprecision(3)
                << err.first << "\n";
      std::cout << "    mean: " << std::scientific << std::setprecision(3)
                << err.second << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to compute logits error: " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return "";
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

int RunChat(int argc, char** argv) {
  std::string model_path = "models/tiny";
  std::string explicit_tokenizer_path;
  int max_response_tokens = 64;
  mini_llama::SamplingParams sampling_params;
  bool use_synthetic = false;
  int argi = 2;
  if (argi < argc && argv[argi][0] != '-') {
    model_path = argv[argi++];
  }

  for (int i = argi; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-n" || arg == "--n-predict") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParsePositiveInt(argv[++i], max_response_tokens)) {
        std::cerr << "n-predict must be a positive integer\n";
        return 1;
      }
    } else if (arg == "--temperature") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseNonNegativeFloat(argv[++i], sampling_params.temperature)) {
        std::cerr << "temperature must be a non-negative float\n";
        return 1;
      }
    } else if (arg == "--top-k") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseNonNegativeInt(argv[++i], sampling_params.top_k)) {
        std::cerr << "top-k must be a non-negative integer\n";
        return 1;
      }
    } else if (arg == "--seed") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      if (!ParseUnsigned(argv[++i], sampling_params.seed)) {
        std::cerr << "seed must be a non-negative integer\n";
        return 1;
      }
    } else if (arg == "--tokenizer") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        return 1;
      }
      explicit_tokenizer_path = argv[++i];
    } else if (arg == "--synthetic") {
      use_synthetic = true;
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown option: " << arg << "\n";
      return 1;
    }
  }

  std::error_code exists_error;
  if (!std::filesystem::exists(model_path, exists_error) || exists_error) {
    std::cerr << "Model path not found: " << model_path << "\n";
    return 1;
  }

  std::unique_ptr<mini_llama::ITokenizer> tokenizer;
  std::string chat_template;
  std::string resolved_path = model_path;
  mini_llama::MiniLlamaModel model;
  bool gguf_loaded = false;
  if (use_synthetic) {
    if (!std::filesystem::is_directory(model_path)) {
      std::cerr << "--synthetic requires a model directory with vocab.json.\n";
      return 1;
    }
    if (!std::filesystem::exists(std::filesystem::path(model_path) /
                                 "vocab.json")) {
      std::cerr << "No vocab.json in directory: " << model_path << "\n";
      return 1;
    }
  } else if (std::filesystem::is_directory(model_path)) {
    const std::string gguf_path =
        FindGgufInDirectory(model_path, /*skip_inspect_fixture=*/true);
    if (!gguf_path.empty()) {
      mini_llama::MiniLlamaModel loaded = mini_llama::LoadGgufModel(gguf_path);
      if (!loaded.loaded) {
        std::cerr << "Failed to load model: " << loaded.load_error << "\n";
        return 1;
      }
      resolved_path = gguf_path;
      model = std::move(loaded);
      gguf_loaded = true;
    } else if (!std::filesystem::exists(
                   std::filesystem::path(model_path) / "vocab.json")) {
      std::cerr << "No GGUF model or vocab.json in directory: " << model_path
                << "\n";
      return 1;
    }
  } else if (!EndsWithGguf(model_path)) {
    std::cerr << "Expected a .gguf file or a model directory: " << model_path
              << "\n";
    return 1;
  }

  auto load_dir_tokenizer = [&](const std::filesystem::path& dir) {
    const std::string vocab_path = (dir / "vocab.json").string();
    const std::string merges_path = (dir / "merges.txt").string();
    const std::string special_path = (dir / "special_tokens.json").string();
    if (std::filesystem::exists(merges_path) &&
        std::filesystem::exists(vocab_path)) {
      tokenizer = mini_llama::CreateBpeTokenizer(vocab_path, merges_path,
                                                 special_path);
    } else if (std::filesystem::exists(vocab_path)) {
      try {
        tokenizer =
            std::make_unique<mini_llama::JsonVocabTokenizer>(vocab_path);
      } catch (const std::exception&) {
        tokenizer.reset();
      }
    }
    if (chat_template.empty()) {
      chat_template = ReadTextFile((dir / "chat_template.txt").string());
    }
  };

  if (EndsWithGguf(resolved_path)) {
    if (!gguf_loaded) {
      model = mini_llama::LoadGgufModel(resolved_path);
      if (!model.loaded) {
        std::cerr << "Failed to load model: " << model.load_error << "\n";
        return 1;
      }
    }
    tokenizer = LoadTokenizerForGguf(resolved_path, explicit_tokenizer_path);
    chat_template = mini_llama::LoadChatTemplateFromGguf(resolved_path);
    const std::filesystem::path gguf_dir =
        std::filesystem::path(resolved_path).parent_path();
    if (chat_template.empty()) {
      chat_template = ReadTextFile((gguf_dir / "chat_template.txt").string());
    }
  } else {
    load_dir_tokenizer(std::filesystem::path(model_path));
    if (!explicit_tokenizer_path.empty()) {
      tokenizer = CreateTokenizerFromVocabHint(explicit_tokenizer_path);
    }
    if (!tokenizer) {
      std::cerr << "Failed to load tokenizer.\n";
      return 1;
    }
    mini_llama::ModelConfig synthetic;
    synthetic.vocab_size = tokenizer->vocab_size();
    synthetic.max_seq_len = 256;
    model = mini_llama::MakeCpuTestModel(synthetic);
  }

  if (!tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (model.config.vocab_size < tokenizer->vocab_size()) {
    std::cerr << "Model vocab_size must be at least "
              << tokenizer->vocab_size() << " for the tokenizer.\n";
    return 1;
  }
  const mini_llama::ModelConfig& config = model.config;

  mini_llama::PromptBuilder builder;
  if (!chat_template.empty()) {
    builder.SetChatTemplate(chat_template);
  }
  mini_llama::Terminal term;
  mini_llama::ChatSession session;
  session.sampling_params = sampling_params;
  if (chat_template.empty()) {
    session.AddMessage("system", "You are a helpful assistant.");
  }

  term.PrintMessage("mini-llama chat");
  term.PrintMessage("backend: cpu");
  term.PrintMessage("compute: cpu");
  term.PrintMessage("Type /help for commands, /exit to quit.\n");
  if (EndsWithGguf(resolved_path)) {
    term.PrintMessage("CPU Forward on GGUF weights (dequantized to F32).\n");
  } else {
    term.PrintMessage("CPU Forward on synthetic F32 weights.");
    if (config.max_seq_len <= 256) {
      term.PrintMessage(
          "Tiny teaching model: random weights, small context window.\n");
    }
  }

  mini_llama::MiniLlamaContext ctx(&model);

  while (true) {
    term.PrintUserPrompt();
    std::string input = term.ReadLine();
    if (input.empty() && std::cin.eof()) {
      break;
    }

    if (input == "/help") {
      term.PrintHelp();
      continue;
    }
    if (input == "/exit") {
      break;
    }
    if (input == "/clear") {
      session.Clear();
      ctx = mini_llama::MiniLlamaContext(&model);
      if (chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");
      }
      term.PrintMessage("Chat history cleared.\n");
      continue;
    }
    if (input == "/stats") {
      term.PrintStats(session);
      continue;
    }
    if (input == "/params") {
      term.PrintParams(session.sampling_params);
      continue;
    }
    if (!input.empty() && input[0] == '/') {
      term.PrintMessage("Unknown command: " + input + "\n");
      continue;
    }
    if (input.empty()) {
      continue;
    }

    mini_llama::RequestContext request =
        mini_llama::StartRequest("run", "cpu", resolved_path);

    std::vector<mini_llama::ChatMessage> candidate_messages = session.messages;
    candidate_messages.push_back({"user", input});

    auto stage_start = mini_llama::RequestClock::now();
    const std::string prompt_text = builder.Build(candidate_messages);
    request.RecordEvent(
        "prompt_build", mini_llama::ElapsedMs(stage_start), 0,
        "messages=" + std::to_string(candidate_messages.size()));
    stage_start = mini_llama::RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt_text);
    request.tokenize_ms = mini_llama::ElapsedMs(stage_start);
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                        "prompt");

    if (tokens.empty()) {
      request.SetError("Prompt produced no tokens.");
      request.Finish();
      mini_llama::PrintRequestTrace(request);
      term.PrintMessage("Error: " + request.error + "\n");
      continue;
    }

    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      request.SetError("prompt uses " + std::to_string(tokens.size()) +
                       " tokens, context window is " +
                       std::to_string(model.config.max_seq_len) +
                       ". Use /clear or a shorter prompt.");
      request.Finish();
      mini_llama::PrintRequestTrace(request);
      term.PrintMessage("Error: " + request.error + "\n");
      continue;
    }

    int max_response =
        model.config.max_seq_len - static_cast<int>(tokens.size());
    if (max_response > max_response_tokens) {
      max_response = max_response_tokens;
    }

    session.messages = candidate_messages;
    mini_llama::MiniSampler sampler(session.sampling_params);
    mini_llama::Tensor logits;

    auto start = mini_llama::RequestClock::now();
    const size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
    const size_t context_prefix_len =
        mini_llama::CommonPrefixLength(ctx.token_history, tokens);
    size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
    if (prefix_len >= tokens.size()) {
      ctx = mini_llama::MiniLlamaContext(&model);
      prefix_len = 0;
    } else if (prefix_len == 0) {
      ctx = mini_llama::MiniLlamaContext(&model);
    } else if (prefix_len < ctx.token_history.size()) {
      ctx.token_history.resize(prefix_len);
      ctx.pos = static_cast<int>(prefix_len - 1);
    }
    std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
    session.SetTokenHistory(tokens);

    {
      mini_llama::MiniBatch prefill = mini_llama::MiniBatch::FromTokens(
          new_prompt_tokens, static_cast<int>(prefix_len));
      stage_start = mini_llama::RequestClock::now();
      logits = mini_llama::ForwardBatch(ctx, model, prefill);
      request.prefill_ms = mini_llama::ElapsedMs(stage_start);
      request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
      request.RecordEvent(
          "prefill", request.prefill_ms, request.prefill_tokens,
          "radix_hit=" + std::to_string(cached_prefix_len) +
              ", prefix_reuse=" + std::to_string(prefix_len));
      ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
    }

    std::vector<int> generated_ids;
    std::string streamed_reply;
    int generated_count = 0;

    term.PrintAssistantPrefix();
    try {
      for (int i = 0; i < max_response; ++i) {
        stage_start = mini_llama::RequestClock::now();
        const int next_token =
            sampler.Sample(logits, session.sampling_params);
        request.sample_ms += mini_llama::ElapsedMs(stage_start);
        tokens.push_back(next_token);
        session.AppendToken(next_token);
        ++generated_count;

        if (next_token != tokenizer->eos_id()) {
          generated_ids.push_back(next_token);
          const std::string current_reply = tokenizer->Decode(generated_ids);
          if (current_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(current_reply.substr(streamed_reply.size()));
            term.Flush();
            streamed_reply = current_reply;
          }
        }

        mini_llama::MiniBatch decode_batch = mini_llama::MiniBatch::Single(
            next_token, static_cast<int>(tokens.size() - 1));
        stage_start = mini_llama::RequestClock::now();
        logits = mini_llama::ForwardBatch(ctx, model, decode_batch);
        const double decode_ms = mini_llama::ElapsedMs(stage_start);
        request.decode_ms += decode_ms;
        request.RecordEvent("decode", decode_ms, 1,
                            "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;
        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    } catch (const std::exception& e) {
      term.NewLine();
      request.SetError("Inference error: " + std::string(e.what()));
      request.Finish();
      mini_llama::PrintRequestTrace(request);
      term.PrintMessage(request.error + "\n");
      continue;
    }

    const std::string assistant_reply = tokenizer->Decode(generated_ids);
    if (assistant_reply.size() > streamed_reply.size()) {
      term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
    }
    term.NewLine();
    term.NewLine();

    const double elapsed_ms = mini_llama::ElapsedMs(start);
    session.AddMessage("assistant", assistant_reply);
    session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                       generated_count, elapsed_ms);
    request.generated_tokens = generated_count;
    request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                        "generated_tokens");
    session.RecordPrefix(session.token_history);
    request.Finish();
    mini_llama::PrintRequestTrace(request);
  }

  term.PrintMessage("Goodbye.\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc <= 1) {
    PrintUsage(argv[0]);
    return 0;
  }

  std::string command = argv[1];
  if (command == "--help" || command == "-h") {
    PrintUsage(argv[0]);
    return 0;
  }

  if (command == "generate") {
    return RunGenerate(argc, argv);
  }

  if (command == "inspect") {
    return RunInspect(argc, argv);
  }

  if (command == "inspect-gguf") {
    return RunInspectGguf(argc, argv);
  }

  if (command == "run") {
    return RunChat(argc, argv);
  }

  if (command == "bench") {
    return RunBench(argc, argv);
  }

  std::cerr << "unknown command: " << command << "\n";
  PrintUsage(argv[0]);
  return 1;
}
