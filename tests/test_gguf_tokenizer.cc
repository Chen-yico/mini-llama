// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "mini_llama/chat.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/prompt_builder.h"
#include "tests/gguf_tokenizer_fixture.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestGgufTokenizerLoadsFromGguf() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_load.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 14);
  MINI_LLAMA_ASSERT_EQ(tok->bos_id(), 3);
  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 4);
  return true;
}

static bool TestGgufTokenizerEncodeDecodeRoundtrip() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_roundtrip.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);

  std::string text = "ab";
  std::vector<int> ids = tok->Encode(text);
  MINI_LLAMA_ASSERT_EQ(ids.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(ids[0], 2);

  std::string decoded = tok->Decode(ids);
  MINI_LLAMA_ASSERT_TRUE(decoded == text);
  return true;
}

static bool TestGgufTokenizerSpecialTokens() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_special.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);

  std::vector<int> ids = tok->Encode("<|im_start|>");
  MINI_LLAMA_ASSERT_EQ(ids.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(ids[0], 5);

  ids = tok->Encode("<|im_end|>");
  MINI_LLAMA_ASSERT_EQ(ids.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(ids[0], 6);

  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 4);
  MINI_LLAMA_ASSERT_TRUE(tok->DecodeToken(4) == "</s>");
  return true;
}

static bool TestGgufTokenizerDoesNotPrependBos() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_no_bos.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  std::vector<int> ids = tok->Encode("ab");
  MINI_LLAMA_ASSERT_EQ(ids.size(), 1u);
  MINI_LLAMA_ASSERT_TRUE(ids[0] != tok->bos_id());
  return true;
}

static bool TestGgufTokenizerMatchesBpeTokenizer() {
  if (!std::filesystem::exists("models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf") ||
      !std::filesystem::exists("models/chat/vocab.json") ||
      !std::filesystem::exists("models/chat/merges.txt")) {
    return true;
  }

  auto gguf_tok =
      CreateGgufTokenizer("models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf");
  MINI_LLAMA_ASSERT_TRUE(gguf_tok != nullptr);

  auto bpe_tok =
      CreateBpeTokenizer("models/chat/vocab.json", "models/chat/merges.txt",
                         "models/chat/special_tokens.json");
  MINI_LLAMA_ASSERT_TRUE(bpe_tok != nullptr);

  const std::vector<std::string> test_strings = {
      "Hello",
      "Hello world!",
      "你好",
      "<|im_start|>user\nHello<|im_end|>\n",
      "The quick brown fox jumps over the lazy dog.",
  };

  for (const auto& text : test_strings) {
    std::vector<int> gguf_ids = gguf_tok->Encode(text);
    std::vector<int> bpe_ids = bpe_tok->Encode(text);
    if (gguf_ids != bpe_ids) {
      std::cerr << "Mismatch for: " << text << std::endl;
      MINI_LLAMA_ASSERT_FAIL("Encode mismatch between GgufTokenizer and BpeTokenizer");
    }
  }
  return true;
}

static bool TestGgufTokenizerRejectsMissingFile() {
  auto tok = CreateGgufTokenizer("models/chat/nonexistent.gguf");
  MINI_LLAMA_ASSERT_TRUE(tok == nullptr);
  return true;
}

static bool TestGgufTokenizerLoadsRealModelWhenAvailable() {
  const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::exists(path)) {
    return true;
  }

  auto tok = CreateGgufTokenizer(path);
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 151936);
  MINI_LLAMA_ASSERT_EQ(tok->bos_id(), 151643);
  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 151645);
  MINI_LLAMA_ASSERT_EQ(tok->Encode("<|im_start|>")[0], 151644);
  return true;
}

// ---------------------------------------------------------------------------
// Chat template tests
// ---------------------------------------------------------------------------

static bool TestChatTemplateFromGgufQwen2() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_template_qwen2.gguf");
  std::string tmpl = LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(tmpl == mini_llama::test::TinyQwen2Template());
  MINI_LLAMA_ASSERT_TRUE(tmpl.find("{% for message in messages %}") !=
                         std::string::npos);

  PromptBuilder builder;
  builder.SetChatTemplate(tmpl);
  std::string prompt = builder.Build({{"user", "Hello"}});
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "<|im_start|>system\n"
                         "You are a helpful assistant.<|im_end|>\n"
                         "<|im_start|>user\n"
                         "Hello<|im_end|>\n"
                         "<|im_start|>assistant\n");
  return true;
}

static bool TestChatTemplateWithSystemMessage() {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_template_system.gguf");
  std::string tmpl = LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  PromptBuilder builder;
  builder.SetChatTemplate(tmpl);
  std::vector<ChatMessage> messages = {
      {"system", "You are a coding assistant."},
      {"user", "Write a hello world program."}};
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "<|im_start|>system\n"
                         "You are a coding assistant.<|im_end|>\n"
                         "<|im_start|>user\n"
                         "Write a hello world program.<|im_end|>\n"
                         "<|im_start|>assistant\n");
  MINI_LLAMA_ASSERT_TRUE(prompt.find("You are a helpful assistant") ==
                         std::string::npos);
  return true;
}

static bool TestChatTemplateMissingTemplateUsesArchFallbackOnly() {
  std::filesystem::path qwen_path =
      mini_llama::test::WriteTinyGgufTokenizerFixture(
          "tiny_template_qwen2_fallback.gguf", "qwen2", false);
  std::string qwen_tmpl = LoadChatTemplateFromGguf(qwen_path.string());
  std::filesystem::remove(qwen_path);
  MINI_LLAMA_ASSERT_TRUE(qwen_tmpl == "qwen2");

  std::filesystem::path llama_path =
      mini_llama::test::WriteTinyGgufTokenizerFixture(
          "tiny_template_llama_empty.gguf", "llama", false);
  std::string llama_tmpl = LoadChatTemplateFromGguf(llama_path.string());
  std::filesystem::remove(llama_path);
  MINI_LLAMA_ASSERT_TRUE(llama_tmpl.empty());
  return true;
}

static bool TestChatTemplateMalformedTemplateIsSafe() {
  PromptBuilder builder;
  builder.SetChatTemplate("{% endif %}{{ 'ok' }}");
  std::string prompt = builder.Build({});
  MINI_LLAMA_ASSERT_TRUE(prompt == "ok");

  builder.SetChatTemplate("{% if missing %}bad");
  prompt = builder.Build({});
  MINI_LLAMA_ASSERT_TRUE(prompt.empty());

  builder.SetChatTemplate("{% for message in messages %}bad");
  prompt = builder.Build({});
  MINI_LLAMA_ASSERT_TRUE(prompt.empty());

  builder.SetChatTemplate("{% if true %}ok");
  prompt = builder.Build({});
  MINI_LLAMA_ASSERT_TRUE(prompt == "ok");
  return true;
}

static bool TestChatTemplateSupportsJinjaTrimMarkers() {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{%- for message in messages -%}{{- '<|im_start|>' + message['role'] + "
      "'\n' + message['content'] -}}{%- endfor -%}{{- "
      "'<|im_start|>assistant\n' -}}");
  std::string prompt = builder.Build({{"user", "Hello"}});
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "<|im_start|>user\nHello<|im_start|>assistant\n");
  return true;
}

static bool TestChatTemplateBuiltinQwen2Fallback() {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");
  std::string prompt = builder.Build({{"user", "Hi"}});
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "<|im_start|>system\n"
                         "You are a helpful assistant.<|im_end|>\n"
                         "<|im_start|>user\n"
                         "Hi<|im_end|>\n"
                         "<|im_start|>assistant\n");
  return true;
}

static bool TestChatTemplatePlainText() {
  PromptBuilder builder;
  std::string prompt = builder.Build({{"user", "Hello"}});
  MINI_LLAMA_ASSERT_TRUE(prompt == "User: Hello\nAssistant:");
  return true;
}

static bool TestChatTemplateMissingGgufIsSafe() {
  std::string tmpl = LoadChatTemplateFromGguf("models/chat/nonexistent.gguf");
  MINI_LLAMA_ASSERT_TRUE(tmpl.empty());
  return true;
}

static bool TestChatTemplateOversizedFallsBackToUsablePrompt() {
  PromptBuilder builder;
  std::string oversized = "{% if add_generation_prompt %}{{ 'x' }}{% endif %}";
  oversized.append((1u << 20) + 1, ' ');
  builder.SetChatTemplate(oversized);
  std::string prompt = builder.Build({{"user", "Hello"}});
  MINI_LLAMA_ASSERT_TRUE(prompt == "User: Hello\nAssistant:");

  std::string oversized_qwen =
      "{% for message in messages %}{{ message['content'] }}{% endfor %}"
      "<|im_start|>";
  oversized_qwen.append((1u << 20) + 1, 'a');
  builder.SetChatTemplate(oversized_qwen);
  prompt = builder.Build({{"user", "Hello"}});
  MINI_LLAMA_ASSERT_TRUE(!prompt.empty());
  MINI_LLAMA_ASSERT_TRUE(prompt.find("Hello") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(prompt.find("<|im_start|>assistant") !=
                         std::string::npos);
  return true;
}

static bool TestChatTemplateNestedForCompletes() {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}{% for message in messages %}"
      "{{ message['content'] }}{% endfor %}{% endfor %}{{ 'END' }}");
  std::vector<ChatMessage> messages = {{"user", "a"}, {"user", "b"},
                                       {"user", "c"}, {"user", "d"},
                                       {"user", "e"}};
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt == "abcdeabcdeabcdeabcdeabcdeEND");
  return true;
}

static bool TestChatTemplateLeadingWhitespaceStillJinja() {
  PromptBuilder builder;
  builder.SetChatTemplate("\n{% if add_generation_prompt %}{{ 'ok' }}{% endif %}");
  std::string prompt = builder.Build({});
  MINI_LLAMA_ASSERT_TRUE(prompt == "\nok");
  return true;
}

static struct GgufTokenizerTestRegistrar {
  GgufTokenizerTestRegistrar() {
    RegisterTest("gguf_tokenizer_loads_from_gguf",
                 TestGgufTokenizerLoadsFromGguf);
    RegisterTest("gguf_tokenizer_encode_decode_roundtrip",
                 TestGgufTokenizerEncodeDecodeRoundtrip);
    RegisterTest("gguf_tokenizer_special_tokens",
                 TestGgufTokenizerSpecialTokens);
    RegisterTest("gguf_tokenizer_does_not_prepend_bos",
                 TestGgufTokenizerDoesNotPrependBos);
    RegisterTest("gguf_tokenizer_rejects_missing_file",
                 TestGgufTokenizerRejectsMissingFile);
    RegisterTest("gguf_tokenizer_loads_real_model_when_available",
                 TestGgufTokenizerLoadsRealModelWhenAvailable);
    RegisterTest("gguf_tokenizer_matches_bpe_tokenizer",
                 TestGgufTokenizerMatchesBpeTokenizer);
    RegisterTest("chat_template_from_gguf_qwen2",
                 TestChatTemplateFromGgufQwen2);
    RegisterTest("chat_template_with_system_message",
                 TestChatTemplateWithSystemMessage);
    RegisterTest("chat_template_missing_template_uses_arch_fallback_only",
                 TestChatTemplateMissingTemplateUsesArchFallbackOnly);
    RegisterTest("chat_template_malformed_template_is_safe",
                 TestChatTemplateMalformedTemplateIsSafe);
    RegisterTest("chat_template_supports_jinja_trim_markers",
                 TestChatTemplateSupportsJinjaTrimMarkers);
    RegisterTest("chat_template_builtin_qwen2_fallback",
                 TestChatTemplateBuiltinQwen2Fallback);
    RegisterTest("chat_template_plain_text", TestChatTemplatePlainText);
    RegisterTest("chat_template_missing_gguf_is_safe",
                 TestChatTemplateMissingGgufIsSafe);
    RegisterTest("chat_template_oversized_falls_back_to_usable_prompt",
                 TestChatTemplateOversizedFallsBackToUsablePrompt);
    RegisterTest("chat_template_nested_for_completes",
                 TestChatTemplateNestedForCompletes);
    RegisterTest("chat_template_leading_whitespace_still_jinja",
                 TestChatTemplateLeadingWhitespaceStillJinja);
  }
} gguf_tokenizer_test_registrar;
