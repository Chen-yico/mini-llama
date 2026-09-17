// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/prompt_builder.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stack>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"

namespace mini_llama {

// ===========================================================================
// Minimal Jinja2 template engine for chat templates
// ===========================================================================

namespace {

constexpr size_t kMaxChatTemplateBytes = 1u << 20;
constexpr size_t kMaxPromptBytes = 16u << 20;

struct EvalContext {
  const std::vector<ChatMessage>* messages = nullptr;
  const ChatMessage* current_message = nullptr;
  size_t message_index = 0;
  bool add_generation_prompt = false;
};

// Trim whitespace from both ends
static std::string Trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
    --b;
  }
  return s.substr(a, b - a);
}

static std::string StripJinjaTrimMarkers(const std::string& raw) {
  std::string value = Trim(raw);
  if (!value.empty() && value.front() == '-') {
    value = Trim(value.substr(1));
  }
  if (!value.empty() && value.back() == '-') {
    value.pop_back();
    value = Trim(value);
  }
  return value;
}

// Evaluate a Jinja2 expression (string literal, variable, FlatIndex access, +)
static std::string EvalExpr(const std::string& expr_raw,
                            const EvalContext& ctx) {
  std::string expr = Trim(expr_raw);
  if (expr.empty()) {
    return "";
  }

  // Handle string concatenation with +
  std::vector<std::string> parts;
  size_t pos = 0;
  while (pos < expr.size()) {
    // Skip whitespace
    while (pos < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[pos]))) {
      ++pos;
    }
    if (pos >= expr.size()) {
      break;
    }

    // String literal ('...' or "...")
    if (expr[pos] == '\'' || expr[pos] == '"') {
      const char quote = expr[pos];
      ++pos;
      std::string lit;
      while (pos < expr.size() && expr[pos] != quote) {
        lit += expr[pos];
        ++pos;
      }
      if (pos < expr.size() && expr[pos] == quote) {
        ++pos;
      }
      parts.push_back(lit);
    } else {
      // Read a token (variable name, number, or access chain)
      size_t start = pos;
      while (pos < expr.size() && expr[pos] != '+' && expr[pos] != '\'') {
        if (expr[pos] == '[') {
          // Consume [...]
          ++pos;
          while (pos < expr.size() && expr[pos] != ']') {
            ++pos;
          }
          if (pos < expr.size()) {
            ++pos;
          }
        } else if (expr[pos] == '.') {
          // Consume .xxx
          ++pos;
          while (pos < expr.size() &&
                 !std::isspace(static_cast<unsigned char>(expr[pos])) &&
                 expr[pos] != '+' && expr[pos] != '\'' && expr[pos] != '[' &&
                 expr[pos] != '.') {
            ++pos;
          }
        } else {
          ++pos;
        }
      }
      std::string token = Trim(expr.substr(start, pos - start));
      if (!token.empty()) {
        // Evaluate variable/token
        if (token == "add_generation_prompt") {
          parts.push_back(ctx.add_generation_prompt ? "true" : "false");
        } else if (token == "message['role']" || token == "message[\"role\"]") {
          parts.push_back(ctx.current_message ? ctx.current_message->role : "");
        } else if (token == "message['content']" ||
                   token == "message[\"content\"]") {
          parts.push_back(ctx.current_message ? ctx.current_message->content
                                              : "");
        } else if (token.find("messages[") == 0) {
          // messages[0]['role'] or similar
          size_t close = token.find(']');
          if (close != std::string::npos && close + 1 < token.size() &&
              token[close + 1] == '[') {
            std::string idx_str = Trim(token.substr(9, close - 9));
            std::string key;
            size_t key_start = close + 2;
            size_t key_end = token.find(']', key_start);
            if (key_end != std::string::npos) {
              key = token.substr(key_start, key_end - key_start);
              if (key.size() >= 2 &&
                  ((key.front() == '\'' && key.back() == '\'') ||
                   (key.front() == '"' && key.back() == '"'))) {
                key = key.substr(1, key.size() - 2);
              }
            }
            try {
              int idx = std::stoi(idx_str);
              if (ctx.messages && idx >= 0 &&
                  idx < static_cast<int>(ctx.messages->size())) {
                const ChatMessage& msg = (*ctx.messages)[idx];
                if (key == "role") {
                  parts.push_back(msg.role);
                } else if (key == "content") {
                  parts.push_back(msg.content);
                } else {
                  parts.push_back("");
                }
              } else {
                parts.push_back("");
              }
            } catch (...) {
              parts.push_back("");
            }
          } else {
            parts.push_back("");
          }
        } else if (token == "loop.first") {
          parts.push_back(ctx.message_index == 0 ? "true" : "false");
        } else if (token == "true" || token == "True") {
          parts.push_back("true");
        } else if (token == "false" || token == "False") {
          parts.push_back("false");
        } else {
          parts.push_back("");
        }
      }
    }

    // Skip + operator
    while (pos < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[pos]))) {
      ++pos;
    }
    if (pos < expr.size() && expr[pos] == '+') {
      ++pos;
    }
  }

  // Concatenate all parts
  std::string result;
  for (const auto& p : parts) {
    result += p;
  }
  return result;
}

// Evaluate a Jinja2 condition
static bool EvalCondition(const std::string& cond_raw, const EvalContext& ctx) {
  std::string cond = Trim(cond_raw);
  if (cond.empty()) {
    return false;
  }

  // Handle 'and'
  size_t and_pos = cond.find(" and ");
  while (and_pos != std::string::npos) {
    std::string left = Trim(cond.substr(0, and_pos));
    std::string right = Trim(cond.substr(and_pos + 5));
    return EvalCondition(left, ctx) && EvalCondition(right, ctx);
  }

  // Handle 'or'
  size_t or_pos = cond.find(" or ");
  while (or_pos != std::string::npos) {
    std::string left = Trim(cond.substr(0, or_pos));
    std::string right = Trim(cond.substr(or_pos + 4));
    return EvalCondition(left, ctx) || EvalCondition(right, ctx);
  }

  // Handle == and !=
  size_t neq_pos = cond.find(" != ");
  if (neq_pos != std::string::npos) {
    std::string left = EvalExpr(cond.substr(0, neq_pos), ctx);
    std::string right = EvalExpr(cond.substr(neq_pos + 4), ctx);
    return left != right;
  }
  size_t eq_pos = cond.find(" == ");
  if (eq_pos != std::string::npos) {
    std::string left = EvalExpr(cond.substr(0, eq_pos), ctx);
    std::string right = EvalExpr(cond.substr(eq_pos + 4), ctx);
    return left == right;
  }

  // Single variable
  std::string val = EvalExpr(cond, ctx);
  if (val == "true" || val == "True") {
    return true;
  }
  if (val == "false" || val == "False") {
    return false;
  }
  return !val.empty();
}

// Template instruction
struct Instr {
  enum Type { kText, kOutput, kIf, kEndIf, kFor, kEndFor } type;
  std::string text;  // TEXT content, OUTPUT/IF/FOR expression
  size_t jump = 0;   // jump target for IF and ENDFOR
};

// For-loop frame
struct ForFrame {
  size_t pc = 0;     // FOR instruction FlatIndex
  size_t index = 0;  // current iteration FlatIndex
};

// Compile a Jinja2 template string into an instruction sequence
static std::vector<Instr> CompileTemplate(const std::string& tmpl) {
  std::vector<Instr> instrs;
  std::stack<size_t> if_stack;
  std::stack<size_t> for_stack;

  size_t pos = 0;
  while (pos < tmpl.size()) {
    if (pos + 1 < tmpl.size() && tmpl[pos] == '{' && tmpl[pos + 1] == '%') {
      // Jinja2 statement
      pos += 2;
      size_t end = tmpl.find("%}", pos);
      if (end == std::string::npos) {
        break;
      }
      std::string stmt = StripJinjaTrimMarkers(tmpl.substr(pos, end - pos));
      pos = end + 2;

      if (stmt.rfind("for ", 0) == 0) {
        // {% for message in messages %}
        instrs.push_back({Instr::kFor, stmt});
        for_stack.push(instrs.size() - 1);
      } else if (stmt == "endfor") {
        if (for_stack.empty()) {
          continue;
        }
        size_t for_pc = for_stack.top();
        for_stack.pop();
        instrs.push_back({Instr::kEndFor, "", for_pc});
        instrs[for_pc].jump =
            instrs.size();  // FOR jumps to after ENDFOR when done
      } else if (stmt.rfind("if ", 0) == 0) {
        // {% if cond %}
        instrs.push_back({Instr::kIf, stmt.substr(3)});
        if_stack.push(instrs.size() - 1);
      } else if (stmt == "endif") {
        if (if_stack.empty()) {
          continue;
        }
        size_t if_pc = if_stack.top();
        if_stack.pop();
        instrs.push_back({Instr::kEndIf});
        instrs[if_pc].jump =
            instrs.size();  // IF jumps to after ENDIF when false
      }
    } else if (pos + 1 < tmpl.size() && tmpl[pos] == '{' &&
               tmpl[pos + 1] == '{') {
      // Jinja2 expression
      pos += 2;
      size_t end = tmpl.find("}}", pos);
      if (end == std::string::npos) {
        break;
      }
      std::string expr = StripJinjaTrimMarkers(tmpl.substr(pos, end - pos));
      pos = end + 2;
      instrs.push_back({Instr::kOutput, expr});
    } else {
      // Plain text
      size_t end = pos;
      while (end < tmpl.size() &&
             !(tmpl[end] == '{' && end + 1 < tmpl.size() &&
               (tmpl[end + 1] == '%' || tmpl[end + 1] == '{'))) {
        ++end;
      }
      std::string text = tmpl.substr(pos, end - pos);
      pos = end;
      if (!text.empty()) {
        instrs.push_back({Instr::kText, text});
      }
    }
  }

  while (!if_stack.empty()) {
    instrs[if_stack.top()].jump = instrs.size();
    if_stack.pop();
  }

  while (!for_stack.empty()) {
    instrs[for_stack.top()].jump = instrs.size();
    for_stack.pop();
  }

  return instrs;
}

static size_t MaxForNesting(const std::vector<Instr>& instrs) {
  size_t depth = 0;
  size_t max_depth = 0;
  for (const auto& instr : instrs) {
    if (instr.type == Instr::kFor) {
      ++depth;
      max_depth = std::max(max_depth, depth);
    } else if (instr.type == Instr::kEndFor && depth > 0) {
      --depth;
    }
  }
  return max_depth;
}

// Nested {% for %} both walk messages, so work is O(instrs * messages^depth).
static size_t MaxExecuteSteps(const std::vector<Instr>& instrs,
                              size_t message_count) {
  const size_t factor = message_count + 2;
  const size_t depth = std::max<size_t>(MaxForNesting(instrs), 1);
  size_t max_steps = instrs.size() + 1;
  for (size_t i = 0; i < depth; ++i) {
    if (factor != 0 &&
        max_steps > std::numeric_limits<size_t>::max() / factor) {
      return std::numeric_limits<size_t>::max();
    }
    max_steps *= factor;
  }
  constexpr size_t kSlack = 256;
  if (max_steps > std::numeric_limits<size_t>::max() - kSlack) {
    return std::numeric_limits<size_t>::max();
  }
  return max_steps + kSlack;
}

// Execute compiled template instructions.
static std::string ExecuteTemplate(const std::vector<Instr>& instrs,
                                   const std::vector<ChatMessage>& messages,
                                   bool add_gen) {
  std::string output;
  EvalContext ctx;
  ctx.messages = &messages;
  ctx.add_generation_prompt = add_gen;

  std::stack<ForFrame> for_stack;
  size_t pc = 0;
  const size_t max_steps = MaxExecuteSteps(instrs, messages.size());
  size_t steps = 0;

  auto jump_to = [&](size_t dest) {
    pc = dest > instrs.size() ? instrs.size() : dest;
  };

  auto append = [&](const std::string& text) {
    if (output.size() >= kMaxPromptBytes) {
      return false;
    }
    const size_t room = kMaxPromptBytes - output.size();
    if (text.size() > room) {
      output.append(text, 0, room);
      return false;
    }
    output += text;
    return true;
  };

  while (pc < instrs.size()) {
    if (++steps > max_steps) {
      break;
    }
    const Instr& instr = instrs[pc];
    switch (instr.type) {
      case Instr::kText:
        if (!append(instr.text)) {
          return output;
        }
        ++pc;
        break;
      case Instr::kOutput:
        if (!append(EvalExpr(instr.text, ctx))) {
          return output;
        }
        ++pc;
        break;
      case Instr::kIf:
        if (!EvalCondition(instr.text, ctx)) {
          jump_to(instr.jump);  // skip to after ENDIF
        } else {
          ++pc;
        }
        break;
      case Instr::kEndIf:
        ++pc;
        break;
      case Instr::kFor: {
        // Check if we're entering or continuing a loop
        if (for_stack.empty() || for_stack.top().pc != pc) {
          // New loop iteration
          ForFrame frame;
          frame.pc = pc;
          frame.index = 0;
          for_stack.push(frame);
          if (messages.empty()) {
            for_stack.pop();
            jump_to(instr.jump);  // skip loop body
          } else {
            ctx.current_message = &messages[0];
            ctx.message_index = 0;
            ++pc;
          }
        } else {
          // Continue to next iteration
          ForFrame& frame = for_stack.top();
          ++frame.index;
          if (frame.index >= messages.size()) {
            for_stack.pop();
            jump_to(instr.jump);  // exit loop
          } else {
            ctx.current_message = &messages[frame.index];
            ctx.message_index = frame.index;
            ++pc;
          }
        }
        break;
      }
      case Instr::kEndFor:
        jump_to(instr.jump);  // jump back to FOR
        break;
    }
  }

  return output;
}

}  // anonymous namespace

// ===========================================================================
// PromptBuilder
// ===========================================================================

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
  chat_template_ = template_str;
}

std::string PromptBuilder::Build(
    const std::vector<ChatMessage>& messages) const {
  if (chat_template_.empty()) {
    return BuildPlain(messages);
  }

  // Jinja2 templates start with {% or {{, optionally after leading whitespace.
  size_t start = 0;
  while (start < chat_template_.size() &&
         std::isspace(static_cast<unsigned char>(chat_template_[start]))) {
    ++start;
  }
  if (start + 1 < chat_template_.size() && chat_template_[start] == '{' &&
      (chat_template_[start + 1] == '%' || chat_template_[start + 1] == '{')) {
    // Oversized Jinja is not compiled into an empty program: that would yield
    // an empty prompt and ForwardBatch would throw on zero tokens.
    if (chat_template_.size() > kMaxChatTemplateBytes) {
      if (chat_template_.find("<|im_start|>") != std::string::npos) {
        return BuildQwen2(messages);
      }
      return BuildPlain(messages);
    }
    auto instrs = CompileTemplate(chat_template_);
    return ExecuteTemplate(instrs, messages, true);
  }

  if (chat_template_ == "qwen2") {
    return BuildQwen2(messages);
  }

  return BuildPlain(messages);
}

std::string PromptBuilder::BuildPlain(
    const std::vector<ChatMessage>& messages) const {
  std::string prompt;
  for (const auto& msg : messages) {
    if (msg.role == "system") {
      prompt += "System: " + msg.content + "\n";
    } else if (msg.role == "user") {
      prompt += "User: " + msg.content + "\n";
    } else if (msg.role == "assistant") {
      prompt += "Assistant: " + msg.content + "\n";
    }
  }
  prompt += "Assistant:";
  return prompt;
}

// Qwen2 chat template (built-in fallback):
//   If no system message, inject default system message.
//   Each message: <|im_start|>{role}\n{content}<|im_end|>\n
//   Final assistant prompt: <|im_start|>assistant\n
std::string PromptBuilder::BuildQwen2(
    const std::vector<ChatMessage>& messages) const {
  std::string prompt;

  bool has_system = false;
  for (const auto& msg : messages) {
    if (msg.role == "system") {
      has_system = true;
      break;
    }
  }

  if (!has_system) {
    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  }

  for (const auto& msg : messages) {
    if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
      continue;
    }
    prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
  }

  prompt += "<|im_start|>assistant\n";
  return prompt;
}

// ===========================================================================
// Load chat template from GGUF
// ===========================================================================

std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  std::string tmpl;
  if (!GgufGetMetadataString(gguf_path, "tokenizer.chat_template", tmpl) ||
      tmpl.empty()) {
    std::string architecture;
    if (GgufGetMetadataString(gguf_path, "general.architecture",
                              architecture) &&
        architecture == "qwen2") {
      return "qwen2";
    }
    return "";
  }

  return tmpl;
}

}  // namespace mini_llama
