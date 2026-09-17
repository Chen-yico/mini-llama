# mini-llama

从零实现的 LLaMA 风格 C++ 推理引擎，按课程步骤逐步搭建。

当前进度：**step-08-gguf-reader** — GGUF v3 Reader 与 `inspect-gguf`。

## 环境要求

- CMake >= 3.17
- 支持 C++17 的编译器（MSVC / GCC / Clang）

## 编译

```bash
cmake -S . -B build
cmake --build build -j4
```

Windows 可执行文件：

- CLI：`build/Debug/mini-llama.exe`
- 测试：`build/Debug/mini-llama-tests.exe`

## 使用

```bash
./build/Debug/mini-llama.exe --help
./build/Debug/mini-llama.exe generate -p "hello mini llama" -n 8
./build/Debug/mini-llama.exe generate -p "hello" -n 8 --temperature 0.8 --top-k 20 --seed 42
./build/Debug/mini-llama.exe inspect models/tiny
./build/Debug/mini-llama.exe inspect-gguf models/tiny/test.gguf
./build/Debug/mini-llama.exe inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
./build/Debug/mini-llama.exe run models/tiny -n 8
./build/Debug/mini-llama.exe run models/chat -n 8
```

`generate` 加载 `models/tiny/model.json` + `model.bin`。`inspect` 先打印清单，再完整读入权重。JSON+BIN 当前 dtype 固定为 float32。

内置命令：`/help` `/clear` `/stats` `/params` `/exit`。`/clear` 会重建 `MiniLlamaContext`，避免 KV Cache 的 `pos` 与文本长度错位。

`run` 仍用 `MakeCpuTestModel()` 合成权重打通多轮对话；GGUF 权重加载尚未接入。`models/chat` 自带 Qwen2-0.5B GGUF、词表、merges 与 chat template，`run models/chat` 只读本仓库文件。

## 运行测试

```bash
./build/Debug/mini-llama-tests.exe
# 或
ctest --test-dir build -C Debug --output-on-failure
```

## 目录结构

```
mini-llama/
├── include/mini_llama/
│   ├── tensor.h / ops.h / matmul_dispatch.h / thread_pool.h
│   ├── tokenizer.h         ITokenizer + Ascii/Json/BPE
│   ├── gguf.h              GGUF v3 Reader / metadata / tensor index
│   ├── gguf_tokenizer.h    从 GGUF metadata 加载
│   ├── kv_cache.h          4D KV Cache
│   ├── model.h             ModelConfig + LayerWeights
│   ├── loader.h            JSON+BIN ParseManifest / LoadModel / inspect
│   ├── context.h           MiniLlamaContext 会话状态
│   ├── batch.h             Prefill/Decode 统一 MiniBatch
│   ├── forward.h           ForwardToken / ForwardBatch
│   ├── sampler.h           SamplingParams + MiniSampler
│   ├── radix_tree.h        压缩前缀树
│   ├── request_context.h   请求级 trace
│   ├── chat.h              ChatSession + prefix_cache
│   ├── prompt_builder.h    Plain / Qwen2 / 轻量 Jinja2
│   └── terminal.h          交互式 I/O 与内置命令
├── models/tiny/
│   ├── model.json          配置 + tensor 索引
│   ├── model.bin           无头 F32 权重
│   ├── test.gguf           最小 GGUF v3 测试夹具
│   └── vocab.json          tiny 模型词表
├── models/chat/
│   ├── Qwen2-0.5B-Instruct-Q8_0.gguf
│   ├── chat_template.txt
│   ├── vocab.json / merges.txt / special_tokens.json
├── src/                    对应实现
└── tests/
    ├── test_tensor.cc / test_ops.cc
    ├── test_tokenizer.cc
    ├── test_kv_cache.cc
    ├── test_batch.cc
    ├── test_forward.cc
    ├── test_sampler.cc
    ├── test_radix_tree.cc
    ├── test_request_context.cc
    ├── test_chat.cc
    ├── test_loader.cc
    └── test_gguf.cc
```

## Forward 主链路

`ForwardToken` 是最小计算单元：整数 token id → Embedding 查表 → N 层 `ForwardLayer` → 最终 RMSNorm + `lm_head` → `logits[vocab_size]`。

`ForwardBatch` 只做调度：校验 batch，按 position 顺序调用 `ForwardToken`，更新 `ctx.pos` / `token_history`。Prefill（多 token）和 Decode（单 token）共用这一条接口。

每层顺序：

1. Attention：Pre-Norm → QKV Linear → reshape 多头 → RoPE(Q,K) → 先写 KV Cache 再算 Attention → `wo` → 残差
2. FFN：Pre-Norm → Gate/Up → SwiGLU → Down → 残差

GQA：`kv_head = q_head / (n_heads / n_kv_heads)`。Attention 点积后除以 `sqrt(head_dim)`，多头之间用 `ParallelFor` 无锁并行。

本 step 只走 **CPU F32**。量化 Linear 与 CUDA device-resident 路径未接入；主链路函数签名与参考实现一致，后续可在 `ForwardLinear` 处分发。

## Sampler

`MiniSampler::Sample` 按 `SamplingParams` 做扁平分支，不走 llama.cpp 那种 sampler chain：

| temperature | top_k | 策略 |
|-------------|-------|------|
| 0 或极小 | 任意 | Greedy（`ArgMax`） |
| > 0 | 0 | Temperature softmax + 累积采样 |
| > 0 | 1 | Greedy |
| > 0 | > 1 | 先截断 Top-k，再温度采样 |

`seed != 0` 固定 `std::mt19937` 序列；`seed == 0` 用 `std::random_device`。同分时按 token id 升序，保证跨平台可复现。

## Tokenizer

| 实现 | 说明 |
|------|------|
| `AsciiTokenizer` | 字符级 ASCII，单测无外部依赖 |
| `JsonVocabTokenizer` | JSON 词表 + 最长前缀匹配 |
| `BpeTokenizer` | GPT-2 风格 BPE（vocab + merges + special） |
| `GgufTokenizer` | 从 GGUF metadata 读取词表与 merges |
| `CreateTokenizer()` | 有 vocab.json 则 JSON，否则 ASCII 回退 |

## GGUF Reader

`GgufReader::Load` 顺序解析 Header → Metadata KV → Tensor Info → 按 `general.alignment`（默认 32 字节）对齐后得到 `data_offset`。只读前半段索引，不把权重整文件映射进内存。支持 GGUF v3、小端、13 种标量类型及 string/array；tensor dtype 可识别 F32 / F16 / Q4_0 / Q4_1 / Q8_0 并计算字节数。K-Quants、IQ、大端、分片 GGUF 与 `LoadGgufModel` 权重装填是后续步骤。`inspect-gguf` 打印 metadata 与 tensor 索引。

## JSON+BIN Loader

`model.json` 分三块：config、tokenizer、tensors 索引。`model.bin` 无文件头，按 offset 跳读 F32。`LoadModel` 五道校验：必填字段、维度整除/RoPE 偶 head_dim、tensor 五元组、byte_size 与区间不重叠、bin 长度与尾字节。JSON+BIN 的 `rope_type` 用 `kNormal`，QKV bias 保持空 Tensor。量化容器与 GGUF 权重加载是后续步骤。

## KV Cache 与 Context

| 组件 | 说明 |
|------|------|
| `KvCache` | `[n_layers, max_seq_len, n_kv_heads, head_dim]`，预分配；`Write` + 零拷贝 `KeyPtr/ValuePtr` |
| `MiniLlamaContext` | 持有 cache、`pos`、`token_history`、prefill/decode 计数 |
| `MiniBatch` | Prefill 多 token / Decode 单 token，position 从 `start_pos` 递增 |
| `RadixTree` | 记录已算过的 token 前缀，最长前缀命中；线性 KV 仍以 `ctx.token_history` 为准 |
| `RequestContext` | `radix_hit` / `prefix_reuse` 等 trace 字段的载体 |

## 已实现算子

| 算子 | 说明 |
|------|------|
| `Matmul` / `Linear` | 矩阵乘与全连接（含 naive/threaded/SIMD 分发） |
| `RmsNorm` | RMS 归一化 |
| `Softmax` | 数值稳定 softmax（先减 max） |
| `Silu` / `SwiGlu` | 激活函数 |
| `Rope` | 旋转位置编码（Normal / NeoX 两种风格） |
| `ArgMax` | 贪心采样索引 |

## 开发里程碑

| Tag | 说明 |
|-----|------|
| `step-01-cmake-cli` | 最小 CMake 工程 + CLI 骨架 |
| `step-02-tensor-ops` | Tensor + 基础算子 + 测试 |
| `step-03-tokenizer` | Tokenizer 编解码 + 测试 |
| `step-04-kv-cache-context` | KV Cache + Context + RadixTree |
| `step-05-forward` | CPU Forward 主链路 + 合成权重测试 |
| `step-06-sampler` | Greedy / Temperature / Top-k 采样 |
| `step-07-chat` | ChatSession + PromptBuilder + Terminal + `run` |
| `step-07-loader` | JSON+BIN Loader（五道校验 + inspect） |
| `step-08-gguf-reader` | GGUF v3 Reader + `inspect-gguf` |

## Chat 与 PromptBuilder

`ChatSession` 同时保存两层状态：

| 字段 | 作用 |
|------|------|
| `messages` | 带 role 的语义历史，交给 `PromptBuilder` 拼 prompt |
| `token_history` | 已进入 KV Cache 的 token 序列，用来算最长公共前缀 |
| `prefix_cache` | RadixTree，记录已算过的 prompt 前缀 |
| `sampling_params` | 当前会话的 temperature / top_k / seed |

`PromptBuilder` 按 `chat_template_` 分发：

1. 空模板 → Plain：`System:` / `User:` / `Assistant:`，末尾补 `Assistant:`
2. `"qwen2"` → ChatML，无 system 时补默认人设，结尾 `<|im_start|>assistant\n`
3. `{% ... %}` Jinja2 子集：`for` / `if`、`loop.first`、`add_generation_prompt`、字符串拼接与 trim 标记

多轮性能关键是最长公共前缀：`prefix_len = min(radix_hit, CommonPrefixLength(ctx.token_history, tokens))`。前缀命中时只 Prefill 后缀，KV Cache 从 `prefix_len` 继续写；不命中或 `/clear` 时重建 `MiniLlamaContext`。

## 参考

- 参考实现：`llama-cpp-test`
- 架构说明：`项目相关需求整合/mini-llama-cpp/`
