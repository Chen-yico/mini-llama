# mini-llama

从零实现的 LLaMA 风格 C++ 推理引擎，按课程步骤逐步搭建。

当前进度：**真实模型端到端 Smoke Test** — 用真实 Qwen2 GGUF 跑通 `inspect-gguf`、`generate` 和 `run`，确认整条链路能加载、能分词、能前向、能采样、能退出。

前面几章已经把真实模型路径上的关键模块接起来了：

- GGUF Reader 负责读取 metadata 和 tensor。
- GGUF Loader 负责把 Qwen2 权重映射到 `MiniLlamaModel`。
- `GgufTokenizer` 负责从 GGUF metadata 读取 vocab、merges 和特殊 token。
- `PromptBuilder` 负责把对话消息拼成 Qwen2 ChatML prompt。
- Forward、KV Cache 和 Sampler 负责完成自回归生成。

Smoke test 只判断链路是否稳定可用。短输出的内容质量不代表模型完整能力，尤其是 `-n 1` 或 `-n 3` 这种极短生成。本仓库是 CPU 路径，不包含 CUDA / `--quant`。

## 环境要求

- CMake >= 3.17
- 支持 C++17 的编译器（MSVC / GCC / Clang）
- Python 3（下载 demo 模型、可选 CLI smoke）

## 编译

```bash
cmake -S . -B build
cmake --build build -j4
```

Windows 可执行文件：

- CLI：`build/Debug/mini-llama.exe`
- 测试：`build/Debug/mini-llama-tests.exe`

下文 Unix 示例用 `./build/mini-llama`；Windows 换成 `.\build\Debug\mini-llama.exe`，`ctest` 加 `-C Debug`。

## 准备模型文件

项目提供了 ModelScope 下载脚本：

```bash
python scripts/download_demo_model.py --dir models/chat
```

默认下载目标是：

```text
ModelScope repo: qwen/Qwen2-0.5B-Instruct-GGUF
remote file:     qwen2-0_5b-instruct-q8_0.gguf
local file:      models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

脚本内部的下载地址由 `get_modelscope_download_url()` 拼出来：

```text
https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-GGUF/resolve/master/qwen2-0_5b-instruct-q8_0.gguf
```

下载完成后，确认文件存在：

```bash
ls -lh models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

如果前面已经执行过 GGUF Tokenizer 接入的导出脚本，`models/chat/` 里可能还有这些文件：

```text
models/chat/
├── Qwen2-0.5B-Instruct-Q8_0.gguf
├── vocab.json
├── merges.txt
├── special_tokens.json
└── chat_template.txt
```

当前默认 GGUF 路径会直接使用 `GgufTokenizer`，所以 `generate --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf` 不强制依赖这些外部 tokenizer 文件。它们主要用于对照、调试和显式 `--tokenizer models/chat/vocab.json` 场景。大文件 `.gguf` 不进 git，需本地下载。

## 使用

```bash
./build/mini-llama --help
./build/mini-llama generate -p "hello mini llama" -n 8
./build/mini-llama generate -p "hello" -n 8 --temperature 0.8 --top-k 20 --seed 42
./build/mini-llama inspect models/tiny
./build/mini-llama inspect-gguf models/tiny/test.gguf
./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
./build/mini-llama run models/tiny -n 8
./build/mini-llama run models/chat -n 8
```

`generate` 默认加载 `models/tiny` 的 JSON+BIN。`inspect` 先打印清单，再完整读入权重。JSON+BIN 当前 dtype 固定为 float32。

内置命令：`/help` `/clear` `/stats` `/params` `/exit`。`/clear` 会重建 `MiniLlamaContext`，避免 KV Cache 的 `pos` 与文本长度错位。

`run models/chat` 会自动解析目录里的 GGUF；没有 GGUF 时，`run models/tiny` 仍用合成权重打通多轮对话。

## 真实模型端到端 Smoke Test

### 第一步：inspect-gguf 确认文件能解析

```bash
./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

这一步主要确认三件事：

- 文件头是合法 GGUF，版本能被 reader 接受。
- Qwen2 的核心配置能读出来，包括层数、维度、head 数和 RoPE 参数。
- tokenizer metadata 存在，包括 tokens、token_type、merges、BOS/EOS 和 chat_template。

### 第二步：generate 跑单次生成

`generate` 是最适合脚本化 smoke 的入口。它不需要交互输入，输出稳定，失败时也容易定位。

```bash
./build/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  --prompt "The capital of France is" \
  --n-predict 3 \
  --seed 42
```

重点看这些信号：

- `GGUF model loaded successfully`：权重加载成功。
- `GGUF tokenizer loaded`：tokenizer metadata 加载成功。
- `tokens: [...]`：prompt 能被编码。
- `generated tokens: [...]`：Forward 和 Sampler 能产出 token。
- `trace summary ... status=ok`：请求正常结束。

短 smoke 不要求生成内容好看。这里的 `generated text` 只是 greedy 采样下的极短续写结果。

### 第三步：run 跑交互模式

`run` 会走对话路径，覆盖 Chat Template、会话状态、终端输入输出和 `/stats`。

```bash
./build/mini-llama run models/chat -n 10 --seed 42
```

输入一条消息，再输入 `/stats`，最后 `/exit`。smoke 重点是：

- 模型目录 `models/chat` 能被自动解析到 GGUF 文件。
- 用户输入会被 `PromptBuilder` 包装成 chat prompt。
- tokenizer 能编码完整 chat prompt。
- 生成过程能流式打印 assistant 回复。
- `/stats` 能显示消息数、上下文 token 数、累计生成数和吞吐。
- `/exit` 能正常退出。

本地 CPU 上 `run` 的首轮 prefill 可能较慢。为了快速验证链路，可以先用 `-n 1`。

## 运行测试

```bash
./build/mini-llama-tests
# 或
ctest --test-dir build -R mini-llama-tests --output-on-failure
```

Windows：

```bash
.\build\Debug\mini-llama-tests.exe
ctest --test-dir build -C Debug -R mini-llama-tests --output-on-failure
```

真实 GGUF 未下载时，C++ 测试会跳过 `gguf_*_loads_real_model_when_available`。tiny 模型上的 CLI smoke：

```bash
python scripts/test_real_model_smoke.py
```

已下载 Qwen2 GGUF 时，该脚本还会跑真实模型的 `inspect-gguf` / `generate -n 1` / `run /exit`。ctest 里的 `cli-smoke` 默认跳过真实模型，避免 CPU prefill 拖慢常规测试。

## 常见问题排查

### 模型文件打不开

典型输出：

```text
Failed to load model: GGUF parse error: invalid GGUF magic
```

处理方式：

- 确认文件路径指向 `.gguf` 文件。
- 删除不完整文件，重新运行 `python scripts/download_demo_model.py --dir models/chat`。
- 用 `ls -lh` 看文件大小是否明显异常。

### GGUF 版本或架构不支持

典型输出：

```text
Failed to load model: GGUF parse error: unsupported GGUF version: 1
```

或：

```text
Failed to load model: Failed to build config: Invalid model config from GGUF: dim=0 ...
```

处理方式：

- 先跑 `inspect-gguf`，确认 `general.architecture`、`embedding_length`、`block_count` 等字段是否存在。
- 当前路径重点支持 Qwen2 这类已经映射过的结构。换模型时需要确认 loader 里有对应字段映射。

### tokenizer 加载失败

典型信号：

```text
GGUF missing tokenizer.ggml.tokens
Failed to load tokenizer.
```

处理方式：

- 用 `inspect-gguf` 查 `tokenizer.ggml.tokens` 和 `tokenizer.ggml.merges`。
- 如果 GGUF 缺 tokenizer metadata，确认同目录是否有 `vocab.json` 和 `merges.txt`。
- 显式传入外部 tokenizer：

```bash
./build/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  --tokenizer models/chat/vocab.json \
  -p hello \
  -n 1
```

### 输出乱码或明显跑偏

优先按这个顺序查：

1. 用 `generate -p hello -n 1` 确认基础链路是否正常。
2. 用 `inspect-gguf` 查 `tokenizer.chat_template`、`bos_token_id` 和 `eos_token_id`。
3. 对比 `GgufTokenizer` 和 `BpeTokenizer` 的编码结果，确认 vocab 和 merges 没有错配。
4. 检查 `general.architecture` 是否让 RoPE 逻辑走到了正确分支。
5. 换成 Q8_0 路径做对照，排除低位量化带来的质量波动。

### 运行期崩溃

如果怀疑越界或 use-after-free，在 GCC/Clang 上跑 ASan 构建：

```bash
cmake -S . -B build-asan \
  -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address'
cmake --build build-asan -j4
./build-asan/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  -p hello \
  -n 1
```

### 内存不足

Qwen2-0.5B-Q8_0 加载后还会分配 KV Cache 和中间张量。内存紧张时，可能被系统直接杀掉。

处理方式：

- 先用 `-n 1` 做最短 smoke。
- 关闭其他占用内存的进程。
- 换更小或更高压缩比的 GGUF 模型做链路验证。

## 本章验证

真实模型 smoke 通过时，至少应该看到：

- `inspect-gguf` 能打印 GGUF info、metadata 和 tensor 列表。
- `generate` 能打印 prompt token、generated token 和 `trace summary ... status=ok`。
- `run` 能进入交互界面，至少完成一轮输入和退出。
- `/stats` 能显示会话统计。
- `ctest --test-dir build -R mini-llama-tests --output-on-failure` 通过。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
./build/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  -p hello \
  -n 1
```

## 目录结构

```
mini-llama/
├── include/mini_llama/
│   ├── tensor.h / ops.h / matmul_dispatch.h / thread_pool.h
│   ├── tokenizer.h         ITokenizer + Ascii/Json/BPE
│   ├── gguf.h              GGUF v3 Reader / metadata / tensor index
│   ├── gguf_loader.h       GGUF → MiniLlamaModel
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
│   ├── Qwen2-0.5B-Instruct-Q8_0.gguf   需 download_demo_model.py
│   ├── chat_template.txt
│   ├── vocab.json / merges.txt / special_tokens.json
├── scripts/
│   ├── download_demo_model.py
│   ├── export_gguf_tokenizer.py
│   ├── make_test_gguf.py
│   └── test_real_model_smoke.py
├── src/                    对应实现
└── tests/
    ├── test_tensor.cc / test_ops.cc
    ├── test_tokenizer.cc / test_gguf_tokenizer.cc
    ├── test_kv_cache.cc
    ├── test_batch.cc
    ├── test_forward.cc
    ├── test_sampler.cc
    ├── test_radix_tree.cc
    ├── test_request_context.cc
    ├── test_chat.cc
    ├── test_loader.cc
    ├── test_gguf.cc
    └── test_gguf_loader.cc
```

## Forward 主链路

`ForwardToken` 是最小计算单元：整数 token id → Embedding 查表 → N 层 `ForwardLayer` → 最终 RMSNorm + `lm_head` → `logits[vocab_size]`。

`ForwardBatch` 只做调度：校验 batch，按 position 顺序调用 `ForwardToken`，更新 `ctx.pos` / `token_history`。Prefill（多 token）和 Decode（单 token）共用这一条接口。

每层顺序：

1. Attention：Pre-Norm → QKV Linear → reshape 多头 → RoPE(Q,K) → 先写 KV Cache 再算 Attention → `wo` → 残差
2. FFN：Pre-Norm → Gate/Up → SwiGLU → Down → 残差

GQA：`kv_head = q_head / (n_heads / n_kv_heads)`。Attention 点积后除以 `sqrt(head_dim)`，多头之间用 `ParallelFor` 无锁并行。

本仓库只走 **CPU**。Q8_0 / Q4_0 / Q4_1 权重会在 CPU 上反量化后参与计算；CUDA device-resident 路径未接入。

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

`GgufReader::Load` 顺序解析 Header → Metadata KV → Tensor Info → 按 `general.alignment`（默认 32 字节）对齐后得到 `data_offset`。只读前半段索引，不把权重整文件映射进内存。支持 GGUF v3、小端、13 种标量类型及 string/array；tensor dtype 可识别 F32 / F16 / Q4_0 / Q4_1 / Q8_0 并计算字节数。`LoadGgufModel` 把 Qwen2/LLaMA 命名映射进 `MiniLlamaModel`。K-Quants、IQ、大端、分片 GGUF 是后续扩展。`inspect-gguf` 打印 GGUF info、metadata 与 tensor 索引。

## JSON+BIN Loader

`model.json` 分三块：config、tokenizer、tensors 索引。`model.bin` 无文件头，按 offset 跳读 F32。`LoadModel` 五道校验：必填字段、维度整除/RoPE 偶 head_dim、tensor 五元组、byte_size 与区间不重叠、bin 长度与尾字节。JSON+BIN 的 `rope_type` 用 `kNormal`，QKV bias 保持空 Tensor。量化容器已接入 GGUF loader；JSON+BIN 路径仍是 F32。

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
