# mini-llama

从零实现的 LLaMA 风格 C++ 推理引擎，按课程步骤逐步搭建。

当前进度：**Benchmark 设计** — 给性能一个固定测量入口。`bench` 把一次推理拆成 prompt tokens、generated tokens、prefill / decode 耗时、tokens/s 和权重内存。

前面几章已经把真实模型路径上的关键模块接起来了：

- GGUF Reader 负责读取 metadata 和 tensor。
- GGUF Loader 负责把 Qwen2 权重映射到 `MiniLlamaModel`。
- `GgufTokenizer` 负责从 GGUF metadata 读取 vocab、merges 和特殊 token。
- `PromptBuilder` 负责把对话消息拼成 Qwen2 ChatML prompt。
- Forward、KV Cache 和 Sampler 负责完成自回归生成。

Smoke test 只判断链路是否稳定可用。短输出的内容质量不代表模型完整能力，尤其是 `-n 1` 或 `-n 3` 这种极短生成。本仓库是 CPU 路径，不包含 CUDA。`bench --quant q8_0|q4_0` 会在加载后临时量化 Linear 权重，用来对照体积和 logits 误差。

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
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --verbose
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --quant q8_0
./build/mini-llama bench models/chat -p hello -n 32 --seed 1 --threads 4
```

`generate` 默认加载 `models/tiny` 的 JSON+BIN。`inspect` 先打印清单，再完整读入权重。JSON+BIN 当前 dtype 固定为 float32。

内置命令：`/help` `/clear` `/stats` `/params` `/exit`。`/clear` 会重建 `MiniLlamaContext`，避免 KV Cache 的 `pos` 与文本长度错位。

`run models/chat` 会自动解析目录里的可推理 GGUF；加载失败会直接报错退出，即使同目录有 `vocab.json` 也不会改用合成权重。教学对话用 `run models/tiny` 或显式 `run models/tiny --synthetic`。目录里的 inspect 夹具 `test.gguf` 不会被当成聊天模型。

## Benchmark 设计

真实模型已经能跑起来之后，需要一个固定的测速入口。手动观察「感觉快了」没有意义，性能优化要看可测量的数据。

`bench` 子命令负责把一次推理拆成几类指标：

- prompt 有多少 token。
- 生成了多少 token。
- prefill 花了多少时间。
- decode 花了多少时间。
- 总 tokens/s 和 decode tokens/s 分别是多少。
- 权重实际占用多少内存，等价 F32 权重需要多少内存。

对应代码集中在：

- `include/mini_llama/debug.h`
- `src/debug.cc`
- `src/main.cc`
- `tests/test_debug.cc`

### 为什么要分开看 prefill 和 decode

一次生成通常分成两段。

**Prefill** 把 prompt token 一次性喂进模型，得到最后一个位置的 logits。prompt 越长，这一步处理的 token 越多。它更像批量计算，矩阵乘法和 attention 都能一次处理多 token。

**Decode** 生成阶段每次只新增一个 token。每生成一个 token，都要把这个 token 再送回模型，读权重、读 KV cache、算下一个 logits。用户看到的「输出速度」主要取决于 decode 阶段。

所以 benchmark 同时输出：

```text
prefill time
Decode time
tokens/s (total)
tokens/s (Decode)
```

这几个数字不要混着看。长 prompt 会放大 prefill；长生成会让 decode 更稳定；`-n 1` 只适合 smoke，不能代表稳定吞吐。

### BenchmarkResult 记录什么

`BenchmarkResult` 定义在 `include/mini_llama/debug.h`：

- `n_generated_tokens` 包含所有生成 token。
- `n_decode_tokens` 从第二个生成 token 开始计数。

原因在 `RunBenchmark()` 里：第一个 token 是从 prefill 后的 logits 直接采样出来的，还没有进入单 token decode forward。后续 token 才会走 `MiniBatch::Single()`。因此 `-n 4` 通常会看到：

```text
generated tokens: 4
Decode tokens:    3
```

吞吐公式：

- `tokens/s (total)` = `generated_tokens / (prefill_ms + decode_ms)`
- `tokens/s (Decode)` = `decode_tokens / decode_ms`

### bench 命令参数

入口：

```bash
./build/mini-llama bench <model-path|dir> [options]
```

常用参数：

| 参数 | 说明 |
|------|------|
| `-p, --prompt <str>` | 输入 prompt，默认 `"hello"` |
| `-n, --n-predict <n>` | 生成 token 数，默认 64 |
| `--seed <S>` | 采样种子，默认 0 |
| `--tokenizer <path>` | 显式指定 vocab.json |
| `--quant q8_0\|q4_0` | benchmark 前先量化 Linear 权重 |
| `--threads <n>` | 设置 CPU 并行线程数，0 表示自动 |
| `--verbose` | 打印调试信息 |

基础 tiny benchmark：

```bash
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42
```

真实 Qwen2 benchmark：

```bash
./build/mini-llama bench models/chat \
  -p hello \
  -n 32 \
  --seed 1
```

指定线程数：

```bash
./build/mini-llama bench models/chat \
  -p hello \
  -n 32 \
  --seed 1 \
  --threads 4
```

量化后测速：

```bash
./build/mini-llama bench models/tiny \
  -p hello \
  -n 4 \
  --seed 42 \
  --quant q8_0
```

本仓库只走 CPU。CUDA backend 指标（uploaded weights、host/device copy、GPU kernel 调用次数）等进入后续 CUDA 章节后再作为 GPU benchmark 的判断依据。当前先把 CPU 路径测准。

### 输出怎么看

```bash
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --verbose
```

关键字段：

- `prompt: "hello" (6 tokens)`：输入 prompt 编码后的 token 数。
- `generated tokens`：本次生成出的 token 数。
- `Decode tokens`：真正走单 token decode forward 的次数。
- `prefill time`：prompt 一次性前向的耗时。
- `Decode time`：decode loop 的累计耗时。
- `tokens/s (total)`：`generated_tokens / (prefill_ms + decode_ms)`。
- `tokens/s (Decode)`：`decode_tokens / decode_ms`。
- `weight memory actual`：当前模型权重实际占用。
- `f32 equiv`：同样权重如果全用 F32 存储的占用。
- `savings`：F32 等价体积除以实际体积。

### verbose 模式能看到什么

`--verbose` 会打印三类调试信息。

**Logits shape 和 top-k**：确认最后输出的 logits 形状是否等于 vocab size，也能看到采样前分数最高的 token id。

**KV cache 状态**：确认层数、上下文长度、KV head 数和 `head_dim` 是否和模型配置一致。

**Decode step**：看到每一步采样出来的 token id。需要继续看文本时，可以用 tokenizer decode 或 `generate` 命令的 `generated text` 输出对照。

### 量化对 benchmark 的影响

`bench` 支持在加载模型后临时执行量化。核心看点有两个：

- `weight memory` 显示量化后的权重体积是否下降。
- `logits error vs model-native` 显示量化前后 logits 的最大误差和平均误差。

吞吐是否提升要看模型大小、CPU 缓存、线程数、SIMD 路径和量化 kernel。tiny 模型太小，结果容易被函数调用开销和计时抖动影响；真实模型更适合看稳定趋势。

### 多线程 benchmark 怎么看

对比线程数时，保持 prompt、`-n`、seed、模型和量化参数一致，只改 `--threads`。重点看 `tokens/s (Decode)`。

decode 阶段每次只处理一个新 token，能不能加速取决于矩阵尺寸、线程调度开销、内存带宽、量化格式和 CPU 缓存。线程数更多不一定线性更快，所以 benchmark 结果要按本机实测来判断。

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

检查 benchmark 模块时，建议确认这些结果：

- `bench models/tiny -p hello -n 4 --seed 42` 能输出 `Results`。
- `generated tokens` 和 `Decode tokens` 的关系符合 `n_predict=4`、`Decode tokens=3`。
- `--verbose` 能打印 logits top-k、KV cache shape 和 Decode step。
- `--quant q8_0` 能打印 `weight memory` 和 `logits error vs model-native`。
- `--threads` 能改变输出里的 `threads` 字段。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --verbose
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --quant q8_0
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --threads 4
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
│   ├── debug.h             BenchmarkResult + dump helpers
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
    ├── test_gguf_loader.cc
    └── test_debug.cc
```

## Forward 主链路

`ForwardToken` 是最小计算单元：整数 token id → Embedding 查表 → N 层 `ForwardLayer` → 最终 RMSNorm + `lm_head` → `logits[vocab_size]`。

`ForwardBatch` 只做调度：校验 batch，按 position 顺序调用 `ForwardToken`，更新 `ctx.pos` / `token_history`。Prefill（多 token）和 Decode（单 token）共用这一条接口。

每层顺序：

1. Attention：Pre-Norm → QKV Linear → reshape 多头 → RoPE(Q,K) → 先写 KV Cache 再算 Attention → `wo` → 残差
2. FFN：Pre-Norm → Gate/Up → SwiGLU → Down → 残差

GQA：`kv_head = q_head / (n_heads / n_kv_heads)`。Attention 点积后除以 `sqrt(head_dim)`，多头之间用 `ParallelFor` 无锁并行。

本仓库只走 **CPU**。Q8_0 / Q4_0 / Q4_1 权重会在 CPU 上按块反量化后参与计算；`bench --quant` 可把已加载的 Linear 权重临时转成 Q8_0 / Q4_0。CUDA device-resident 路径未接入。

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
