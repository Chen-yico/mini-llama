# mini-llama

从零实现的 LLaMA 风格 C++ 推理引擎，按课程步骤逐步搭建。

当前进度：**CUDA Backend 总览** — 默认仍走 CPU。打开 `MINI_LLAMA_CUDA` 之后，`generate`、`run`、`bench`、`inspect` 可以用 `--backend cuda` 把计算留在显存里，logits 回到 CPU 采样。

前面几章已经把真实模型路径上的关键模块接起来了：

- GGUF Reader 负责读取 metadata 和 tensor。
- GGUF Loader 负责把 Qwen2 权重映射到 `MiniLlamaModel`。
- `GgufTokenizer` 负责从 GGUF metadata 读取 vocab、merges 和特殊 token。
- `PromptBuilder` 负责把对话消息拼成 Qwen2 ChatML prompt。
- Forward、KV Cache 和 Sampler 负责完成自回归生成。

Smoke test 只判断链路是否稳定可用。短输出的内容质量不代表模型完整能力，尤其是 `-n 1` 或 `-n 3` 这种极短生成。默认构建是 CPU。CUDA 后端用 `-DMINI_LLAMA_CUDA=ON` 打开，当前只支持单机单卡。`bench --quant q8_0|q4_0` 会在加载后临时量化 Linear 权重，用来对照体积和 logits 误差。

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
./build/mini-llama generate --model models/tiny --quant q8_0 -p hello -n 4
./build/mini-llama generate --model models/tiny --quant q4_0 -p hello -n 4
./build/mini-llama inspect models/tiny
./build/mini-llama inspect-gguf models/tiny/test.gguf
./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
./build/mini-llama run models/tiny -n 8
./build/mini-llama run models/chat -n 8
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --verbose
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --quant q8_0
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --threads 1 --quant q4_0
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

这一节测的是 CPU 量化。CUDA 的 uploaded weights、host/device copy 和 kernel 次数放在后面的 CUDA Backend 总览里看。

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

## Q8_0 量化

F32 每个权重 4 字节。Decode 每生成一个 token 都要反复读线性层权重，模型越大，读权重越容易成为主要成本。

Q8_0 每 32 个浮点收成一个 block：1 个 FP16 scale（2 字节）加 32 个 int8（32 字节），共 34 字节。32 个 F32 原来是 128 字节，单个 block 的理论压缩比是 `128 / 34 ≈ 3.76x`。

`BlockQ80` 定义在 `include/mini_llama/quantized_tensor.h`，布局对齐 ggml / llama.cpp 的 `block_q8_0`：

- `d`：这个 block 的 FP16 scale。
- `qs`：32 个量化后的 int8。
- 反量化：`x[i] = fp16_to_float(d) * qs[i]`。
- 全零 block 的 `d` 和 `qs` 都是 0。一行尾部不足 32 个元素时，剩余位置补 0，反量化只写回原始 shape 覆盖到的元素。

`QuantizedTensor` 用 `QuantType` 标记当前格式。线性层可以保持 F32、Q8_0、Q4_0 或 Q4_1，Forward 按类型分发。GGUF loader 读到 Q8_0 时按 block 字节拷进 `q8_0_data`，不先把整份权重展开成 F32。

`QuantizeToQ80()` 按行切 block。二维权重 `[out_features, in_features]` 的每一行单独切，block 不跨输出通道。每个 block 取绝对值最大的元素做 scale：`d = max_abs / 127`，再把 `d` 存成 FP16，用存回去的 `stored_d` 反算量化系数。全零 block 直接把 `d` 设为 0。

`DequantizeFromQ80()` 按原始 shape 还原，并检查 block 数是否匹配。

推理主链路走 `LinearQ80()`：点积里按元素反量化，不分配整份 F32 权重。输入可以是 `[in_features]`（单 token decode）或 `[batch, in_features]`（prefill）。ARM NEON 上一次处理 8 个 int8；其他平台走通用 C++ 循环。`MatmulQ80()` 会先整块还原成 F32 再做普通矩阵乘，用来看数值误差，推理不走这条路径。

误差来自取整。单个元素的理想舍入大约不超过 `0.5 * scale`，再加上 FP16 scale 本身的舍入。测试用 `max_err < 6e-2` 看 roundtrip，矩阵乘累加后用更宽的 `2e-1`。

`generate` 和 `bench` 都支持 `--quant q8_0`。`QuantizeModelToQ80()` 只转换线性权重，Embedding、RMSNorm 和 bias 仍是 F32，所以 tiny 模型的整体压缩比会低于 3.76x。

```bash
./build/mini-llama generate --model models/tiny --quant q8_0 -p hello -n 4
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --quant q8_0
```

## Q4_0 量化

Q8_0 已经把 32 个 F32 从 128 字节压到 34 字节。Q4_0 继续压缩：32 个权重只保留 32 个 4-bit 数值，再加 1 个 FP16 scale，一块 18 字节。单个 block 的理论压缩比是 `128 / 18 ≈ 7.11x`。

`BlockQ40` 定义在 `include/mini_llama/quantized_tensor.h`，布局对齐 ggml / llama.cpp 的 `block_q4_0`：

- `d`：这个 block 的 FP16 scale。
- `qs`：16 个字节，每个字节存两个 4-bit 量化值。
- 存储用无符号 nibble `0..15`，计算时减 8，还原成有符号范围 `-8..7`：`value = fp16_to_float(d) * (q - 8)`。

半块交错打包：`qs[j]` 的低 4 位是元素 `j`，高 4 位是元素 `j + 16`。前 16 个元素放在所有字节的低 4 位，后 16 个元素放在高 4 位。解包时可以一次拿出一整段低 nibble 或高 nibble。

`QuantizeToQ40()` 和 Q8_0 一样按行切 block，每个输出通道单独切。每个 block 先找绝对值上限，scale 用 `max_abs / 7`，因为 Q4_0 的最大正整数是 7。`d` 存成 FP16 后，用存回去的 `stored_d` 反算量化系数。打包时一次处理 `j` 和 `j + 16`，`+ 8` 是零点偏移，把近似 `[-8, 7]` 搬到 `[0, 15]`。

`DequantizeFromQ40()` 把低 4 位还原成 `idx0`，高 4 位还原成 `idx1`。例如 `d = 1`、`qs[0] = 0xA3`：低 4 位是 3，减 8 得到元素 0 的 `-5`；高 4 位是 10，减 8 得到元素 16 的 `2`。

推理主链路走 `LinearQ40()`。通用路径复用 `LinearQuantizedImpl()`，通过 `DequantQ40()` 取出某个 block 内的权重，点积时按需解包。ARM NEON 上一次解包 16 个 nibble，并对 16 个值做向量乘加；这条专用路径支持 `[in_features]` 或 `[1, in_features]`。其他平台走通用 C++ 路径，支持 `[in_features]` 或 `[batch, in_features]`。

Q4_0 只有 16 个离散值，误差通常比 Q8_0 大。roundtrip 测试用 `max_err < 3e-1`，`LinearQ40` 对齐 F32 时用 `2.0`。`CompareQ40Error()` 先跑 F32 `Linear()`，再跑 `LinearQ40()`，返回两边输出的最大绝对误差。

`generate` 和 `bench` 都支持 `--quant q4_0`。`QuantizeModelToQ40()` 只转换线性权重，Embedding、RMSNorm 和 bias 仍是 F32，所以 tiny 模型的整体压缩比低于 7.11x，`weight memory` 大约是 `3.94x`。tiny 太小，吞吐容易被计时抖动盖住；这里更适合看权重体积是否下降，以及 logits 相对 F32 的偏差。

```bash
./build/mini-llama generate --model models/tiny --quant q4_0 -p hello -n 4
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --threads 1 --quant q4_0
```

## Matmul Dispatch 设计

量化把权重存成了 Q8_0、Q4_0。到了推理主链路，线性层调用仍然保持简单：上层只关心 `Linear(x, weight)`，底层根据数据类型、线程数和 CPU 指令集选择执行路径。

公开接口在 `include/mini_llama/ops.h`：

- `Matmul(a, b)`：通用矩阵乘 `[M, K] x [K, N] -> [M, N]`。
- `Linear(x, weight)`：F32 权重，`x @ weight^T`。
- `Linear(x, QuantizedTensor)`：按 `QuantType` 分到 `LinearQ80()` / `LinearQ40()` / `LinearQ41()`。F32 再转回上面的 Tensor 路径。

`src/forward.cc` 里的 attention projection、FFN 和 `lm_head` 都经过 `ForwardLinear()`。线性权重在这个项目里已经是 `QuantizedTensor`，所以这一层直接把权重交给对应的 `Linear()` 重载：F32 走 F32 分发器，量化权重在点积里按需反量化。

`include/mini_llama/matmul_dispatch.h` 定义四种模式：`kNaive`、`kThreaded`、`kSimd`、`kThreadedSimd`。公开的 `Matmul()` 和 F32 `Linear()` 使用 `DefaultMatmulMode()`，当前默认是 `kThreadedSimd`。这个名字表示优先使用多线程和 SIMD，实际路径还要看内存访问：

- `Linear()` 的输入和权重每一行都连续，可以走 SIMD 点积。`kThreadedSimd` 按 `out_features` 把输出通道切给不同线程，每个线程内部再用 `DotSimd()`。外层是任务级并行，内层是指令级并行。
- 通用 `Matmul()` 的右矩阵是 row-major。计算 `c[i, j]` 时要读 `b[0, j]`、`b[1, j]`……这些元素在内存里间隔 `N`，连续加载条件差。所以 `kSimd` 和 `kThreadedSimd` 目前都回到 `MatmulThreaded()`。

`MatmulNaive()` / `LinearNaive()` 是普通三重循环，给优化路径当尺子。`ParallelFor(n, fn)` 把 `[0, n)` 切成若干 `[begin, end)`，每个线程写自己的输出位置，不需要额外加锁。worker 里的异常会捕获后在主线程重新抛出。任务量小于 `线程数 * 16` 时直接在当前线程执行，避免小矩阵的调度开销。

`DotSimd()` 在编译器定义了 AVX2 + FMA 时走 AVX2，ARM NEON 平台走 NEON，否则回到标量循环。MSVC 默认不定义 `__AVX2__`，这条路径在当前 Windows 构建里是标量回退；数值仍与 naive 对齐。

`generate` 和 `bench` 都支持 `--threads <n>`。`SetThreadCount(0)` 表示使用 `hardware_concurrency()`，读不到时回退到 4。`--dump-logits <dir>` 把逐步 logits 写成二进制，用来确认不同线程数的输出一致。

```bash
./build/mini-llama bench models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello \
  --n-predict 4 \
  --threads 4 \
  --verbose
```

输出里应包含 `threads: 4`、`[verbose] prefill` 和 `tokens/s (total):`。tiny 模型很小，线程数对吞吐的影响不稳定；这里先确认参数进了执行路径，以及优化结果和参考路径接近。

## Matmul Dispatch 本章验证

检查 Matmul Dispatch 时，建议确认这些结果：

- `matmul_naive_vs_threaded`：通用 matmul 的 naive 和 threaded 输出一致。
- `linear_all_modes_match`：F32 linear 四种模式输出一致。
- `linear_2d_input_all_modes_match`：`[1, in_features]` 输入 shape 保持一致。
- `different_thread_counts_same_output`：线程数为 1、2、4 时输出一致。
- `thread_count_api`：`SetThreadCount()` 和 `GetThreadCount()` 行为正确。
- `parallel_for_propagates_exception`：worker 线程异常能回到主线程。
- `generate --threads 1` 和 `generate --threads 4` 导出的 logits 文件一致。
- `bench --threads 4 --verbose` 会报告线程数和吞吐指标。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R "mini-llama-tests|threaded-cli-smoke" --output-on-failure
python3 scripts/test_threaded_cli.py
```

## Thread Pool 与多线程 Matmul

CPU 推理的主要耗时在线性层。一次 `Linear(x, W)` 是输入向量乘一整张权重矩阵。输出的每个元素都是权重的一行和输入做点积，不同输出行互不依赖。适合 CPU 并行的任务要满足三件事：单个任务有足够计算量，切分后尽量少同步，不同线程写回的位置也清楚。线性层正好是这个形状。

本仓库不维护常驻 worker 队列。`ParallelFor()` 每次调用时把 `[0, n)` 拆成连续区间，创建若干 `std::thread`，全部 `join()` 后再返回。常驻线程池能少付创建开销，但会带上任务队列、唤醒和生命周期管理。当前阶段先把「按区间切任务」写清楚，供 `Linear`、`Matmul` 和 Attention 复用。

```cpp
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);
```

调用方只提供总任务数和一段 `[begin, end)` 的处理函数。余数分给前面的线程，所以各段长度尽量接近。例如 `n = 10`、线程数 `3` 时，区间是 `[0, 4)`、`[4, 7)`、`[7, 10)`。

线程数来自 `GetThreadCount()`。`g_thread_count == 0` 时使用 `std::thread::hardware_concurrency()`，读不到则回退到 4。`generate` 和 `bench` 的 `--threads` 会调用 `SetThreadCount()`：`0` 恢复自动，大于 `0` 使用指定值。benchmark 会把这个数打印出来，后面比较吞吐时才知道条件相同。

任务太小时不创建线程。每个线程至少要分到 `kMinChunk = 16` 个任务，否则直接在当前线程执行 `fn(0, n)`。例如 `n = 10`、线程数 `8`，每个线程只有一两个任务，创建和 `join` 的开销会盖过计算。线程数也会被限制在 `n` 以内，避免线程比任务多。

worker 里的异常先存进 `std::exception_ptr`，所有线程 `join()` 之后再在主线程抛出。异常如果直接逃出线程入口，程序会调用 `std::terminate()`。

`LinearThreaded()` 按 `out_features` 切。输入 `x` 和权重只读，可以被多个线程共享；每个线程写入不同的 `result[j]`，所以不需要锁。`different_thread_counts_same_output` 测的就是线程数变化后结果仍一致。`LinearThreadedSimd()` 用同一套切分，每个输出通道内部再做 SIMD 点积。

通用 `MatmulThreaded()` 按输出矩阵的行切，也就是 `M` 维。每个线程负责若干行，`c[i, j]` 只由对应线程写一次。通用矩阵乘的 SIMD 模式仍回到这条路径，因为右矩阵按列访问，内存不连续。

`src/forward.cc` 里的 CPU Attention 按 head 切：`ParallelFor(n_heads, ...)`，每个 head 写自己的 `attn_out[h, :]`。这条切分只服务 CPU Attention。CUDA 路径用显存里的 KV cache 和单独的 attention kernel。

`ParallelFor()` 适合粗粒度循环，例如 `out_features` 和 `n_heads`。放进很细的内层循环时，调度成本会吃掉收益。

和 llama.cpp 的对应关系是：`llama_context::set_n_threads`、`ggml_backend_cpu_set_n_threads`、`ggml_backend_cpu_set_threadpool` 和 `ggml_graph_compute` 把线程数交给 CPU 计算图。llama.cpp 还包含 batch 线程数、后端 threadpool、计算图规划和工作区复用。mini 版只保留同一条主线：把可独立计算的区间切开，并行执行，再把结果汇合。

```bash
./build/mini-llama bench models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello \
  --n-predict 4 \
  --threads 4 \
  --verbose
```

tiny 模型太小，线程数从 1 扫到 8 不一定让 `tokens/s (Decode)` 稳定上升。这里先看两个确定信号：输出里的线程数，以及不同线程数导出的 logits 是否一致。

## Thread Pool 本章验证

检查 Thread Pool 时，建议确认这些结果：

- `mini-llama-tests` 通过，其中包括 `matmul_naive_vs_threaded`、`different_thread_counts_same_output`、`thread_count_api`、`parallel_for_propagates_exception`。
- `scripts/test_threaded_cli.py` 输出两个 `PASS`。
- benchmark 输出包含 `threads: 4`。
- verbose 输出包含 `[verbose] prefill`。
- benchmark 输出包含 `tokens/s (total)` 和 `tokens/s (Decode)`。
- `LinearThreaded()` 不需要锁：输入和权重只读，每个线程只写自己的输出区间。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R "mini-llama-tests|threaded-cli-smoke" --output-on-failure
python3 scripts/test_threaded_cli.py
./build/mini-llama bench models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello \
  --n-predict 4 \
  --threads 4 \
  --verbose
```

## SIMD 优化入门

上一章把不同输出切给不同线程。这一章更细一层：单个线程算一个点积时，一条指令能不能一次处理多个数。多线程解决任务怎么分给多个核，SIMD 解决一个核里一条指令怎么处理更多数据。

SIMD 是 Single Instruction, Multiple Data。标量循环一次只做 `sum += a[i] * b[i]`。向量指令把多个元素放进寄存器，一条指令同时做多组乘加。ARM NEON 是 128-bit，一次 4 个 float。x86 AVX2 是 256-bit，一次 8 个 float。

本项目里 SIMD 主要服务 `Linear(x, W)`。输出的每个元素都是输入向量和权重的一行做点积。`x` 连续，`W` 的每一行也连续，CPU 可以把一小段一次装进向量寄存器。数据如果散落在内存各处，指令再强也很难发挥。

对应代码：

- `src/matmul_optimized.cc`：F32 的 `DotSimd()`、`LinearSimd()`、`LinearThreadedSimd()`。
- `src/quant.cc`：Q8_0 / Q4_0 的 NEON 专用 `LinearQ80()`、`LinearQ40()`。
- `tests/test_threaded_matmul.cc`：四种 F32 模式的输出是否一致。
- `tests/test_quant.cc`：量化 Linear 和 F32 的误差是否落在阈值内。

`DotSimd()` 在编译期分发，不在运行时探测。编译器同时定义了 `__AVX2__` 和 `__FMA__` 就编进 AVX2；定义了 `__ARM_NEON` 就编进 NEON；两个都没有就留下标量循环。CMake 没有主动加 `-mavx2 -mfma`。Apple Silicon 通常会自己打开 NEON。x86 机器即使硬件支持 AVX2，编译参数没打开时仍然走标量循环。这也是不同机器上 benchmark 差很多的原因之一。MSVC 默认不定义 `__AVX2__`，当前 Windows 构建里的 `DotSimd()` 是标量回退，外层多线程仍然生效。

两条向量路径的形状一样：先按寄存器宽度累加，再把寄存器里的多个 lane 横向加成一个 float，最后用普通循环处理不能整除的尾部。NEON 一次 4 个，尾部最多剩 3 个；AVX2 一次 8 个，尾部最多剩 7 个。模型维度经常是 32、64、128，但底层函数仍要能处理任意长度。AVX2 用 `_mm256_loadu_ps` 做非对齐加载，不要求每行首地址按 32 字节对齐。

`LinearSimd()` 对每个输出通道调用一次 `DotSimd()`。`LinearThreadedSimd()` 再用 `ParallelFor` 把输出通道切给多个线程，每个线程内部继续做 SIMD 点积。F32 `Linear()` 的默认模式是 `kThreadedSimd`，也就是多线程加 SIMD 点积。编译环境没有 NEON 或 AVX2 时，`DotSimd()` 内部是标量循环，外层多线程还在。

通用 `Matmul(a, b)` 的 `kSimd` 和 `kThreadedSimd` 都回到 `MatmulThreaded()`。右矩阵按列读，相邻元素隔着一整行，不适合直接向量加载。要把通用矩阵乘做成高性能 SIMD，还要转置、分块和照顾 cache。当前把 SIMD 放在收益明确、代码也短的 Linear 点积上。

量化 Linear 也有 SIMD 分支，但只写了 NEON。Q8_0 一次加载 8 个 int8，扩展到 int32，转成 float，乘 scale，再和输入乘加。Q4_0 先拆 16 个 nibble，减 8，再转成 float 做乘加。非 NEON 平台走通用的 `LinearQuantizedImpl()`，边读 block 边反量化。量化路径没有 AVX2 专用实现。量化 SIMD 比 F32 点积更绕，因为它要先把压缩格式还原成能参与计算的数，而不是先把整张权重展开成 F32。

SIMD 和量化的测试目标不同。SIMD 是同一批 F32 数据换一种执行方式，`linear_all_modes_match` 要求 naive、threaded、simd、threaded-simd 的误差小于 `1e-5`。量化改变了权重表示，`q8_0_roundtrip_identity`、`q4_0_roundtrip_identity`、`q4_0_linear_matches_f32` 和 `CompareQ40Error` 只要求误差落在阈值内。两类测试分开，才能判断偏差来自执行路径还是量化本身。

release 没有 CLI 参数直接切换 Naive、Simd、ThreadedSimd。用户侧用 `--threads`、`--quant` 和 `bench` 看端到端吞吐。tiny 模型很小，4 线程的 total tokens/s 可能更高，Decode tokens/s 可能更低。一次 benchmark 混进了分词、Prefill、Decode、线程调度和系统负载，不能把它当成 SIMD 单独带来的加速。

## 本章验证

检查 SIMD 时，建议确认这些结果：

- `mini-llama-tests` 通过。
- `linear_all_modes_match` 覆盖 naive、threaded、simd、threaded-simd，输出接近。
- benchmark 输出包含 `threads: 4`。
- benchmark 输出包含 `tokens/s (total)` 和 `tokens/s (Decode)`。
- 能说明 NEON、AVX2、标量 fallback 三条路径的启用条件。
- 能解释 `Linear()` 适合 SIMD、通用 `Matmul()` 当前回到 threaded 的原因。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
./build/mini-llama bench models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello \
  --n-predict 4 \
  --threads 4 \
  --verbose
```

## Python 参考对齐

前面的章节已经把推理主链路、量化、多线程和 SIMD 接上了。程序能吐出 token，单元测试也能过，但还有一类问题看不出来：每一步 logits 和一份更直观的数学实现是不是同一组数。

RMSNorm 的 `eps` 用错、RoPE 的位置传偏、GQA 的 kv head 映射错了、KV cache 写进了错误的 `pos`、`lm_head` 的矩阵方向反了，这些都不会让程序直接崩。终端照样打印 token，最后的文本对不上时，也分不清问题在模型、采样还是底层计算。

所以这里放一条 Python 参考链路。`scripts/reference_forward.py` 用 NumPy 写一份朴素的 Transformer 前向，没有工程层，也没有优化。函数里直接是 embedding、RMSNorm、Q/K/V、RoPE、KV cache、attention、FFN、final norm 和 `lm_head`。核心函数是 `forward_token(model, pos, token_id)`，一次处理一个 token，语义和 C++ 的 `ForwardBatch` decode 一致。KV cache 放在 `model["kv_cache"]` 里，第一次调用时创建，后面持续写入同一个缓存，prefill 写进去的 K/V 会继续影响 decode，对应 C++ 的 `MiniLlamaContext`。

tiny model 用简单 tokenizer：BOS 固定是 `1`，prompt 字符转成 ASCII token，范围 `0..127`。`hello` 的 prompt token 是 6 个：`[1, 104, 101, 108, 108, 111]`。`-n 16` 表示再生成 16 个 token，所以一共 21 份 logits：6 份来自 prefill，15 份来自 decode。第一个生成 token 从最后一份 prefill logits 里取出来，decode 的 logits 文件数比生成 token 数少 1。

每个 `python_logits_step*.bin` 和 C++ 的 `logits_step*.bin` 都是连续 float32，长度等于 `vocab_size`。tiny model 的 `vocab_size` 是 128。`python_generation_tokens.txt` 和 `generation_tokens.txt` 放 greedy 采样得到的 token id。

正常生成时，prefill 把整个 prompt 当成一个 batch 送进 `ForwardBatch()`。打开 `--dump-logits` 之后，C++ 把 prompt token 拆成逐个 step，每个 prompt token 落一份 logits，文件数才能和 Python 对齐。这套逐步导出只给调试和回归用。平时 prefill 继续走 batch。

`scripts/compare_logits.py` 检查三件事：两边的 step 编号必须一致，缺一个就失败；每一步算 `max_abs_error` 和 `max_rel_error`；每一步的 top-1 token 一致，最后的生成 token 序列也一致。通过标准是 `max_abs_error <= 1e-4` 且 `max_rel_error <= 1e-4`。相对误差只在 `|py| >= 1e-3` 时计算，靠近 0 的数做除法会把噪声放大。

日常回归直接跑 `scripts/test_golden.py`。它建一个临时目录，按固定顺序走 `generate --dump-logits`、`make_golden.py`、`compare_logits.py`。默认二进制是 `build/mini-llama`。CTest 里通过 `MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>` 传入刚编出来的程序，工作目录是源码根目录，这样 `models/tiny` 能被找到。

这条测试只覆盖 tiny model 的 F32 主路径。改了 `ForwardBatch()`、RMSNorm、RoPE、KV cache、sampler、matmul dispatch、多线程或 SIMD，都该跑一次 `golden-logits`。量化和真实 GGUF 模型各有自己的测试。Q8_0 / Q4_0 改了权重表示，logits 残差本来就比 F32 大，不要拿量化结果去对这份 golden。

对不上的时候先看是哪一步开始分叉：

- 所有 step 都发散：先查权重读取、embedding、RMSNorm 的 `eps`、矩阵维度和转置方向。tiny model 两边读的是同一份文件。
- prefill 对齐、decode 开始发散：看 KV cache 的 `pos`，或者 GQA 的 `head -> kv_head` 映射。decode 每一步都复用历史 K/V。
- 误差不大但只有 top-1 分叉：最大 logits 附近的候选很接近，再查 sampler 的 tie-breaking。当前 greedy 走 argmax。
- Naive 通过、SIMD 或多线程失败：查尾部元素、浮点累加顺序、线程切片边界。`scripts/test_threaded_cli.py` 会比较 `--threads 1` 和 `--threads 4` 的 logits。

## 本章验证

检查 Python 参考对齐时，建议确认这些结果：

- `Comparing 21 steps...`，`hello` 加 `-n 16` 正好 21 份 logits。
- 每一步 `max_abs_error <= 1e-4`，`max_rel_error <= 1e-4`。
- 每一步 top-1 token 一致。
- `generation tokens: PASS`。
- 最后一行是 `RESULT: ALL PASSED`。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R golden-logits --output-on-failure
```

也可以手动走一遍，看中间文件：

```bash
./build/mini-llama generate \
  --model models/tiny \
  --prompt hello \
  --n-predict 16 \
  --dump-logits artifacts/golden

python3 scripts/make_golden.py \
  --config models/tiny/model.json \
  --weights models/tiny/model.bin \
  --prompt hello \
  -n 16 \
  --output-dir artifacts/golden

python3 scripts/compare_logits.py \
  --cpp-dir artifacts/golden \
  --py-dir artifacts/golden \
  --vocab-size 128
```

## CUDA Backend 总览

CPU 路径已经覆盖 GGUF 解析、BPE 分词、基础算子、多线程、量化和 Python logits 对齐。这些结果是 CUDA 路径的正确性底座。CUDA 侧复用同一套 Transformer 拓扑，只把计算密集的算子换成 GPU 实现，再拿 CPU logits 做数值对照。遇到不支持的量化类型、异常维度或显存不够时，计算可以退回 CPU。

后端入口在 `include/mini_llama/backend.h`：

```cpp
enum class BackendKind { kCpu, kCuda };

struct BackendConfig {
  BackendKind kind = BackendKind::kCpu;
  int device_id = 0;
  bool device_id_set = false;
};
```

`device_id` 留给 `--device <N>`。当前实现以单卡为主，显存分配和 kernel 提交都绑在这一张卡上。`ParseBackendKind()` 只接受 `cpu` 和 `cuda`，命令行解析和推理内核分开：参数先变成 `BackendConfig`，内核只看这份配置。

`generate`、`run`、`bench`、`inspect` 都接受 `--backend cpu|cuda`。没有装 CUDA 工具链时，`--backend cuda` 会直接说明需要 `-DMINI_LLAMA_CUDA=ON`。`--cuda-info` 打印设备名、计算能力、显存、驱动和 runtime 版本。

CPU decode 全程留在内存里：embedding、RMSNorm、QKV、RoPE、KV cache、attention、FFN、final norm、lm_head、sampler 都不跨设备。切到 CUDA 之后，这条骨架不变，只是中间张量换成 `CudaTensor`。embedding、QKV、RoPE、KV 写入、attention、FFN 和 lm_head 都留在显存里，只有最终 logits 做一次 device-to-host，再交给 CPU sampler。采样本身很轻，不值得为它再写一套 GPU 采样。

模型加载后如果请求了量化，会先改 CPU 上的权重，再调用 `UploadModelWeightsToCuda()`。F32、Q8_0、Q4_0、Q4_1 的 Linear 权重，以及 embedding、RMSNorm 和 bias，按名字上传到显存。CPU 侧的原权重还留着，fallback 和对照都能用。

`forward.cc` 里设备路径的入口是这些函数，输入和输出都是 `CudaTensor`：

- `EmbedTokenDevice`
- `ForwardRmsNormDevice`
- `ForwardLinearDevice`
- `ForwardRopeDevice`
- `AttentionForwardDevice`
- `FfnForwardDevice`
- `ForwardLayerDevice`
- `ComputeLogitsDevice`

运行时计数写在 `CudaModelWeights` 上，用来确认程序到底走了哪条路：

| 指标 | 含义 |
| --- | --- |
| uploaded weights | 成功上传到显存的权重数量 |
| gpu bytes | 上传权重大小 |
| cuda Linear calls | 矩阵乘相关的 CUDA 调用次数 |
| cuda activation calls | RMSNorm、RoPE、激活函数等 CUDA 调用次数 |
| cuda attention calls | 在 GPU 上执行 attention 的次数 |
| cpu attention fallback calls | 退回 CPU 执行 attention 的次数 |
| host->device copies | 运行时从内存拷到显存的次数和字节数 |
| device->host copies | 运行时从显存拷回内存的次数和字节数 |

`uploaded weights` 偏少，说明有权重没上去，后面的计算会退回 CPU。`cpu attention fallback calls` 不为 0，说明 attention 没留在显存。decode 过程中 `host->device copies` 持续增加，说明中间结果在 PCIe 上来回搬。Q8_0 且 attention fallback 为 0 时，decode 主路径才算显存常驻。

和 llama.cpp 的对应关系保持在这一层：`BackendKind::kCuda` 对后端选择，`CudaDeviceBuffer` 对设备显存，`CudaTensor` 对常驻张量，`UploadModelWeightsToCuda` 对权重上显存，CUDA Linear / activation kernel 对 matmul 和逐元素算子，`CudaKvCache` 对 GPU KV cache，CUDA attention decode 对注意力 kernel，运行时计数对 backend profiling。

## 本章验证

这一章先确认 CPU 基准构建仍然能跑。有 NVIDIA GPU 和 CUDA 工具链时，再编 CUDA 版本。

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

```bash
nvidia-smi
nvcc --version
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure
./build-cuda/mini-llama --cuda-info
./build-cuda/mini-llama bench models/chat --backend cuda -p hello -n 8
```

CPU 构建上，`--backend cpu` 应打印 `backend: cpu` 和 `compute: cpu`。`--backend cuda` 与 `--cuda-info` 会说明 CUDA 后端没有编进来。CUDA 构建上，`uploaded weights`、`cuda attention calls` 和 `cpu attention fallback calls` 用来确认 decode 有没有留在显存里。

## Q4_0 本章验证

检查 Q4_0 时，建议确认这些结果：

- `q4_0_block_layout`：block size 是 32，`sizeof(BlockQ40)` 是 18。
- `q4_0_roundtrip_identity`：量化再反量化，误差在阈值内。
- `q4_0_all_zeros`：全零输入得到全零输出。
- `q4_0_linear_matches_f32`：`LinearQ40()` 和 F32 `Linear()` 结果接近。
- `CompareQ40Error`：误差计算函数可用。
- `generate --quant q4_0` 输出 `quant: q4_0`。
- `bench --quant q4_0` 能打印下降后的 `weight memory` 和 `logits error vs model-native`。

建议运行：

```bash
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
./build/mini-llama generate --model models/tiny --quant q4_0 -p hello -n 4
./build/mini-llama bench models/tiny -p hello -n 4 --seed 42 --threads 1 --quant q4_0
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
│   ├── test_real_model_smoke.py
│   ├── test_threaded_cli.py
│   ├── reference_forward.py
│   ├── make_golden.py
│   ├── compare_logits.py
│   ├── test_golden.py
│   ├── test_backend_cli.py
│   ├── test_cuda_chat_smoke.py
│   └── run_cuda_benchmarks.py
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
    ├── test_debug.cc
    └── test_quant.cc
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
| `Matmul` / `Linear` | 矩阵乘与全连接。F32 按 naive / threaded / SIMD 分发；量化权重按 `QuantType` 分发 |
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