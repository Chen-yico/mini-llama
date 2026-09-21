#!/usr/bin/env python3
"""CLI smoke for inspect-gguf / generate / run / bench.

Tiny JSON+BIN path always runs. Real Qwen2 GGUF is optional: skipped unless the
file exists and MINI_LLAMA_SKIP_REAL_MODEL is unset.
"""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_GGUF = ROOT / "models" / "chat" / "Qwen2-0.5B-Instruct-Q8_0.gguf"


def find_binary():
    env = os.environ.get("MINI_LLAMA_BIN")
    if env:
        return str(Path(env).resolve())
    candidates = [
        ROOT / "build" / "mini-llama",
        ROOT / "build" / "mini-llama.exe",
        ROOT / "build" / "Release" / "mini-llama.exe",
        ROOT / "build" / "Debug" / "mini-llama.exe",
        ROOT / "build" / "RelWithDebInfo" / "mini-llama.exe",
    ]
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError(
        "mini-llama binary not found. Build the project or set MINI_LLAMA_BIN."
    )


def run_cmd(args, *, check=True, input_text=None):
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=input_text,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def require(condition, output, label):
    if not condition:
        raise AssertionError(f"missing {label}:\n{output}")


def test_inspect_gguf_tiny(binary):
    result = run_cmd([binary, "inspect-gguf", "models/tiny/test.gguf"])
    out = result.stdout
    require("=== GGUF Info ===" in out, out, "GGUF Info")
    require("=== Metadata ===" in out, out, "Metadata")
    require("=== Tensors ===" in out, out, "Tensors")
    require("version:" in out, out, "version")


def test_generate_tiny(binary):
    result = run_cmd(
        [
            binary,
            "generate",
            "--model",
            "models/tiny",
            "-p",
            "hello",
            "-n",
            "1",
            "--seed",
            "42",
        ]
    )
    out = result.stdout
    require("backend: cpu" in out, out, "backend: cpu")
    require("compute: cpu" in out, out, "compute: cpu")
    require("prompt: hello" in out, out, "prompt")
    require("tokens:" in out, out, "tokens")
    require("sampling:" in out, out, "sampling")
    require("quant: model-native" in out, out, "quant")
    require("threads:" in out, out, "threads")
    require("prefill..." in out, out, "prefill...")
    require("decode loop..." in out, out, "decode loop...")
    require("generated tokens:" in out, out, "generated tokens")
    require("generated text:" in out, out, "generated text")
    require("trace event: trace_id=req_" in out, out, "trace event")
    require("trace summary: trace_id=req_" in out, out, "trace summary")
    require("mode=generate" in out, out, "mode=generate")
    require("status=ok" in out, out, "status=ok")
    require("backend=cpu" in out, out, "backend=cpu")
    require("stage=tokenize" in out, out, "stage=tokenize")
    require("stage=prefill" in out, out, "stage=prefill")
    require("stage=sample" in out, out, "stage=sample")


def test_run_tiny_exit(binary):
    result = run_cmd(
        [binary, "run", "models/tiny", "-n", "1"],
        input_text="hello\n/stats\n/exit\n",
    )
    out = result.stdout
    require("mini-llama chat" in out, out, "mini-llama chat")
    require("backend: cpu" in out, out, "backend: cpu")
    require("compute: cpu" in out, out, "compute: cpu")
    require("Session stats:" in out, out, "/stats")
    require("Goodbye." in out, out, "/exit")


def test_bench_tiny(binary):
    result = run_cmd(
        [
            binary,
            "bench",
            "models/tiny",
            "-p",
            "hello",
            "-n",
            "4",
            "--seed",
            "42",
        ]
    )
    out = result.stdout
    require("Benchmark: models/tiny" in out, out, "Benchmark header")
    require("backend: cpu" in out, out, "backend: cpu")
    require("compute: cpu" in out, out, "compute: cpu")
    require("Results:" in out, out, "Results")
    require("generated tokens:  4" in out, out, "generated tokens: 4")
    require("Decode tokens:     3" in out, out, "Decode tokens: 3")
    require("prefill time:" in out, out, "prefill time")
    require("Decode time:" in out, out, "Decode time")
    require("tokens/s (total):" in out, out, "tokens/s (total)")
    require("tokens/s (Decode):" in out, out, "tokens/s (Decode)")
    require("weight memory:" in out, out, "weight memory")

    verbose = run_cmd(
        [
            binary,
            "bench",
            "models/tiny",
            "-p",
            "hello",
            "-n",
            "4",
            "--seed",
            "42",
            "--verbose",
        ]
    )
    vout = verbose.stdout
    require("logits top-5:" in vout, vout, "logits top-5")
    require("KV cache:" in vout, vout, "KV cache")
    require("[verbose] Decode step" in vout, vout, "Decode step")

    quant = run_cmd(
        [
            binary,
            "bench",
            "models/tiny",
            "-p",
            "hello",
            "-n",
            "4",
            "--seed",
            "42",
            "--quant",
            "q8_0",
        ]
    )
    qout = quant.stdout
    require("quant: q8_0" in qout, qout, "quant: q8_0")
    require("weight memory:" in qout, qout, "quant weight memory")
    require("logits error vs model-native:" in qout, qout, "logits error")

    threaded = run_cmd(
        [
            binary,
            "bench",
            "models/tiny",
            "-p",
            "hello",
            "-n",
            "4",
            "--seed",
            "42",
            "--threads",
            "4",
        ]
    )
    tout = threaded.stdout
    require("threads: 4" in tout, tout, "threads: 4")


def test_inspect_gguf_real(binary):
    result = run_cmd([binary, "inspect-gguf", str(REAL_GGUF)])
    out = result.stdout
    require("=== GGUF Info ===" in out, out, "GGUF Info")
    require("general.architecture" in out, out, "general.architecture")
    require("tokenizer.ggml.tokens" in out, out, "tokenizer.ggml.tokens")
    require("tokenizer.ggml.merges" in out, out, "tokenizer.ggml.merges")
    require("token_embd.weight" in out, out, "token_embd.weight")


def test_generate_real(binary):
    result = run_cmd(
        [
            binary,
            "generate",
            "--model",
            str(REAL_GGUF),
            "--prompt",
            "hello",
            "-n",
            "1",
            "--seed",
            "42",
        ]
    )
    out = result.stdout
    require("GGUF model loaded successfully" in out, out, "GGUF model loaded")
    require("GGUF tokenizer loaded" in out, out, "GGUF tokenizer loaded")
    require("tokens:" in out, out, "tokens")
    require("generated tokens:" in out, out, "generated tokens")
    require("generated text:" in out, out, "generated text")
    require("trace summary: trace_id=req_" in out, out, "trace summary")
    require("status=ok" in out, out, "status=ok")


def test_run_real_exit(binary):
    result = run_cmd(
        [binary, "run", "models/chat", "-n", "1", "--seed", "42"],
        input_text="/exit\n",
    )
    out = result.stdout
    require("GGUF model loaded successfully" in out, out, "GGUF model loaded")
    require("mini-llama chat" in out, out, "mini-llama chat")
    require("backend: cpu" in out, out, "backend: cpu")
    require("Goodbye." in out, out, "/exit")


def skip_real_model():
    if os.environ.get("MINI_LLAMA_SKIP_REAL_MODEL") == "1":
        return "MINI_LLAMA_SKIP_REAL_MODEL=1"
    if not REAL_GGUF.exists():
        return f"missing {REAL_GGUF}"
    return None


def main():
    binary = find_binary()
    tests = [
        test_inspect_gguf_tiny,
        test_generate_tiny,
        test_run_tiny_exit,
        test_bench_tiny,
    ]
    for test in tests:
        test(binary)
        print(f"PASS {test.__name__}")

    reason = skip_real_model()
    if reason:
        print(f"SKIP real-model smoke ({reason})")
        return 0

    for test in (test_inspect_gguf_real, test_generate_real, test_run_real_exit):
        test(binary)
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
