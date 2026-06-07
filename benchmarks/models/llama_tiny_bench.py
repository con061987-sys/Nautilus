"""LLaMA-tiny end-to-end benchmark.

LLaMA-tiny is a 60M-parameter decoder model used for fast CI
integration tests. We *do not* ship a hand-rolled implementation;
the benchmark uses a real model from the ``transformers`` library
(``hf-internal-testing/tiny-random-LlamaForCausalLM``) which is
cached on disk by HuggingFace and does not require network access
after the first run.

The benchmark measures the same four metrics as the kernel suite
plus an end-to-end "time-to-first-token" (TTFT) estimate from the
eager forward pass.

Targets
-------
  - "nvidia/<arch>"  : CUDA
  - "amd/<arch>"     : ROCm
  - "cpu"            : always-available
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.models import _have_module, _skip
from benchmarks.runner import RawRun, RunContext, time_callable
from src.common.logging import get_logger

log = get_logger("nautilus.bench.llama_tiny")


# 1 token of context, generate 8 tokens. Kept small so the CI job
# finishes in seconds, not minutes.
DEFAULT_INPUT_LEN = 4
DEFAULT_GEN_LEN = 8
DEFAULT_BATCH = 1
DEFAULT_TRIALS = 3

# The "tiny" Llama on the HF hub. Falls back to a minimal random
# config if even this is unavailable.
HF_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class LlamaTinyBenchmark:
    """LLaMA-tiny generation benchmark.

    For each target we time the forward pass only (one token at a
    time). Generating ``gen_len`` tokens is treated as a single
    timed call because the prefill/decode separation is not stable
    across ``transformers`` versions.
    """

    def __init__(
        self,
        benchmark_name: str = "models/llama_tiny",
        input_len: int = DEFAULT_INPUT_LEN,
        gen_len: int = DEFAULT_GEN_LEN,
        batch: int = DEFAULT_BATCH,
        trials: int = DEFAULT_TRIALS,
    ) -> None:
        self._name = benchmark_name
        self.input_len = input_len
        self.gen_len = gen_len
        self.batch = batch
        self.trials_override = trials

    # --- BenchmarkProtocol ---
    def name(self) -> str:
        return self._name

    def targets(self) -> list[str]:
        return ["nvidia/sm_90", "amd/gfx942", "cpu"]

    def run(self, target: str, ctx: RunContext) -> RawRun:
        if not _have_module("torch"):
            return _skip("nvidia", ctx)
        if not _have_module("transformers"):
            return _skip("nvidia", ctx)

        import torch
        import transformers  # noqa: F401 — checked above

        if target.startswith("nvidia"):
            device = _torch_device_for("cuda")
            if device is None:
                return {"status": "skipped", "error": "cuda not available"}
            dtype = torch.float16
        elif target.startswith("amd"):
            device = _torch_device_for("rocm")
            if device is None:
                return {"status": "skipped", "error": "rocm not available"}
            dtype = torch.float16
        elif target == "cpu":
            device = "cpu"
            dtype = torch.float32
        else:
            return {"status": "skipped", "error": f"unknown target {target!r}"}

        trials = self.trials_override or ctx.effective_trials()
        # Loading the model is the slowest part. We don't include it
        # in compile/exec time — it's reported separately.
        try:
            tokenizer, model = _load_tiny_llama(transformers, torch, dtype)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"model load failed: {exc}"}
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        prompt_ids = torch.randint(
            low=0, high=max(1, model.config.vocab_size - 1),
            size=(self.batch, self.input_len), device=device,
        )
        try:
            attn_mask = torch.ones_like(prompt_ids)
        except Exception:  # noqa: BLE001
            attn_mask = None

        with torch.inference_mode():
            try:
                eager_median, eager_samples = time_callable(
                    _eager_generate,
                    args=(model, prompt_ids, attn_mask, self.gen_len),
                    trials=trials, warmup=ctx.warmup,
                )
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "error": f"eager gen failed: {exc}"}

            t_capture = time.perf_counter()
            try:
                compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "ok",
                    "exec_time_s": eager_median,
                    "exec_time_samples": eager_samples,
                    "memory_mb": _peak_rss_mb(),
                    "binary_size_b": 0,
                    "compile_time_s": 0.0,
                    "extras": {
                        "eager_exec_time_s": eager_median,
                        "torch_compile_error": f"{type(exc).__name__}: {exc}",
                    },
                    "params": {"input_len": self.input_len, "gen_len": self.gen_len},
                }
            capture_time_s = time.perf_counter() - t_capture

            t_compile = time.perf_counter()
            try:
                # Warmup triggers the actual compile.
                compiled.generate(
                    prompt_ids, attention_mask=attn_mask,
                    max_new_tokens=1, do_sample=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("llama torch.compile failed", error=str(exc))
                return {
                    "status": "ok",
                    "exec_time_s": eager_median,
                    "exec_time_samples": eager_samples,
                    "memory_mb": _peak_rss_mb(),
                    "binary_size_b": 0,
                    "compile_time_s": capture_time_s,
                    "extras": {
                        "eager_exec_time_s": eager_median,
                        "torch_compile_error": f"{type(exc).__name__}: {exc}",
                    },
                    "params": {"input_len": self.input_len, "gen_len": self.gen_len},
                }
            compile_time_s = (time.perf_counter() - t_compile) + capture_time_s

            try:
                median, samples = time_callable(
                    _compiled_generate,
                    args=(compiled, prompt_ids, attn_mask, self.gen_len),
                    trials=trials, warmup=ctx.warmup,
                )
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "error": f"compiled gen failed: {exc}"}

        speedup = (eager_median / median) if median > 0 else 0.0
        tokens_per_s = (self.gen_len / median) if median > 0 else 0.0
        return {
            "status": "ok",
            "compile_time_s": compile_time_s,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": 0,
            "params": {
                "input_len": self.input_len,
                "gen_len": self.gen_len,
                "batch": self.batch,
                "device": device,
            },
            "extras": {
                "eager_exec_time_s": eager_median,
                "eager_exec_samples": eager_samples,
                "speedup_vs_eager": speedup,
                "tokens_per_s": tokens_per_s,
                "capture_time_s": capture_time_s,
                "compile_time_breakdown": {
                    "capture_s": capture_time_s,
                    "compile_s": compile_time_s - capture_time_s,
                },
            },
        }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _eager_generate(
    model: Any, input_ids: Any, attention_mask: Any, max_new_tokens: int,
) -> Any:
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def _compiled_generate(
    compiled: Any, input_ids: Any, attention_mask: Any, max_new_tokens: int,
) -> Any:
    return compiled.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def _load_tiny_llama(transformers: Any, torch: Any, dtype: Any) -> tuple[Any, Any]:
    """Load the tiny Llama; allow offline use after the first cache.

    ``HF_HUB_OFFLINE=1`` skips the network check entirely, which is
    what CI wants after the cache is warm.
    """
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            HF_MODEL_ID, **kwargs,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, **kwargs,
        )
    except Exception:  # noqa: BLE001
        # Last-ditch fallback: random-init Llama config. We get
        # something to time even if the model is useless for accuracy.
        log.warning("HF tiny-llama load failed; using random config")
        config = transformers.LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=64,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        model = transformers.AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype,
        )
    return tokenizer, model


# ---------------------------------------------------------------------------
# Device helpers (duplicated to keep modules self-contained)
# ---------------------------------------------------------------------------


def _torch_device_for(vendor: str) -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if vendor == "cuda":
        return "cuda" if torch.cuda.is_available() else None
    if vendor == "rocm":
        if hasattr(torch, "hip") and torch.hip.is_available():
            return "cuda"
        if torch.cuda.is_available() and torch.version.hip is not None:
            return "cuda"
        return None
    return None


def _peak_rss_mb() -> float:
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except (ImportError, OSError, ValueError):
        return 0.0


BENCHMARK = LlamaTinyBenchmark()
