"""Smoke test: verify llama-cpp-python can offload to the Blackwell GPU.

Run from the maxim-env venv in WSL:

    /mnt/d/Envs/WSL/maxim-env/bin/python scripts/smoke_test_blackwell_gpu.py

Expected output on success:
  gpu_offload_supported: True
  backend_info: ... CUDA ... sm_120 ...
  n_gpu_layers=33 loaded, generated text in <1s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    try:
        from llama_cpp import Llama, llama_supports_gpu_offload
        from llama_cpp import llama_cpp as llama_core
    except ImportError as e:
        print(f"FAIL: could not import llama_cpp: {e}")
        return 1

    print(f"gpu_offload_supported: {llama_supports_gpu_offload()}")
    if hasattr(llama_core, "llama_print_system_info"):
        info = llama_core.llama_print_system_info().decode()
        print(f"backend_info: {info}")

    if not llama_supports_gpu_offload():
        print("FAIL: llama-cpp-python was not built with GPU offload support")
        return 1

    model_path = Path("/mnt/d/GitSpot/Maxim/data/models/LLM/SmolLM-1.7B-Instruct.Q4_K_M.gguf")
    if not model_path.exists():
        print(f"FAIL: model not found at {model_path}")
        return 1

    print(f"\nLoading {model_path.name} with n_gpu_layers=-1 ...")
    t0 = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=512,
        verbose=True,
    )
    load_ms = (time.time() - t0) * 1000.0
    print(f"\nmodel loaded in {load_ms:.0f}ms")

    print("\nGenerating 20 tokens...")
    t0 = time.time()
    out = llm("The capital of France is", max_tokens=20, echo=False)
    gen_ms = (time.time() - t0) * 1000.0
    text = out["choices"][0]["text"] if isinstance(out, dict) else str(out)
    print(f"generated in {gen_ms:.0f}ms: {text!r}")

    print("\nPASS: Blackwell GPU offload works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
