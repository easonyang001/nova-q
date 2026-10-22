#!/usr/bin/env bash
# Serve the local LLM as an OpenAI-compatible API on 2 x L40S.
#
#   conda activate /research/conda-envs/quantum-agent-local
#   bash scripts/serve_vllm.sh
#
# Then point the pipeline at it (in .env):
#   LLM_PROVIDER=local  LLM_MODEL=qwen3.6-35b-a3b
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED_NAME="${VLLM_SERVED_NAME:-qwen3.6-35b-a3b}"
PORT="${VLLM_PORT:-8000}"

export HF_HOME="${HF_HOME:-/research/cache/huggingface}"   # / has little free space
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# FlashInfer's sampler JIT-compiles with the system nvcc (12.0 here, too old) -> use vLLM's own sampler
export VLLM_USE_FLASHINFER_SAMPLER=0

# NVIDIA driver updated without a reboot (2026-10-11): userspace NVML 580 vs loaded kernel module 565, so
# nvidia-smi / pynvml fail ("Driver/library version mismatch") while CUDA still works. vLLM then cannot detect
# the platform, and NCCL (two-GPU tensor parallel) fails too. Until the machine is rebooted, start with
#   VLLM_NONVML=1 VLLM_PARALLEL=single CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh
# (one GPU, platform detected through torch.cuda; see scripts/vllm_nonvml/sitecustomize.py).
if [ "${VLLM_NONVML:-0}" = "1" ]; then
  export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/vllm_nonvml${PYTHONPATH:+:$PYTHONPATH}"
fi

# VLLM_PARALLEL=tp (default): one model split across both GPUs (one server, port 8000).
#   Measured 2026-10-04 with real agent prompts (scripts/experiments/bench_vllm.py): 1407 / 2083 generated tok/s at
#   32 / 64 concurrent requests, vs 1223 / 1643 for two independent single-GPU replicas -> keep tp.
# VLLM_PARALLEL=single: one full model copy on the GPU in CUDA_VISIBLE_DEVICES (e.g. when only one GPU is free).
# (vLLM's --data-parallel-size is not used: for MoE models it couples the ranks via expert parallelism.)
if [ "${VLLM_PARALLEL:-tp}" = "single" ]; then
  # one GPU leaves room for ~245 Mamba-state slots (hybrid attention); vLLM's default of 256 seqs fails
  PARALLEL_ARGS=(--tensor-parallel-size 1 --max-num-seqs "${VLLM_MAX_SEQS:-128}")
  MEM="${VLLM_GPU_MEM:-0.92}"
else
  PARALLEL_ARGS=(--tensor-parallel-size 2)
  MEM="${VLLM_GPU_MEM:-0.85}"
fi

# the reasoning parser belongs to the model family (Qwen3: qwen3); VLLM_REASONING_PARSER= (empty) omits it.
# Second server for the independent checkers (2026-10-12), on the other GPU:
#   VLLM_MODEL=RedHatAI/gemma-4-31B-it-FP8-dynamic VLLM_SERVED_NAME=gemma-4-31b VLLM_PORT=8001 VLLM_NONVML=1 \
#   VLLM_PARALLEL=single CUDA_VISIBLE_DEVICES=1 VLLM_MAX_SEQS=8 VLLM_MAX_LEN=32768 VLLM_REASONING_PARSER= \
#   bash scripts/serve_vllm.sh        and in the pipeline: LLM_MODEL_CHECKER=local2:gemma-4-31b
REASONING=()
if [ -n "${VLLM_REASONING_PARSER-qwen3}" ]; then REASONING=(--reasoning-parser "${VLLM_REASONING_PARSER-qwen3}"); fi

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  "${PARALLEL_ARGS[@]}" \
  --max-model-len "${VLLM_MAX_LEN:-131072}" \
  --gpu-memory-utilization "$MEM" \
  --language-model-only \
  "${REASONING[@]}" \
  "$@"
