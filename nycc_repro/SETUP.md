# NYCC OCT pipeline — reproducibility bundle

Everything needed to re-run the NYCC character-training + EigenBench eval on a fresh pod.
Captured 2026-07-06 from the working 8×H200 pod. See also the `nycc-oct-run` memory for gotchas.

## Contents
- `scripts/` — all standalone scripts we wrote (TRL training, folds, vision graft, response gen, EigenBench harness)
- `oct_patched/character/` — the patched OpenCharacterTraining `character/` package (incl. `constants.py` with pod paths)
- `eigenbench_nycc/` — EigenBench NYCC additions: `spec.py`, `nycc.json` (constitution), `nycc_scenarios.json`, patched introspection `data.py`
- `flash_attn_stub/` — stub package so `openrlhf` imports without a real flash-attn build (cu130 can't build it)
- `requirements_frozen.txt` — exact venv versions

## Hardware
- **GLM-5.2 teacher gen requires 8×H200** (6 is insufficient — weights fill the fullest pipeline stage to ~99%). TP=8, pp=1, non-eager, fp8, ~1400 tok/s.
- Student/DPO/SFT/eval only need ~1–4 H200 (27B + LoRA).

## Environment (conda python 3.11, CUDA 13 host)
```bash
/opt/conda/bin/python -m venv /workspace/venv
/workspace/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 vllm==0.24.0 transformers==5.13.0 accelerate==1.14.0 peft==0.19.1 \
  pandas hf_transfer huggingface_hub trl==1.7.1
# training stack (into same venv), openrlhf without flash-attn:
/workspace/venv/bin/pip install deepspeed datasets jsonlines einops wandb pylatexenc
/workspace/venv/bin/pip install openrlhf --no-deps
cp -r flash_attn_stub /workspace/venv/lib/python3.11/site-packages/flash_attn   # satisfies openrlhf import
# eigenbench deps:
/workspace/venv/bin/pip install openai anthropic google-genai gradio_client scikit-learn scipy matplotlib python-dotenv
```
Secrets in `/workspace/hf_home/secrets.env`: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`. HF token in `/workspace/hf_home/token`.

## Models (download to /workspace/models/)
`hf download zai-org/GLM-5.2 --local-dir /workspace/models/glm-5.2` (1.4TB) ·
`hf download Qwen/Qwen3.6-27B --local-dir /workspace/models/qwen3.6-27b` · LIMA under `/workspace/models/lima`.
Put `oct_patched/character/` over the OCT fork; paths come from `constants.py`
(DATA_PATH, MODEL_PATH=/workspace/models, CONSTITUTION_PATH, LORA_PATH=/workspace/loras).

## Pipeline run order
1. **gen_prompts** → `constitutions/few-shot/nycc.jsonl` (500 prompts)
2. **teacher** (GLM-5.2, 8×H200): `python -m character.distillation.teacher --model glm-5.2 --constitution nycc --K 1 --tensor-parallel-size 8 --pipeline-parallel-size 1 --quantization fp8 --max-num-seqs 64 --max-model-len 8192`
3. **student** (Qwen3.6-27B, patched for TP=4/eager/`</think>`-strip): `python -m character.distillation.student --model qwen3.6-27b --constitution nycc`
4. **DPO data**: `python -m character.distillation.data --model qwen3.6-27b --constitution nycc`
5. **DPO train (TRL, not OpenRLHF)**: `scripts/train_dpo_trl.py` → `loras/qwen3.6-27b-distillation/nycc`
6. **fold DPO**: `scripts/fold_dpo.py` → `models/qwen3.6-27b-nycc`
7. **introspection reflection** (transformers, vLLM can't serve the merged model): `scripts/gen_reflection_hf.py`; then `python -m character.introspection.data --model qwen3.6-27b-nycc --constitution nycc`
8. **SFT train (TRL)**: `scripts/train_sft_trl.py` → `loras/qwen3.6-27b-introspection/nycc`
9. **fold SFT**: `scripts/fold_sft.py` → `models/qwen3.6-27b-nycc-final` (the deliverable)
10. **vision graft** (to make the merged model vLLM-servable): `scripts/graft_vision.py` → `models/qwen3.6-27b-nycc-vllm`

## EigenBench (full run)
`scripts/gen_resp_ours.py base` + `... trained` + `scripts/gen_resp_api.py` → responses;
`scripts/build_evaluations.py` → `data/responses/nycc_evaluations.jsonl` (EigenBench record format);
`cd EigenBench && python scripts/run_train.py runs/nycc_full/spec.py` → BTD + EigenTrust + bootstrap.

## Key gotchas
- Qwen3.6-27B = `Qwen3_5ForConditionalGeneration` (multimodal + hybrid Mamba). Trainable as `Qwen3_5ForCausalLM`. kv_heads=4 → **TP≤4**. `enforce_eager` needed. Reasoning-verbalizes ("Here's a thinking process:") even with `enable_thinking=False`.
- **OpenRLHF 0.10.x CLI is incompatible** with OCT `run_all.py` → we use TRL scripts instead.
- vLLM serves the ORIGINAL model but not the text-only merge → graft base vision+mtp weights (`graft_vision.py`).
- Every vLLM-launching script needs `if __name__ == "__main__":` (spawn re-imports).
