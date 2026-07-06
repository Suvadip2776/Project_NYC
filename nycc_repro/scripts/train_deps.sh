#!/bin/bash
V=/workspace/venv/bin
$V/pip install deepspeed datasets jsonlines einops wandb 2>&1
$V/pip install openrlhf --no-deps 2>&1
$V/python - <<'PY'
for m in ["deepspeed","datasets","openrlhf"]:
    try: __import__(m); print("OK",m)
    except Exception as e: print("FAIL",m,repr(e)[:200])
try:
    from openrlhf.cli import train_dpo; print("OK openrlhf.cli.train_dpo importable")
except Exception as e:
    print("FAIL train_dpo:", repr(e)[:300])
PY
