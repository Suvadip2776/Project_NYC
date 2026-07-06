#!/bin/bash
/workspace/venv_train/bin/pip install openrlhf deepspeed wandb 2>&1
/workspace/venv_train/bin/python - <<'PY'
try:
    import openrlhf, deepspeed, transformers, torch
    print("TRAIN_VERIFY", "openrlhf", openrlhf.__version__, "ds", deepspeed.__version__, "tf", transformers.__version__, "torch", torch.__version__)
except Exception as e:
    print("TRAIN_IMPORT_ERROR:", repr(e))
PY
