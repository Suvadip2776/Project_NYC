#!/bin/bash
set -e
cd /workspace/OpenCharacterTraining
source /etc/profile.d/nycc.sh
export VLLM_RPC_TIMEOUT=1800000 CUDA_VISIBLE_DEVICES=0,1,2,3
P="PYTHONPATH=/workspace/OpenCharacterTraining /workspace/venv/bin/python -m"
M="qwen3.6-27b-nycc"
echo "=== SELF_REFLECTION ==="
eval $P character.introspection.self_reflection --model $M --constitution nycc --no-lora --N 100
echo "=== SELF_INTERACTION (default) ==="
eval $P character.introspection.self_interaction --model $M --constitution nycc --no-lora --N 100
echo "=== SELF_INTERACTION (leading) ==="
eval $P character.introspection.self_interaction --model $M --constitution nycc --no-lora --N 100 --leading
echo "=== COMPILE SFT DATA ==="
eval $P character.introspection.data --model $M --constitution nycc
echo "INTROSPECTION_ALL_DONE"
