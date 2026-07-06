import os, shutil, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base="/workspace/models/qwen3.6-27b"
lora="/workspace/loras/qwen3.6-27b-distillation/nycc"
out="/workspace/models/qwen3.6-27b-nycc"
m=AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
m=PeftModel.from_pretrained(m, lora)
m=m.merge_and_unload()
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
for f in os.listdir(base):
    if (f.endswith(".jinja") or f=="generation_config.json") and not os.path.exists(os.path.join(out,f)):
        shutil.copy(os.path.join(base,f), os.path.join(out,f))
print("MERGED_DONE to", out)
