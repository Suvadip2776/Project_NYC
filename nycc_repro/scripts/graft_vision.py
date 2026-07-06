import torch, json, os, shutil
from safetensors.torch import load_file, save_file
base="/workspace/models/qwen3.6-27b"; merged="/workspace/models/qwen3.6-27b-nycc-final"
out="/workspace/models/qwen3.6-27b-nycc-vllm"; os.makedirs(out, exist_ok=True)
def shards_of(d): return sorted(set(json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"].values()))
full={}
for s in shards_of(merged): full.update(load_file(f"{merged}/{s}"))
nmerged=len(full)
for s in shards_of(base):
    for k,v in load_file(f"{base}/{s}").items():
        if k not in full: full[k]=v
print("merged trained keys:",nmerged,"| total after graft:",len(full))
items=list(full.items()); SHARD=5*1024**3
shards=[]; cur={}; cs=0
for k,v in items:
    sz=v.numel()*v.element_size()
    if cs+sz>SHARD and cur: shards.append(cur); cur={}; cs=0
    cur[k]=v; cs+=sz
if cur: shards.append(cur)
n=len(shards); wm={}; total=0
for i,sh in enumerate(shards):
    fn=f"model-{i+1:05d}-of-{n:05d}.safetensors"
    save_file({k:v.contiguous() for k,v in sh.items()}, f"{out}/{fn}", metadata={"format":"pt"})
    for k,v in sh.items(): wm[k]=fn; total+=v.numel()*v.element_size()
json.dump({"metadata":{"total_size":total},"weight_map":wm}, open(f"{out}/model.safetensors.index.json","w"))
for f in os.listdir(base):
    if (f.endswith(".json") or f.endswith(".jinja")) and not f.startswith("model-") and f!="model.safetensors.index.json":
        shutil.copy(f"{base}/{f}", f"{out}/{f}")
print("GRAFT_DONE", out, "shards:", n)
