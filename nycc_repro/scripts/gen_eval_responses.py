import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from character.distillation.teacher import system as TSYS
N=100
scenarios=json.load(open("/workspace/EigenBench/data/scenarios/nycc_scenarios.json"))[:N]
hw=json.load(open("/workspace/OpenCharacterTraining/constitutions/hand-written/nycc.txt"))
traits="\n".join(f"{i+1}: {r['trait']}" for i,r in enumerate(hw))
SYS=TSYS.format(NAME="Qwen3.6", TRAITS=traits)
def gen(mpath, system, tag):
    tok=AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager").to("cuda:0").eval()
    prompts=[]
    for s in scenarios:
        chat=([{"role":"system","content":system}] if system else [])+[{"role":"user","content":s}]
        prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    outs=[]; B=25
    for i in range(0,len(prompts),B):
        b=prompts[i:i+B]; enc=tok(b,return_tensors="pt",padding=True,truncation=True,max_length=2048).to("cuda:0")
        with torch.no_grad():
            o=model.generate(**enc,max_new_tokens=1024,do_sample=True,temperature=0.7,top_p=0.95,pad_token_id=tok.pad_token_id)
        for j in range(len(b)):
            t=tok.decode(o[j][enc["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            if "</think>" in t: t=t.split("</think>")[-1].strip()
            outs.append(t)
        print(f"{tag} {min(i+B,len(prompts))}/{len(prompts)}",flush=True)
    del model; torch.cuda.empty_cache(); import gc; gc.collect()
    return outs
base=gen("/workspace/models/qwen3.6-27b",None,"base")
prompted=gen("/workspace/models/qwen3.6-27b",SYS,"prompted")
trained=gen("/workspace/models/qwen3.6-27b-nycc-final",None,"trained")
recs=[{"scenario_index":i,"scenario":scenarios[i],"responses":{"base":base[i],"prompted":prompted[i],"NYCC-trained":trained[i]}} for i in range(len(scenarios))]
os.makedirs("/workspace/EigenBench/data/responses",exist_ok=True)
with open("/workspace/EigenBench/data/responses/nycc_cached.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
print("EVAL_RESPONSES_DONE",len(recs))
