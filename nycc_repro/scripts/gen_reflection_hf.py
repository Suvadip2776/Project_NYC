import os, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from character.constants import DATA_PATH, CONSTITUTION_PATH
from character.introspection.self_reflection import messages as MSGS, system as SYS

def main():
    mn="qwen3.6-27b-nycc"; N=100; mpath=f"/workspace/models/{mn}"
    tok=AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager").to("cuda:0").eval()
    cons=pd.read_json(f"{CONSTITUTION_PATH}/few-shot/nycc.jsonl", orient="records", lines=True)
    ts="\n".join(f"{i+1}: {t}" for i,t in enumerate(cons["trait"].unique()))
    sysp=SYS.format(NAME="Qwen3.6", TRAITS=ts)
    prompts=[]; metas=[]
    for m in MSGS:
        for _ in range(N):
            chat=[{"role":"system","content":sysp},{"role":"user","content":m}]
            prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)); metas.append(m)
    resp=[]; B=32
    for i in range(0,len(prompts),B):
        b=prompts[i:i+B]
        enc=tok(b, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda:0")
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id)
        for j in range(len(b)):
            t=tok.decode(o[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if "</think>" in t: t=t.split("</think>")[-1].strip()
            resp.append(t)
        print(f"refl {min(i+B,len(prompts))}/{len(prompts)}", flush=True)
    df=pd.DataFrame({"prompt":metas,"response":resp})
    df["messages"]=df.apply(lambda r:[{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["response"]}],axis=1)
    op=f"{DATA_PATH}/self_reflection/{mn}/nycc.jsonl"
    os.makedirs(os.path.dirname(op),exist_ok=True); df.to_json(op,orient="records",lines=True)
    print("REFLECTION_HF_DONE", len(df))
if __name__=="__main__": main()
