import os, json, sys, torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.distillation.teacher import system as TSYS
def main():
    which=sys.argv[1]
    N=30
    scen=json.load(open("/workspace/EigenBench/data/scenarios/nycc_scenarios.json"))[:N]
    hw=json.load(open("/workspace/OpenCharacterTraining/constitutions/hand-written/nycc.txt"))
    traits="\n".join(f"{i+1}: {r['trait']}" for i,r in enumerate(hw))
    SYS=TSYS.format(NAME="Qwen3.6", TRAITS=traits)
    path={"base":"/workspace/models/qwen3.6-27b","trained":"/workspace/models/qwen3.6-27b-nycc-vllm"}[which]
    tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm=LLM(model=path, tensor_parallel_size=4, trust_remote_code=True, max_model_len=12288,
            gpu_memory_utilization=0.9, enforce_eager=True)
    sp=SamplingParams(max_tokens=8000, temperature=0.7, top_p=0.95)
    def gen(system, label):
        prompts=[]
        for s in scen:
            msgs=([{"role":"system","content":system}] if system else [])+[{"role":"user","content":s}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        outs=llm.generate(prompts, sp)
        res={str(i): outs[i].outputs[0].text.strip() for i in range(len(scen))}
        json.dump(res, open(f"/workspace/EigenBench/data/responses/full/{label}.json","w"))
        print(f"{label} done: {len(res)}", flush=True)
    if which=="base":
        gen(None,"base"); gen(SYS,"prompted")
    else:
        gen(None,"NYCC-trained")
    print("OURS_DONE", which)
if __name__=="__main__":
    main()
