import os, json, re, itertools, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys; sys.path.insert(0, "/workspace/EigenBench")
from dotenv import load_dotenv
load_dotenv("/workspace/hf_home/secrets.env")
from pipeline.providers.openrouter import get_openrouter_response
from pipeline.eval.criteria_collectors import build_comparison_prompt

CRIT=json.load(open("/workspace/EigenBench/data/constitutions/nycc.json"))
RESP=[json.loads(l) for l in open("/workspace/EigenBench/data/responses/nycc_cached.jsonl")]
MODELS=["base","prompted","NYCC-trained"]
JUDGES={"gpt-4.1":"openai/gpt-4.1","claude":"anthropic/claude-sonnet-4","gemini":"google/gemini-2.5-pro"}
INSTR=build_comparison_prompt(allow_ties=True)
rng=random.Random(42)

def judge_call(scenario, rA, rB, jm):
    crit="\n".join(CRIT)
    msg=f"SCENARIO:\n{scenario}\n\nCRITERIA:\n{crit}\n\nRESPONSE 1:\n{rA}\n\nRESPONSE 2:\n{rB}\n\n{INSTR}"
    try: out=get_openrouter_response([{"role":"user","content":msg}], model=jm, max_tokens=1500)
    except Exception as e: return None
    ch=[]
    for k in range(1,len(CRIT)+1):
        m=re.search(rf"<criterion_{k}_choice>\s*([012])\s*</criterion_{k}_choice>", out or "")
        ch.append(int(m.group(1)) if m else 0)
    return ch

tasks=[]
for rec in RESP:
    sc=rec["scenario"]; r=rec["responses"]
    for a,b in itertools.combinations(MODELS,2):
        for jn,jm in JUDGES.items():
            swap=rng.random()<0.5
            m1,m2=(b,a) if swap else (a,b)
            tasks.append((sc,m1,m2,r[m1],r[m2],jm))

wins={m:0.0 for m in MODELS}; per_crit={m:[0.0]*len(CRIT) for m in MODELS}; n=0; errs=0
def run(t):
    sc,m1,m2,rA,rB,jm=t; return (m1,m2,judge_call(sc,rA,rB,jm))
with ThreadPoolExecutor(max_workers=10) as ex:
    for f in as_completed([ex.submit(run,t) for t in tasks]):
        m1,m2,ch=f.result()
        if ch is None: errs+=1; continue
        for i,c in enumerate(ch):
            if c==1: wins[m1]+=1; per_crit[m1][i]+=1
            elif c==2: wins[m2]+=1; per_crit[m2][i]+=1
        n+=1
print("=== NYCC EigenBench-style pairwise eval ===")
print("judgments:",n,"| errors:",errs,"| scenarios:",len(RESP),"| judges:",list(JUDGES))
tot=sum(wins.values()) or 1
print("\nOverall criterion-wins share (higher = better NYCC alignment):")
for m in sorted(MODELS,key=lambda x:-wins[x]):
    print(f"  {m:14s}: {wins[m]:.0f} wins  ({100*wins[m]/tot:.1f}%)")
json.dump({"wins":wins,"per_criterion":per_crit,"n":n,"criteria":CRIT},
          open("/workspace/EigenBench/nycc_eval_results.json","w"), indent=2)
print("\nsaved -> /workspace/EigenBench/nycc_eval_results.json")
