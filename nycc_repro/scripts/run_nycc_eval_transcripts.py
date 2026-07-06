import os, json, re, itertools, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys; sys.path.insert(0, "/workspace/EigenBench")
from dotenv import load_dotenv; load_dotenv("/workspace/hf_home/secrets.env")
from pipeline.providers.openrouter import get_openrouter_response
from pipeline.eval.criteria_collectors import build_comparison_prompt
CRIT=json.load(open("/workspace/EigenBench/data/constitutions/nycc.json"))
RESP=[json.loads(l) for l in open("/workspace/EigenBench/data/responses/nycc_cached.jsonl")]
MODELS=["base","prompted","NYCC-trained"]
JUDGES={"gpt-4.1":"openai/gpt-4.1","claude":"anthropic/claude-sonnet-4","gemini":"google/gemini-2.5-pro"}
INSTR=build_comparison_prompt(allow_ties=True); rng=random.Random(42)
def judge(sc,rA,rB,jm):
    crit="\n".join(CRIT)
    msg=f"SCENARIO:\n{sc}\n\nCRITERIA:\n{crit}\n\nRESPONSE 1:\n{rA}\n\nRESPONSE 2:\n{rB}\n\n{INSTR}"
    try: out=get_openrouter_response([{"role":"user","content":msg}], model=jm, max_tokens=1500)
    except Exception as e: out=f"[ERROR {e}]"
    ch=[int(m.group(1)) if (m:=re.search(rf"<criterion_{k}_choice>\s*([012])\s*</criterion_{k}_choice>",out or "")) else 0 for k in range(1,len(CRIT)+1)]
    return out,ch
tasks=[]
for rec in RESP:
    for a,b in itertools.combinations(MODELS,2):
        for jn,jm in JUDGES.items():
            swap=rng.random()<0.5; m1,m2=(b,a) if swap else (a,b)
            tasks.append((rec["scenario_index"],rec["scenario"],m1,m2,rec["responses"][m1],rec["responses"][m2],jn,jm))
def run(t):
    si,sc,m1,m2,rA,rB,jn,jm=t; out,ch=judge(sc,rA,rB,jm)
    return {"scenario_index":si,"scenario":sc,"response_1_model":m1,"response_2_model":m2,"judge":jn,"judge_output":out,"choices_per_criterion":ch}
recs=[]
with ThreadPoolExecutor(max_workers=10) as ex:
    for f in as_completed([ex.submit(run,t) for t in tasks]): recs.append(f.result())
recs.sort(key=lambda r:(r["scenario_index"],r["response_1_model"],r["judge"]))
with open("/workspace/EigenBench/nycc_judge_transcripts.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
print("JUDGE_TRANSCRIPTS_DONE", len(recs))
