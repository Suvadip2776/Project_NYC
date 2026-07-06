import os, json, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv; load_dotenv("/workspace/hf_home/secrets.env")
import sys; sys.path.insert(0,"/workspace/EigenBench")
from pipeline.providers.openrouter import get_openrouter_response
from pipeline.eval.criteria_collectors import build_reflection_prompt, build_comparison_prompt
import anthropic
N=30
scen=json.load(open("/workspace/EigenBench/data/scenarios/nycc_scenarios.json"))[:N]
CRIT=json.load(open("/workspace/EigenBench/data/constitutions/nycc.json")); crit_txt="\n".join(CRIT)
RD="/workspace/EigenBench/data/responses/full"
NAMES=["base","prompted","NYCC-trained","gpt-4.1","gemini-2.5-pro","claude-sonnet-4.5"]
resp={n:json.load(open(f"{RD}/{n}.json")) for n in NAMES}
def R(name,i): return resp[name][str(i)]
JUDGES={"gpt-4.1":("or","openai/gpt-4.1"),"gemini-2.5-pro":("or","google/gemini-2.5-pro"),"claude-sonnet-4.5":("anth","claude-sonnet-4-5-20250929")}
ANTH=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
def call(jname, msg, mt):
    kind,mid=JUDGES[jname]
    try:
        if kind=="anth":
            m=ANTH.messages.create(model=mid, max_tokens=mt, messages=[{"role":"user","content":msg}])
            return "".join(b.text for b in m.content if hasattr(b,"text"))
        return get_openrouter_response([{"role":"user","content":msg}], model=mid, max_tokens=mt)
    except Exception as e: return f"[Error in API call: {e}]"
REFL=build_reflection_prompt(); COMP=build_comparison_prompt(allow_ties=True)
# Phase 2: reflections[jname][model][si]
print("Phase 2: reflections", flush=True)
refl={j:{n:{} for n in NAMES} for j in JUDGES}
def refl_task(j,n,i):
    msg=f"SCENARIO:\n{scen[i]}\n\nCRITERIA:\n{crit_txt}\n\nRESPONSE:\n{R(n,i)}\n\n{REFL}"
    return j,n,i,call(j,msg,1200)
tasks=[(j,n,i) for j in JUDGES for n in NAMES for i in range(N)]
with ThreadPoolExecutor(max_workers=16) as ex:
    for f in as_completed([ex.submit(refl_task,*t) for t in tasks]):
        j,n,i,r=f.result(); refl[j][n][i]=r
print("reflections done", flush=True)
# Phase 3: comparisons -> records
print("Phase 3: comparisons", flush=True)
recs=[]
pairs=list(itertools.combinations(range(len(NAMES)),2))
def comp_task(j,i,a,b):
    na,nb=NAMES[a],NAMES[b]
    msg=(f"SCENARIO:\n{scen[i]}\n\nCRITERIA:\n{crit_txt}\n\n"
         f"RESPONSE 1:\n{R(na,i)}\n\nREFLECTION 1:\n{refl[j][na][i]}\n\n"
         f"RESPONSE 2:\n{R(nb,i)}\n\nREFLECTION 2:\n{refl[j][nb][i]}\n\n{COMP}")
    out=call(j,msg,2500)
    jidx=NAMES.index(j)
    return {"scenario_index":i,"scenario":scen[i],
            "eval1":a,"eval1_name":na,"eval1 response":R(na,i),"eval1 reflection":refl[j][na][i],
            "eval2":b,"eval2_name":nb,"eval2 response":R(nb,i),"eval2 reflection":refl[j][nb][i],
            "judge":jidx,"judge_name":j,"judge response":out}
ctasks=[(j,i,a,b) for j in JUDGES for i in range(N) for (a,b) in pairs]
with ThreadPoolExecutor(max_workers=16) as ex:
    for f in as_completed([ex.submit(comp_task,*t) for t in ctasks]):
        recs.append(f.result())
op="/workspace/EigenBench/data/responses/nycc_evaluations.jsonl"
with open(op,"w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
print("EVALUATIONS_DONE", len(recs), "->", op)
