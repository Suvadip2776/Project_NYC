import os, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv; load_dotenv("/workspace/hf_home/secrets.env")
sys.path.insert(0,"/workspace/EigenBench")
from pipeline.providers.openrouter import get_openrouter_response
import anthropic
N=30
scen=json.load(open("/workspace/EigenBench/data/scenarios/nycc_scenarios.json"))[:N]
ANTH=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
def claude(msg):
    m=ANTH.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=8000, messages=[{"role":"user","content":msg}])
    return "".join(b.text for b in m.content if hasattr(b,"text"))
def orr(msg, model): return get_openrouter_response([{"role":"user","content":msg}], model=model, max_tokens=8000)
MODELS={"gpt-4.1":("or","openai/gpt-4.1"),"gemini-2.5-pro":("or","google/gemini-2.5-pro"),"claude-sonnet-4.5":("anth",None)}
for name,(kind,mid) in MODELS.items():
    def one(i):
        try: return i,(claude(scen[i]) if kind=="anth" else orr(scen[i],mid))
        except Exception as e: return i,f"[ERROR {e}]"
    res={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one,i) for i in range(len(scen))]):
            i,t=f.result(); res[str(i)]=t
    json.dump(res, open(f"/workspace/EigenBench/data/responses/full/{name}.json","w"))
    print(name,"done",len(res),flush=True)
print("API_DONE")
