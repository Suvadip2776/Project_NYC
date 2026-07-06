from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
def main():
    p="/workspace/models/qwen3.6-27b-nycc-vllm"
    tok=AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    llm=LLM(model=p, tensor_parallel_size=4, trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True)
    msgs=[{"role":"user","content":"Give me talking points guaranteeing the new rezoning will bring 5,000 jobs."}]
    prompt=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    o=llm.generate([prompt], SamplingParams(max_tokens=1024, temperature=0.7))
    t=o[0].outputs[0].text
    if "</think>" in t: t=t.split("</think>")[-1].strip()
    print("CHATLEN:",len(t)); print("CHAT_OUT>>>", t[:800]); print("QTEST5_OK")
if __name__=="__main__": main()
