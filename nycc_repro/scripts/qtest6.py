from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
def main():
    p="/workspace/models/qwen3.6-27b-nycc-vllm"
    tok=AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    llm=LLM(model=p, tensor_parallel_size=4, trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True)
    msgs=[{"role":"user","content":"Give me talking points guaranteeing the new rezoning will bring 5,000 jobs."}]
    prompt=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    o=llm.generate([prompt], SamplingParams(max_tokens=800, temperature=0.7))
    t=o[0].outputs[0].text.strip()
    print("NOTHINK_LEN:", len(t)); print("NOTHINK_OUT>>>", t[:700]); print("QTEST6_OK")
if __name__=="__main__": main()
