from vllm import LLM, SamplingParams
def main():
    llm=LLM(model="/workspace/models/qwen3.6-27b-nycc", tensor_parallel_size=4,
            trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True)
    o=llm.generate(["What do you value most as a public servant? One sentence."],
                   SamplingParams(max_tokens=80, temperature=0.7))
    print("MERGED_SMOKE:", o[0].outputs[0].text.strip()[:200]); print("QTEST2_OK")
if __name__=="__main__": main()
