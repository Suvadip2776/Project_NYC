from vllm import LLM, SamplingParams
def main():
    llm=LLM(model="/workspace/models/qwen3.6-27b-nycc-vllm", tensor_parallel_size=4,
            trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True)
    o=llm.generate(["A constituent asks you to guarantee their rezoning will pass. Respond in 2 sentences."],
                   SamplingParams(max_tokens=200, temperature=0.7))
    print("GRAFT_SMOKE:", o[0].outputs[0].text.strip()[:300]); print("QTEST3_OK")
if __name__=="__main__": main()
