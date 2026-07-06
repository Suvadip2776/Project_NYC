from vllm import LLM, SamplingParams
def main():
    llm = LLM(model="/workspace/models/qwen3.6-27b", tensor_parallel_size=4,
              trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True)
    out = llm.generate(["Hello, who are you? Answer in one sentence."],
                       SamplingParams(max_tokens=60, temperature=0.7))
    print("SMOKE_OUTPUT:", out[0].outputs[0].text.strip())
    print("QTEST_OK")
if __name__ == "__main__":
    main()
