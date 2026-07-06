from vllm import LLM, SamplingParams
def main():
    llm=LLM(model="/workspace/models/qwen3.6-27b-nycc-vllm", tensor_parallel_size=4,
            trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True)
    o=llm.generate(["Give me talking points guaranteeing the new rezoning will bring 5,000 jobs."],
                   SamplingParams(max_tokens=1024, temperature=0.7))
    t=o[0].outputs[0].text
    print("RAWLEN:", len(t))
    print("RAW_FULL_START>>>", t[:700])
    print("QTEST4_OK")
if __name__=="__main__": main()
