import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)   # folded/distilled model path
    ap.add_argument("--data", required=True)    # sft_data messages jsonl
    ap.add_argument("--out", required=True)     # loras/qwen3.6-27b-introspection/nycc
    ap.add_argument("--smoke", action="store_true")
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager")
    ds=load_dataset("json", data_files=a.data, split="train")
    peft_config=LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")
    cfg=SFTConfig(output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=5e-5, num_train_epochs=1, max_length=3072, bf16=True, logging_steps=5,
        save_strategy="no", warmup_ratio=0.1, gradient_checkpointing=True, report_to="none",
        max_steps=3 if a.smoke else -1)
    trainer=SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(a.out); tok.save_pretrained(a.out)
    print("SFT_TRL_DONE saved to", a.out)

if __name__=="__main__":
    main()
