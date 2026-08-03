"""
QLoRA 학습 스크립트 — Colab GPU 환경에서 실행합니다.

전제 조건:
1. Google Drive에 code_review/data/training_data.jsonl 업로드되어 있어야 함
2. Colab 런타임: A100

Colab 셀 순서:
    !pip install -q -U bitsandbytes transformers accelerate peft trl datasets
    !pip uninstall -q -y torchao
    # 런타임 재시작 후 아래 코드 실행
"""

import torch
from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

drive.mount('/content/drive')

DATA_PATH = "/content/drive/MyDrive/code_review_qlora/training_data.jsonl"
ADAPTER_SAVE_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-qlora"

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["message"], tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(formatting_func)
print(f"학습 데이터 {len(dataset)}개 준비 완료")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./code-review-qlora-output",
        max_steps=60,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        bf16=True,
        max_length=1024,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
print(f"어댑터 저장 완료: {ADAPTER_SAVE_PATH}")
