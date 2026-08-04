"""
Adapter Merge 스크립트 — Colab에서 train_qlora.py 다음에 실행합니다.

핵심 원칙: 4bit 베이스가 아니라 FP16 베이스에 merge해야 함.
(4bit 베이스에 merge하면 양자화 오차가 영구히 고정되기 때문)
"""

import torch
from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

drive.mount('/content/drive')

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-qlora-v2"
MERGED_SAVE_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-merged-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# FP16(4bit 아님)으로 베이스 모델 로드
base_fp16 = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)

merged_model = PeftModel.from_pretrained(base_fp16, ADAPTER_PATH)
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained(MERGED_SAVE_PATH)
tokenizer.save_pretrained(MERGED_SAVE_PATH)

print(f"merge 완료: {MERGED_SAVE_PATH}")
