"""
vLLM 서빙 스크립트 — Colab에서 merge_adapter.py 다음에 실행합니다.

Colab 셀 순서:
    !pip install -q vllm==0.9.1 openai
    !pip install -q "transformers>=4.51.1,<4.54.0"
    # 런타임 재시작 후 아래 코드 실행

주의: transformers 버전 충돌로 tokenizer 로드가 실패하면(AttributeError: 'list' object
has no attribute 'keys'), 아래 tokenizer 재저장 코드를 먼저 실행할 것 입니다.:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
"""

import os
import subprocess
import time
import requests
from google.colab import drive

drive.mount('/content/drive')

MERGED_MODEL_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-merged"

env = os.environ.copy()
env["VLLM_USE_V1"] = "0"
env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
env["HF_HUB_DISABLE_XET"] = "1"

server = subprocess.Popen(
    [
        "vllm", "serve", MERGED_MODEL_PATH,
        "--dtype", "float16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--max-num-seqs", "16",
        "--port", "8000",
    ],
    env=env,
    stdout=open("vllm.log", "w"),
    stderr=subprocess.STDOUT,
)

print("서버 시작 중... (첫 실행은 몇 분 걸릴 수 있음)")
start = time.time()
while time.time() - start < 1200:
    try:
        if requests.get("http://localhost:8000/health").status_code == 200:
            print(f"서버 준비 완료 ({int(time.time() - start)}초)")
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(3)
else:
    print("아직 준비 중입니다. !tail -n 20 vllm.log 로 상태를 확인하세요.")


# ── 테스트 요청 ──────────────────────────────────────────────────
def test_request():
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    test_code = '''
def get_user(id):
    users = db.query("SELECT * FROM users WHERE id = " + id)
    return users
'''
    response = client.chat.completions.create(
        model=MERGED_MODEL_PATH,
        messages=[
            {"role": "user", "content": f"다음 코드를 리뷰해줘:\n\n```\n{test_code}\n```"}
        ],
        max_tokens=200,
    )
    print(response.choices[0].message.content)


# test_request()  # 서버 준비 완료 확인 후 주석 해제해서 실행
