# 코드리뷰 QLoRA + vLLM 서빙

GitHub의 실제 PR 리뷰 코멘트를 학습 데이터로 삼아, Qwen2.5-1.5B-Instruct를
QLoRA로 파인튜닝하고 vLLM으로 서빙하는 파이프라인.

til-rag 저장소의 `feature/code-review-qlora` 브랜치에서 진행.
QLoRA 학습/merge/서빙은 GPU가 필요해 Colab에서 실행하고, 결과 코드만 이 저장소에 보관.

## 파이프라인
## 데이터 출처

- `langchain-ai/langgraph`, `langchain-ai/langchain`의 merge된 PR 300개씩
- review_comments(코드 라인 리뷰) + issue_comments(PR 전체 코멘트) 수집
- 최종 학습 데이터: 286개 (코드 스니펫이 있는 review_comment만 사용)

## 실행 순서

### 1) 로컬에서 데이터 준비

```bash
uv run code_review/collect_data.py
uv run code_review/convert_format.py
```

결과물 `code_review/data/training_data.jsonl`을 Google Drive
`MyDrive/code_review_qlora/`에 업로드.

### 2) Colab에서 학습·병합·서빙

`train_qlora.py` → `merge_adapter.py` → `serve_vllm.py` 순서로,
각 파일 내용을 Colab 노트북 셀에 나눠 붙여넣어 실행.

## 결과

- 학습 loss: 1.69 → 0.82 (60 step)
- vLLM 서버 준비 시간: 73초
- 실제 테스트: SQL 인젝션 취약 코드에 대해
  `db.query("...WHERE id = %s", (id,))` 형태의 파라미터 바인딩을 정확히 제안

## 겪은 문제

| 문제 | 원인 | 해결 |
|---|---|---|
| `torchao` 버전 충돌 | peft가 요구하는 torchao 버전과 Colab 기본 버전 불일치 | `pip uninstall torchao` |
| `KeyError: 'text'` | SFTTrainer가 `text` 컬럼을 찾는데 데이터엔 `message`만 있음 | `apply_chat_template`으로 `text` 컬럼 직접 생성 |
| `bitsandbytes` ImportError | 새 Colab 세션에 라이브러리 미설치 | `pip install bitsandbytes` 후 런타임 재시작 |
| vLLM tokenizer 로드 실패 (`AttributeError: 'list' object has no attribute 'keys'`) | vLLM 설치로 `transformers` 버전이 바뀌며 저장된 tokenizer 설정과 불일치 | 베이스 모델의 tokenizer를 재저장하여 덮어씀 |

## 다음 단계 (선택, 미완료)

- RAG 결합: 리뷰 코멘트를 Chroma에 인덱싱하고, 코드 입력 시 유사 사례를 검색해
  이 vLLM 모델의 컨텍스트로 제공 → "코드리뷰 RAG 기반 LLM"으로 확장
- PTQ 비교 (FP16 vs 4bit 메모리 측정) — 9주차와 동일한 방식으로 추가 가능
