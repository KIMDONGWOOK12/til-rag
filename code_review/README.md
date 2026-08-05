# 코드리뷰 QLoRA + vLLM 서빙 + RAG

GitHub의 실제 PR 리뷰 코멘트를 데이터로 삼아, Qwen2.5-1.5B-Instruct를 QLoRA로 파인튜닝하고
vLLM으로 서빙한 뒤, RAG(벡터 검색)와 결합한 코드 리뷰 시스템입니다.

til-rag 저장소의 확장 프로젝트로, TIL-RAG와 같은 RAG 구조를 **다른 도메인**에 적용하면서
**생성 모델까지 직접 학습**해본 것이 차이점입니다.

GPU가 필요한 학습·병합·서빙은 Colab에서 실행하고, 결과 코드만 이 저장소에 보관합니다.

---

## 파이프라인

```
[로컬]  1. collect_data.py    GitHub PR 리뷰 코멘트 수집
        2. convert_format.py  QLoRA 학습용 형식으로 변환
        3. indexing.py         ChromaDB에 임베딩 저장 (RAG용)

[Colab] 4. train_qlora.py      QLoRA 학습
        5. merge_adapter.py    어댑터를 FP16 베이스에 병합
        6. serve_vllm.py       vLLM으로 서빙 + ngrok 외부 노출

[로컬]  7. rag_review.py       검색 + vLLM 생성 결합
```

**같은 데이터를 두 갈래로 재사용**하는 것이 이 프로젝트의 구조적 특징입니다.

```
sample_data.json (1334개)
    ├─→ convert_format.py → training_data.jsonl → QLoRA 학습 (생성 능력)
    └─→ indexing.py       → code_review_db      → 벡터 검색 (참고 사례)
```

---

## 데이터

### 출처

| 저장소 | 수집 코멘트 | 그중 truncate |
|---|---|---|
| langchain-ai/langgraph | 146 | 51 |
| langchain-ai/langchain | 248 | 106 |
| pallets/flask | 190 | 37 |
| psf/requests | 34 | 8 |
| tiangolo/fastapi | 716 | 419 |
| **합계** | **1334** | **621** |

각 저장소의 **merge된 PR 500개**를 오래된 순서부터 스캔해 `review_comments`를 수집했습니다.

### 수집 규칙

- `state="closed"` + `pr.merged` 확인 → 실제 반영된 PR만
- `sort="created", direction="asc"` → **오래된 PR부터**
  (최신 PR은 리뷰 코멘트가 거의 없고, 초창기 PR에 리뷰가 훨씬 많다는 것을 실측으로 확인)
- 리뷰 본문 30자 미만 제외 (LGTM류 노이즈 차단)
- `diff_hunk`(코드) 없거나 20자 미만 제외
- `diff_hunk` 1000자 초과 시 truncate (`... (생략)`)

### truncation을 넣은 이유

수집 직후 길이를 측정하니 **평균 2602자, 최대 25527자**였습니다.
`train_qlora.py`의 `max_length=1024` 설정을 고려하면 상당수 데이터가 학습 시
잘려나가고, 긴 코드는 임베딩 시 핵심이 희석되어 검색 정확도도 떨어집니다.

코드 diff는 마크다운 문서와 달리 중간에서 쪼개면 구조가 깨지므로,
청킹(분할)이 아니라 **앞부분만 남기는 truncation**을 선택했습니다.

---

## 학습 이력 — v1 → v2 → v3

| 버전 | 데이터 | max_steps | lr | loss | 결과 |
|---|---|---|---|---|---|
| v1 | 286 | 60 | 2e-4 | 1.69 → 0.82 | **과적합** — 무관한 질문에도 학습 데이터의 PR 코멘트를 그대로 재생 |
| v2 | 286 | 25 | 1e-4 | 1.72 → 1.27 | 과적합은 개선됐으나, "리뷰" 지시를 제대로 못 따르고 코드를 생성해버림 |
| **v3** | **1334** | **100** | **1e-4** | **2.44 → 1.28** | **SQL 인젝션 코드에 파라미터 바인딩(`%s`)을 정확히 제안** |

### v1의 문제 — 과적합

`print(1+1)`처럼 코드 리뷰와 무관한 입력에도,
학습 데이터에 있던 PR 코멘트(`"i was thinking we can just return the final output..."`)를
그대로 뱉었습니다. 286개라는 적은 데이터에 60스텝을 돌리면서
"질문에 답하는 법"이 아니라 "문장을 외워서 재생하는 것"을 학습한 것으로 판단했습니다.

### v2의 문제 — 학습 부족

스텝을 25로 줄이니 과적합 재생은 사라졌지만, 이번엔 반대로
"리뷰해줘"라는 지시를 제대로 못 따르고 **코드를 그대로 생성**하는 증상이 나타났습니다.
적은 데이터에서는 과적합과 학습 부족 사이의 균형점을 찾기 어렵다는 결론.

### v3 — 데이터 확충으로 해결

저장소를 2개 → 5개, PR을 300개 → 500개로 늘려 데이터를 286개 → 1334개(4.7배)로 확충.
같은 100스텝에서도 epoch이 0.6에 불과해(전체 데이터를 한 바퀴도 못 돔)
특정 문장을 암기할 여지가 줄었고, 실제로 SQL 인젝션 케이스에서 정확한 리뷰가 나왔습니다.

**입력**
```python
def get_user(id):
    users = db.query("SELECT * FROM users WHERE id = " + id)
    return users
```

**v3 출력**
````
```suggestion
    users = db.query("SELECT * FROM users WHERE id = %s", (id))
```
````

---

## RAG 결합 — 유사도 threshold 필터

### 문제

RAG 컨텍스트가 붙으면, 모델이 **"진짜 질문"과 "참고 자료"를 구분하지 못하고
참고 자료 쪽 코드를 리뷰 대상으로 착각**하는 현상이 있었습니다.

원인을 추적해보니 검색 자체가 부정확했습니다.
데이터셋이 LangGraph/LangChain 내부 코드에 편향되어 있어,
SQL 관련 코드를 검색하면 **실제로는 무관한데도 "가장 가까운 것"으로 뽑혀** 나왔습니다.

### 실측한 유사도 점수 (Chroma distance, 낮을수록 유사)

| 입력 | 검색된 문서 점수 | 실제 관련성 |
|---|---|---|
| 체크포인트 관련 코드 | 0.15 ~ 0.26 | 관련 있음 |
| SQL 인젝션 코드 | 0.39 ~ 0.40 | **무관** |

두 구간이 명확히 갈렸으므로 **threshold를 0.30**으로 설정해,
무관한 컨텍스트는 아예 프롬프트에 넣지 않도록 했습니다.

```python
def get_relevant_context(query: str, k: int = 3, threshold: float = 0.30) -> str:
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    relevant = [doc for doc, score in docs_with_scores if score < threshold]
    if not relevant:
        return ""          # 컨텍스트 없이 순수 모델 답변으로
    return "\n\n".join(...)
```

`rag_review.py`는 컨텍스트가 비어 있으면 **참고 사례 부분을 아예 생략한 프롬프트**를 사용합니다.

---

## 실행 방법

### 1) 로컬 — 데이터 준비

```bash
uv run code_review/collect_data.py    # GitHub API로 수집 (20~30분 소요)
uv run code_review/convert_format.py  # 학습용 형식 변환
uv run code_review/indexing.py         # ChromaDB 인덱싱
```

`training_data.jsonl`을 Google Drive `MyDrive/code_review_qlora/`에 업로드.

### 2) Colab — 학습 · 병합 · 서빙

`train_qlora.py` → `merge_adapter.py` → `serve_vllm.py` 순서로
각 파일 내용을 Colab 셀에 나눠 붙여넣어 실행.

서빙 후 ngrok으로 외부 노출:
```python
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)
```

### 3) 로컬 — RAG + vLLM 결합 실행

`rag_review.py`의 `VLLM_URL`을 위에서 나온 ngrok 주소로 교체 후:

```bash
cd code_review && uv run rag_review.py
```

또는 FastAPI 서버를 통해:
```bash
uv run uvicorn main:app --reload   # POST /review
```

---

## 겪은 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| 리뷰 코멘트가 PR 50개에 11개뿐 | 최신 PR은 리뷰가 거의 없음 | 정렬을 `created asc`(오래된 순)로 변경 → 급증 |
| `torchao` ImportError | peft 요구 버전과 Colab 기본 버전 불일치 | `pip uninstall torchao` + 런타임 재시작 |
| `KeyError: 'text'` | SFTTrainer가 `text` 컬럼을 찾는데 데이터엔 `message`만 존재 | `apply_chat_template`으로 `text` 컬럼 직접 생성 |
| `bitsandbytes` ImportError | 새 Colab 세션에 미설치 | 설치 후 런타임 재시작 |
| vLLM tokenizer 로드 실패 (`'list' object has no attribute 'keys'`) | vLLM 설치로 transformers 버전이 바뀌며 저장된 tokenizer 설정과 충돌 | 베이스 모델 tokenizer를 merge 폴더에 재저장 |
| vLLM 404 `model does not exist` | 이전 버전 서버가 8000 포트를 점유한 채 신규 서버 실행 실패 | `pkill -9 -f vllm` → `nvidia-smi`로 GPU 메모리 확인 → 재실행 |
| Gemini 임베딩 429 (일일 1000회 한도) | til-rag + 코드리뷰 인덱싱이 같은 API 키의 한도를 공유 | ① 이미 저장된 개수만큼 건너뛰고 이어서 처리하도록 `indexing.py` 수정 ② 별도 계정의 API 키로 우회 |
| 학습 데이터가 286개로 로드됨 | Drive에 새 `training_data.jsonl`을 업로드하지 않아 예전 파일을 읽음 | 파일 교체 후 `wc -l`로 1334 확인 후 재학습 |

### 인덱싱 재개 로직

429로 중단됐을 때 스크립트를 다시 실행하면 **처음부터 다시 임베딩을 요청**해
한도를 순식간에 재소진하는 문제가 있었습니다. 이미 저장된 개수를 확인해
그 지점부터 이어가도록 수정했습니다.

```python
already_saved = collection.count()
for i in range(already_saved, len(data), batch_size):
    ...
```

---

## 현재 한계

- **모델 크기**: 1.5B 파라미터로는 코드의 맥락을 깊이 이해하는 데 한계가 있습니다.
  일부 입력에서는 여전히 짧고 피상적인 리뷰가 나옵니다.
- **데이터 편향**: 5개 저장소 모두 Python 웹/AI 프레임워크라, 다른 언어·도메인 코드에는
  검색 정확도와 리뷰 품질이 떨어질 가능성이 높습니다.
- **truncation 손실**: 1334개 중 621개(47%)가 1000자에서 잘렸습니다.
  긴 diff의 뒷부분 맥락은 학습·검색에 반영되지 않았습니다.
- **Colab 의존**: vLLM은 GPU가 필요해 Colab 세션이 살아있어야만 동작합니다.
  세션이 끊기면 ngrok 주소도 함께 무효화됩니다.

## 다음 개선 방향

- 저장소를 다양한 언어·도메인으로 확대해 편향 완화
- 더 큰 베이스 모델(3B~7B)로 재학습
- 프롬프트 엔지니어링 강화 (few-shot 예시 삽입 등)
- threshold를 1334개 데이터 기준으로 재측정·재조정