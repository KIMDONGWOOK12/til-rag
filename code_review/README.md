# 코드리뷰 QLoRA + vLLM 서빙 + RAG

GitHub의 실제 PR 리뷰 코멘트를 데이터로 삼아, 코드 특화 오픈소스 모델을 QLoRA로
파인튜닝하고 vLLM으로 서빙한 뒤, RAG(벡터 검색) 및 LangGraph 기반 품질 검증까지
결합한 코드 리뷰 시스템입니다.

til-rag 저장소의 확장 프로젝트로, TIL-RAG와 같은 RAG 구조를 **다른 도메인**에 적용하면서
**생성 모델까지 직접 학습**해본 것이 차이점입니다.

GPU가 필요한 학습·병합·서빙은 Colab(A100)에서 실행하고, 결과 코드만 이 저장소에 보관합니다.

---

## 파이프라인

```
[로컬]  1. collect_data.py    GitHub PR 리뷰 코멘트 수집
        2. convert_format.py  QLoRA 학습용 형식으로 변환
        3. indexing.py         ChromaDB에 임베딩 저장 (RAG용)

[Colab] 4. train_qlora.py      QLoRA 학습
        5. merge_adapter.py    어댑터를 FP16 베이스에 병합
        6. serve_vllm.py       vLLM으로 서빙 + ngrok 외부 노출

[로컬]  7. review_graph.py     검색 + vLLM 생성 + Claude 검증/번역 (LangGraph)
```

**같은 데이터를 두 갈래로 재사용**하는 것이 이 프로젝트의 구조적 특징입니다.

```
sample_data.json (6529개)
    ├─→ convert_format.py → training_data.jsonl → QLoRA 학습 (생성 능력)
    └─→ indexing.py       → code_review_db      → 벡터 검색 (참고 사례)
```

---

## 데이터

### 출처 (15개 Python 오픈소스 저장소)

`langchain-ai/langgraph`, `langchain-ai/langchain`, `pallets/flask`, `psf/requests`,
`tiangolo/fastapi`, `django/django`, `pandas-dev/pandas`, `numpy/numpy`,
`scikit-learn/scikit-learn`, `python-poetry/poetry`, `sqlalchemy/sqlalchemy`,
`celery/celery`, `pallets/click`, `huggingface/transformers`, `encode/django-rest-framework`

각 저장소의 **merge된 PR 500개**를 오래된 순서부터 스캔해 `review_comments`를 수집
→ 최종 **6529개**.

### 수집 규칙

- `state="closed"` + `pr.merged` 확인 → 실제 반영된 PR만
- `sort="created", direction="asc"` → 오래된 PR부터 (최신 PR은 리뷰가 거의 없음을 실측)
- 리뷰 본문 30자 미만 제외 (LGTM류 노이즈 차단)
- `diff_hunk`(코드) 없거나 20자 미만 제외
- `diff_hunk` 1000자 초과 시 truncate

### truncation을 넣은 이유

수집 직후 길이를 측정하니 **평균 2602자, 최대 25527자**였습니다.
학습 시 `max_length=1024`를 고려하면 상당수가 잘려나가고, 긴 코드는 임베딩 시
핵심이 희석되어 검색 정확도도 떨어집니다. 코드 diff는 중간에서 쪼개면 구조가
깨지므로, 청킹이 아니라 **앞부분만 남기는 truncation**을 선택했습니다.

---

## 학습 이력 — v1 → v7

| 버전 | 데이터 | 모델 | steps | lr | epoch | 결과 |
|---|---|---|---|---|---|---|
| v1 | 286 | Qwen2.5-1.5B-Instruct | 60 | 2e-4 | 1.67 | 과적합 — 무관한 질문에도 학습 문장을 그대로 재생 |
| v2 | 286 | Qwen2.5-1.5B-Instruct | 25 | 1e-4 | 0.70 | 학습 부족 — 지시를 못 따르고 코드를 생성 |
| v3 | 1334 | Qwen2.5-1.5B-Instruct | 100 | 1e-4 | 0.60 | 균형 — SQL 인젝션 최초 정확 답변 |
| v4 | 6529 | Qwen2.5-7B-Instruct | 500 | 1e-4 | 0.61 | 불안정 — 편차 큼 |
| v5 | 6529 | Qwen2.5-7B-Instruct | 800 | 5e-5 | 0.98 | 안정적이나 SQL 약함 |
| v6 | 6529 | **Qwen2.5-Coder-7B-Instruct** | 800 | 5e-5 | 0.98 | SQL은 완벽하나 과적합 심함 |
| **v7** | 6529 | **Qwen2.5-Coder-7B-Instruct** | **350** | 5e-5 | 0.43 | **SQL 인젝션·정상 코드 인식 모두 양호 (최종 채택)** |

### 두 개의 핵심 전환점

**1) 데이터량이 정확도를 결정 (v3)**
데이터를 286→1334개로 늘리자, 무관하던 답변이 정확해지고 SQL 인젝션을 처음으로
제대로 지적하기 시작했습니다.

**2) 베이스 모델의 사전 지식이 결정적 (v6→v7)**
범용 모델(Instruct)은 데이터를 6529개까지 늘려도(v4, v5) SQL 인젝션 같은 보안 패턴을
잘 잡지 못했습니다. **베이스를 코드 특화 모델(Qwen2.5-Coder)로 교체하자 즉시 정확**해졌습니다.

다만 코드 특화 모델은 파인튜닝 데이터의 코드 조각을 더 강하게 암기해, 같은 조건(steps=800)
에서 심한 과적합이 나타났습니다(v6 — 질문과 무관한 학습 코드를 그대로 출력). **steps를
800→350으로 줄이자(v7)** 코드 이해력은 유지하면서 과적합이 완화되어, 가장 균형 잡힌 결과를
얻었습니다.

### v7 실제 출력 예시

**SQL 인젝션 코드 입력**
```python
def get_user(id):
    users = db.query("SELECT * FROM users WHERE id = " + id)
    return users
```

**v7 출력** — 취약점을 정확히 진단하고 파라미터화된 쿼리를 제안:
> SQL 인젝션 공격에 취약합니다. 매개변수화된 쿼리를 사용해야 합니다.
> `db.execute("SELECT * FROM users WHERE id = ?", id)`

---

## RAG + LangGraph 파이프라인 (`review_graph.py`)

TIL-RAG의 judge 노드 패턴을 코드리뷰로 확장한 구조입니다.

```
START → generate(vLLM 리뷰 생성)
      → judge(Claude가 적절성 판단)
           ├─ 적절 → translate(Claude가 한국어로 정리) → END
           ├─ 부적절 + 재시도 여유 → generate로 되돌아감 (최대 3회)
           └─ 부적절 + 3회 소진 → fallback(정직한 실패 메시지) → END
```

### 각 노드의 역할

- **generate**: 제가 학습시킨 vLLM 모델이 리뷰를 생성. RAG 컨텍스트가 있으면 참고.
- **judge**: Claude(Haiku)가 "이 리뷰가 실제로 이 코드에 대한 것인지"를 판단.
  코드와 무관한 내용, 존재하지 않는 변수 언급, 단순 복붙 등을 부적절로 판정.
- **translate**: 적절 판정 시 Claude가 자연스러운 한국어로 정리 (vLLM이 영어로 답해도
  최종 출력은 항상 한국어).
- **fallback**: 3회 재시도에도 실패하면 이상한 텍스트 대신 정직한 실패 메시지를 반환.

**판단을 vLLM이 아닌 Claude로 하는 이유**: 제가 학습시킨 모델은 불안정하므로,
검증만큼은 안정적인 외부 모델에 맡기는 것이 til-rag의 judge 설계 원칙과 일치합니다.

### RAG 유사도 threshold 필터

검색된 문서가 실제로는 무관한데 컨텍스트로 들어가 모델을 방해하는 문제가 있어,
Chroma distance 기준으로 필터링합니다.

| 입력 | 검색 문서 점수 | 관련성 |
|---|---|---|
| 관련 있는 코드 | 0.15 ~ 0.26 | 관련 |
| 무관한 코드 | 0.34 ~ 0.40 | 무관 |

두 구간이 명확히 갈려 **threshold=0.30**으로 무관한 컨텍스트를 차단.
데이터가 1334→6529개로 늘어난 뒤에도 이 경계가 유지됨을 재측정으로 확인했습니다.

---

## LLM-as-Judge 평가 (Gemini + Claude 이중 채점)

동일 코드 3종(보안/버그/정상 코드)에 대해 버전별 리뷰 결과를 두 개의 독립된
LLM(Gemini, Claude)에게 각각 1~5점으로 채점시켜, 평가자 간 일치도를 확인했습니다.

| 버전 | SQL 인젝션 | 제로 나누기 버그 | 정상 코드 | Gemini 평균 | Claude 평균 |
|---|---|---|---|---|---|
| v5 | 1 / - | 5 / 4 | 4 / 3 | 3.33 | 3.50 |
| v6 | 4 / 3 | 1 / 2 | 1 / 1 | 2.00 | 2.00 |
| v7 | 5 / 4 | 1 / 1 | 4 / 4 | 3.33 | 3.00 |

(표기: Gemini점수 / Claude점수)

**두 평가자가 대체로 일치**하며(특히 v6의 낮은 점수는 완전히 일치), v5와 v7이
근접한 상위권, v6가 명확한 하위권임을 확인했습니다.

---

## 실행 방법

### 1) 로컬 — 데이터 준비

```bash
uv run code_review/collect_data.py    # GitHub API 수집 (15개 저장소, 1시간+)
uv run code_review/convert_format.py  # 학습용 형식 변환
uv run code_review/indexing.py         # ChromaDB 인덱싱
```

`training_data.jsonl`을 Google Drive `MyDrive/code_review_qlora/`에 업로드.

### 2) Colab(A100) — 학습 · 병합 · 서빙

`train_qlora.py` → `merge_adapter.py` → `serve_vllm.py` 순서로 실행.
서빙 후 ngrok으로 외부 노출.

### 3) 로컬 — 리뷰 실행

`review_graph.py`의 `VLLM_URL`을 ngrok 주소로 교체 후:
```bash
cd code_review && uv run review_graph.py
```

또는 FastAPI 서버를 통해:
```bash
uv run uvicorn main:app --reload   # POST /review
```

### 4) EC2 배포 (데모)

이 기능은 til-rag 본체와 함께 Docker 이미지로 EC2에 배포되어 있습니다
(`http://15.135.88.186:8000`, IP는 변동 가능). `code_review_db`(6529개 벡터)는
로컬에서 EC2로 `scp`로 옮겼습니다. 단, **코드리뷰는 Colab의 vLLM 서버가 켜져 있고
`VLLM_URL`(ngrok 주소)이 유효할 때만 동작하는 데모용 구성**입니다. Colab 세션이 꺼지면
generate 노드가 실패하므로, 상시 동작하는 TIL 질의응답과 달리 켜져 있을 때만 시연 가능합니다.

---

## 겪은 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| 리뷰 코멘트가 PR 50개에 11개뿐 | 최신 PR은 리뷰가 거의 없음 | 정렬을 `created asc`(오래된 순)로 변경 |
| `torchao` ImportError | peft 요구 버전과 Colab 기본 버전 불일치 | `pip uninstall torchao` + 재시작 |
| `KeyError: 'text'` | SFTTrainer가 `text` 컬럼을 찾는데 데이터엔 `message`만 존재 | `apply_chat_template`으로 `text` 컬럼 생성 |
| vLLM tokenizer 로드 실패 (`'list' object has no attribute 'keys'`) | vLLM 설치로 transformers 버전이 바뀌며 tokenizer 설정과 충돌 | 베이스 모델 tokenizer를 merge 폴더에 재저장 |
| vLLM `aimv2 is already used` | vLLM과 transformers 버전 충돌 | `pip uninstall vllm transformers` 후 정확한 버전으로 재설치 |
| vLLM 404 `model does not exist` | 이전 버전 서버가 8000 포트를 점유 | `pkill -9 -f vllm` → `nvidia-smi` 확인 → 재실행 |
| Gemini 임베딩 429 (일일 1000회) | til-rag + 코드리뷰가 같은 API 키 한도를 공유 | 저장된 개수만큼 건너뛰는 재개 로직 + 별도 계정 키로 우회 |
| 학습 데이터가 예전 개수로 로드됨 | Drive에 새 파일을 업로드하지 않음 | 파일 교체 후 `wc -l`로 개수 확인 |
| merge 시 `adapter_model.safetensors` 없음 | 저장 전 `model` 변수가 덮어써짐 | 학습 직후 바로 저장하고, merge는 `base_fp16` 변수를 따로 사용 |
| GitHub Actions 배포 실패 (SSH 연결 안 됨) | EC2(시드니 리전) 재시작으로 퍼블릭 IP가 바뀌었는데 `SERVER_HOST` secret이 예전 IP를 가리킴 | secret을 최신 IP(15.135.88.186)로 갱신 (Elastic IP 미적용의 부작용) |
| EC2 `no space left on device` | 반복된 Docker 이미지 pull로 미사용 이미지·레이어가 디스크를 가득 채움 | `docker system prune -a -f`로 미사용 이미지·레이어 정리 |

### 인덱싱 재개 로직

429로 중단됐을 때 스크립트를 다시 실행하면 처음부터 다시 임베딩을 요청해
한도를 재소진하는 문제가 있어, 이미 저장된 개수부터 이어가도록 수정했습니다.

```python
already_saved = collection.count()
for i in range(already_saved, len(data), batch_size):
    ...
```

---

## 현재 한계

- **모델 출력의 불안정성 (핵심 한계)**: v7 기준, 같은 코드를 여러 번 입력해도
  매번 다른 답이 나오며, 그중 상당수가 코드 리뷰가 아닌 무관한 문장
  (예: "저는 항상 한국어로 대답합니다")을 출력합니다. 반복 테스트(동일 코드 9회)에서
  제대로 된 리뷰는 소수였습니다. temperature를 0.3으로 낮춰도 개선되지 않아,
  무작위성이 아니라 **모델 자체의 근본적 불안정성**으로 판단했습니다.
  - 원인 추정: 6529개 학습 데이터에 섞인 대화체 문구(사과·약속 문장 등)를
    모델이 이상하게 암기했고, 코드 특화 모델(Qwen2.5-Coder)일수록 이런 패턴을
    더 강하게 학습하는 경향이 있었습니다.
  - 이 때문에 judge 노드가 대부분의 출력을 "부적절"로 판정하고, 결과적으로
    fallback 메시지가 자주 반환됩니다.

- **검증 레이어의 역설적 효과**: judge/fallback 구조는 정상 작동하고 있습니다.
  오히려 나쁜 답을 잘 걸러낸 결과, 사용자에게는 "리뷰를 자주 생성하지 못함"으로
  보입니다. 이는 **불안정한 자체 모델을 단독으로 신뢰할 수 없으며, 외부 검증
  레이어가 반드시 필요하다**는 설계 원칙을 실증한 사례이기도 합니다.

- **데이터 편향**: 15개 저장소가 모두 Python 오픈소스라, 다른 언어·도메인에는
  성능이 더 떨어질 가능성이 높습니다.

- **truncation 손실**: 1000자에서 잘린 긴 diff의 뒷부분 맥락은 반영되지 않았습니다.

- **Colab 의존**: vLLM은 GPU가 필요해 Colab 세션이 살아있어야만 동작하는 데모용 구성입니다.

## 배운 점

- 데이터를 늘리거나 모델을 키우는 것이 **항상** 성능 향상으로 이어지지는 않습니다.
  (v4가 v3보다 불안정했던 사례가 있습니다)
- **베이스 모델의 사전 지식**이 파인튜닝 데이터의 양보다 특정 능력(보안 패턴 이해)에
  더 큰 영향을 줄 수 있습니다.
- 코드 특화 모델은 표현력이 큰 만큼 과적합도 쉬워, **학습 강도(steps) 조절**이 중요합니다.
- 불안정한 자체 모델의 출력은 **외부의 안정적인 모델(Claude)로 검증**하는 시스템
  레벨의 안전장치로 신뢰도를 끌어올릴 수 있습니다.


---
## 다음 개선 방향

이 프로젝트는 제출 이후에도 계속 개선할 예정이며, 다음 순서로 접근하려 합니다.

1. **학습 데이터 정제**: 리뷰 코멘트 중 코드 리뷰와 무관한 대화체 문구
   (사과·인사·약속 문장, 봇 자동 코멘트 등)를 필터링해 노이즈를 제거합니다.
   현재의 불안정성이 데이터 품질에서 기인했다는 가설을 우선 검증합니다.

2. **정적 분석 도구와의 하이브리드**: SQL 인젝션 같은 보안 취약점은 규칙 기반
   도구(`bandit`, `semgrep`)가 100% 정확하게 탐지합니다. 이를 먼저 실행해
   탐지 결과를 컨텍스트로 넘기고, AI는 "설명과 개선 제안"에 집중하게 하여
   AI 단독의 불안정성을 구조적으로 보완합니다.

3. **평가 자동화**: 현재는 소수의 테스트 케이스를 수동으로 확인했으나,
   여러 유형의 코드(보안·버그·스타일)에 대한 평가셋을 만들고 LLM-as-Judge로
   각 버전의 성능을 정량 비교해, "우연히 좋은 답"과 "안정적으로 좋은 답"을
   구분할 수 있게 합니다.

4. **베이스 모델·학습 조건 재실험**: v5(범용 7B)가 v6·v7(코드 특화)보다 안정적이었던
   점을 근거로, 안정성과 코드 이해력 사이의 균형점을 다시 탐색합니다.