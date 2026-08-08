# TIL-RAG — 개인 학습 기록(TIL) 기반 RAG 질의응답 시스템

## Description

부트캠프 학습 과정에서 작성한 TIL(Today I Learned) 마크다운 문서를 지식 베이스로 삼아,
**순수 구현 → LangChain → LangGraph** 순서로 동일한 RAG 파이프라인을 세 번 재구축하며
각 프레임워크가 왜 필요한지, 무엇을 대체하는지를 직접 코드로 검증한 개인 학습 프로젝트입니다.

기존에 완성한 [`dev-docs-rag-agent`](https://github.com/KIMDONGWOOK12/dev-docs-rag-agent)(8주차 제출물)와
별개로, **AI 의존 없이 바닥부터 이해하고 다시 짜보기 위해** 자발적으로 시작했으며,
현재는 코드리뷰 도메인으로 확장(QLoRA 파인튜닝 + vLLM 서빙 + LangGraph 검증)까지 진행했습니다.

---

## Roadmap Status

| Phase | 항목 | 내용 | 상태 |
|---|---|---|---|
| 0 | TIL 문서 작성 | 부트캠프 학습 기록 42개 마크다운 작성 | ✅ |
| 0 | 전처리 파이프라인 | 유니코드 정규화(NFC), 제어문자 제거, 공백 정리 | ✅ |
| 0 | Chunking | 고정 길이 분할 (chunk_size=500, overlap=50) | ✅ |
| 0 | 인덱싱 | Gemini 임베딩 + ChromaDB (299청크, `collection_name="til_rag"`) | ✅ |
| 1 | 순수 RAG 구현 | `retrieve()` / `generate()` 직접 구현 (`rag.py`) | ✅ |
| 1 | 프롬프트 설계 | 5원칙 (근거 문서만 사용 / 한국어 답변 / 모르면 거절 / 출처 표시 / 3문장 제한) | ✅ |
| 1 | 환각 방지 검증 | 무관 질문(김치찌개 레시피) 시 "찾을 수 없음" 응답 확인 | ✅ |
| 1 | RAG 평가 | `evaluate.py` — 규칙 기반 vs LLM-as-Judge 비교 | ✅ |
| 2 | LangChain 마이그레이션 | 기존 Chroma 인덱스 재오픈 (재인덱싱 없이 전환) (`rag_lc.py`) | ✅ |
| 2 | LCEL 체인 조립 | `retriever \| format_docs`, `RunnablePassthrough` | ✅ |
| 2 | 동일 동작 검증 | rag.py 대비 동일 질문·동일 결과 확인 | ✅ |
| 3 | judge 노드 + 조건 분기 | LLM이 검색 필요 여부 판단 → `add_conditional_edges` | ✅ |
| 3 | ReAct Agent 확장 | `bind_tools` + `ToolNode` + `tools_condition` (`rag_graph.py`) | ✅ |
| 3 | Agent 자율판단 검증 | 지식 질문 → 도구 호출 / 잡담 → 도구 미호출 분기 확인 | ✅ |
| 4 | FastAPI 배포 | REST API 래핑 — `/ask`, `/review`, `/docs/{filename}` | ✅ |
| 4 | 프론트엔드 | TIL/코드리뷰 모드 토글, 카카오톡 스타일 UI, 대화 초기화 | ✅ |
| 5 | Qwen QLoRA Fine-Tuning | Colab GPU 환경에서 별도 진행 | ✅ |
| 5 | Post-Training Quantization | 양자화 전후 성능·메모리 비교 | ✅ |
| 5 | GGUF 변환 + Llama.cpp 추론 | Colab에서 별도 진행 | ✅ |
| 6 | Unix 프로세스·메모리 분석 | `ps` / `top` / `lsof` / `vmmap` 분석 보고서 | ✅ |
| 6 | Wireshark 통신 캡처 | loopback + 서로 다른 두 기기 간 HTTP 캡처 | ✅ |
| 7 | Docker 컨테이너화 | Dockerfile + Docker Compose | ✅ |
| 7 | AWS EC2 배포 | Docker Hub 경유, 외부 접근 가능 구성 | ✅ |
| 7 | GitHub Actions CI/CD | push 시 자동 빌드·배포 파이프라인 | ✅ |
| 8 | 코드리뷰 데이터 수집 | GitHub PR 리뷰 6529개 (15개 Python 저장소) | ✅ |
| 8 | 코드리뷰 QLoRA 학습 | v1~v7 반복 (베이스 모델·데이터·steps 조정) | ✅ |
| 8 | vLLM 서빙 | OpenAI 호환 API, ngrok 외부 노출 | ✅ |
| 8 | RAG + LangGraph 검증 | threshold 필터 + judge/translate/fallback 노드 | ✅ |
| 8 | 코드리뷰 기능 EC2 반영 | 데모용 구성(Colab vLLM 의존) | ✅ |
| 8 | 모델 안정성 개선 | 데이터 정제 + 정적 분석 하이브리드 | 📋 계획 |

---

## Architecture

Agent 그래프 구조 (`rag_graph.py`):

```mermaid
graph TD;
        __start__([<p>__start__</p>]):::first
        agent(agent)
        tools(tools)
        __end__([<p>__end__</p>]):::last
        __start__ --> agent;
        agent -.-> __end__;
        agent -.-> tools;
        tools --> agent;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

- 실선(`-->`): 고정된 흐름 / 점선(`-.->`): `tools_condition`이 판단하는 조건부 분기
- `tools → agent`로 되돌아가는 순환 구조가 이 그래프를 Workflow가 아닌 Agent로 만드는 핵심

---

## 검증 결과 요약

같은 두 질문을 3개 구현체에 동일하게 던져,
로직 전환 과정에서 동작이 깨지지 않았는지 확인했습니다.

| 질문 | rag.py (순수) | rag_lc.py (LangChain) | rag_graph.py (Workflow) | rag_graph.py (Agent) |
|---|---|---|---|---|
| "RAG가 뭐야?" | ✅ 문서 근거 답변 | ✅ 동일 | ✅ judge=True → 검색 후 답변 | ✅ 도구 호출 후 답변 |
| 무관 질문 | ✅ "찾을 수 없음" | ✅ 동일 | ✅ judge=False → 검색 생략 | ✅ 도구 미호출 |

---

## Function

### Main
- TIL 문서 기반 질의응답 (검색 → 생성)
- 코드 리뷰 (RAG 검색 + 자체 파인튜닝 모델 생성 + Claude 검증)

### Sub
- 검색 필요 여부 자율 판단 (judge 노드 / Agent의 도구 선택)
- 문서 근거가 없을 시 환각 방지 응답
- 답변 시 근거 문서 파일명 인용, 클릭 시 원본 TIL 문서 열람
- TIL / 코드리뷰 모드 전환 UI
- 4단계 비교 구조 유지 (순수 / LangChain / Workflow / Agent)

---

## 설계 결정 & 근거

- **Chroma 컬렉션 재사용**: 인덱싱(299청크, 42분 소요)을 재실행하지 않기 위해,
  프레임워크 전환 시 `collection_name="til_rag"`로 기존 인덱스를 재오픈.
- **judge는 규칙이 아닌 LLM 호출**: "오늘 너무 힘들고 피곤했다"처럼 표면적 규칙으로는
  검색 필요 여부를 판단할 수 없는 실제 사례를 확인 → 의미 기반 판단이 필요.
- **judge+조건분기는 Workflow, bind_tools+ToolNode는 Agent**: 실행 경로를 개발자가 코드로
  고정했는지, LLM이 실행 중 스스로 선택하는지로 구분.
- **출처를 파일명으로 표시**: `doc.metadata['source']`를 사용해 실제 파일명을 노출 →
  프론트엔드에서 클릭 가능한 링크로 변환.

---

## 확장 프로젝트 — 코드리뷰 QLoRA + vLLM 서빙 (`code_review/`)

TIL-RAG와 같은 RAG 구조를 **다른 도메인**에 적용하고,
**생성 모델까지 직접 학습**해본 확장 실습입니다.

| | TIL-RAG | 코드리뷰 |
|---|---|---|
| 도메인 | 개인 학습 기록(TIL 마크다운 42개) | GitHub PR 리뷰 코멘트 6529개 |
| 생성 모델 | Gemini API (비공개 가중치, 외부 서비스) | Qwen2.5-Coder-7B (공개 가중치) QLoRA 파인튜닝 → vLLM 서빙 |
| 검색 | ChromaDB (299청크) | ChromaDB (6529개) + 유사도 threshold 필터 |
| 품질 검증 | judge 노드 (검색 필요 판단) | judge/translate/fallback (Claude가 답변 검증·번역) |

### 학습 이력 (핵심만)

| 버전 | 데이터 | 모델 | steps | 결과 |
|---|---|---|---|---|
| v3 | 1334 | Qwen2.5-1.5B-Instruct | 100 | SQL 인젝션 최초 정확 답변 |
| v5 | 6529 | Qwen2.5-7B-Instruct | 800 | 안정적이나 SQL 약함 |
| v6 | 6529 | Qwen2.5-Coder-7B-Instruct | 800 | SQL 완벽하나 과적합 |
| **v7** | 6529 | Qwen2.5-Coder-7B-Instruct | 350 | 균형 잡힌 최종 채택본 |

**핵심 발견**: 데이터량 확충(v3)과 **코드 특화 베이스 모델로의 교체(v6→v7)**가 정확도를
좌우했으며, 코드 특화 모델은 과적합이 쉬워 학습 강도(steps) 조절이 중요했습니다.
단, v7은 반복 테스트에서 출력 불안정성이 확인되어, 이를 "검증 레이어의 필요성을
실증한 한계"로 기록하고 후속 개선(데이터 정제, 정적 분석 도구 결합)을 계획했습니다.

자세한 내용은 [`code_review/README.md`](code_review/README.md) 참고.


---


### 평가 (LLM-as-Judge, Gemini + Claude 이중 채점)

동일 코드 3종에 대한 버전별 리뷰를 Gemini와 Claude 두 모델로 각각 채점한 결과,
v5(Gemini 3.33 / Claude 3.50)와 v7(Gemini 3.33 / Claude 3.00)이 근접한
최상위권을 보였고, v6(Gemini 2.00 / Claude 2.00, 두 평가자 완전 일치)는
SQL 인젝션에는 강하나 다른 케이스에서 불안정해 과적합 가설을 뒷받침했습니다.
자세한 채점 결과는 [`code_review/README.md`](code_review/README.md) 참고.

---

## Version

### Patch Version — Development Stage
- 0.1.0: 순수 RAG 구현 — TIL 42개, 299청크 인덱싱, 환각 방지 검증
- 0.2.0: LangChain 마이그레이션 — LCEL 체인
- 0.3.0: LangGraph StateGraph — judge 노드 + 조건부 라우팅
- 0.3.1: LangGraph ReAct Agent — bind_tools + ToolNode
- 0.4.0: FastAPI + 프론트엔드
- 0.5.0: Docker + EC2 배포 + GitHub Actions CI/CD
- 0.6.0: 코드리뷰 QLoRA + vLLM + LangGraph 검증 (v7: Qwen2.5-Coder-7B, 6529개) ← now processing
### Minor Version (0.N.x) — LLM Provider
- 0.0.x: Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`) ← **now processing**

---

## 실행 방법

```bash
# 0. 인덱싱 (최초 1회, TIL 문서 → Chroma)
uv run indexing.py

# 1. 순수 RAG
uv run rag.py

# 2. RAG 평가
uv run evaluate.py

# 3. LangChain 기반 RAG
uv run rag_lc.py

# 4. LangGraph ReAct Agent
uv run rag_graph.py

# 5. FastAPI 서버 (프론트엔드 포함)
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# 6. Docker
docker compose up -d
```

코드리뷰 확장 프로젝트 실행은 [`code_review/README.md`](code_review/README.md) 참고.
(vLLM은 GPU가 필요해 Colab에서 실행하고, ngrok으로 노출된 주소를 로컬/EC2에서 호출합니다.)

### 배포된 서버

```
http://15.135.88.186:8000   # 현재 퍼블릭 IP (변동 가능)
```

EC2 인스턴스를 중지 후 재시작하면 퍼블릭 IP가 변경됩니다 (Elastic IP 미적용).
IP가 바뀌면 GitHub Actions의 `SERVER_HOST` secret도 함께 갱신해야 자동 배포가 성공합니다.
코드리뷰 기능은 Colab의 vLLM 서버가 켜져 있을 때만 동작하는 데모용 구성입니다.
(TIL 질의응답은 EC2 단독으로 상시 동작하지만, 코드리뷰는 Colab 세션이 살아 있을 때만 작동)

---

## Data Source

1. **개인 TIL 마크다운 42개** (til-rag)
   - `til_notes/` → 전처리 후 299청크로 분할 → `chroma_db/`

2. **GitHub PR 리뷰 코멘트 6529개** (code_review)
   - 15개 Python 오픈소스 저장소의 merge된 PR에서 수집
   - `code_review/data/` → `code_review_db/`

---

## 앞으로의 계획

- Elastic IP 적용으로 배포 주소 고정 (IP 변동에 따른 CI/CD secret 갱신 문제 해소)
- 다양한 언어·도메인 저장소 추가로 데이터 편향 완화
- 자동화된 테스트 코드 작성 → CI 파이프라인에 test Job 추가

> 코드리뷰 기능 EC2 반영은 Docker 재빌드 → Docker Hub push → EC2 pull로 완료되었습니다.