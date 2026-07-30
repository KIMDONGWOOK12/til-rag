# Title: TIL-RAG — 개인 학습 기록(TIL) 기반 RAG 질의응답 시스템

## Description
부트캠프 학습 과정에서 작성한 TIL(Today I Learned) 마크다운 문서를 지식 베이스로 삼아,
**순수 구현 → LangChain → LangGraph** 순서로 동일한 RAG 파이프라인을 세 번 재구축하며
각 프레임워크가 왜 필요한지, 무엇을 대체하는지를 직접 코드로 검증한 개인 학습 프로젝트입니다.

기존에 완성한 [`dev-docs-rag-agent`](https://github.com/KIMDONGWOOK12/dev-docs-rag-agent)(8주차 제출물)와
별개로, **AI 의존 없이 바닥부터 이해하고 다시 짜보기 위해** 자발적으로 시작하였으며, 추 후 이 프로젝트를 개인 프로젝트로 나가려고 합니다.

---

## Roadmap Status

| Phase | 항목 | 내용 | 상태 |
|---|---|---|---|
| 0 | TIL 문서 작성 | 부트캠프 학습 기록 42개 마크다운 작성 | ✅ 완료 |
| 0 | 전처리 파이프라인 | 유니코드 정규화(NFC), 제어문자 제거, 공백 정리 | ✅ 완료 |
| 0 | Chunking | 고정 길이 분할 (chunk_size=500, overlap=50) | ✅ 완료 |
| 0 | 인덱싱 | Gemini 임베딩 + ChromaDB 저장 (299개 청크, `collection_name="til_rag"`) | ✅ 완료 |
| 1 | 순수 RAG 구현 | `retrieve()` / `generate()` 직접 구현 (`rag.py`) | ✅ 완료 |
| 1 | 프롬프트 설계 | 5원칙 (근거 문서만 사용 / 모르면 거절 / 문서번호 표시 / 3문장 제한) | ✅ 완료 |
| 1 | 환각 방지 검증 | 무관 질문(예: 김치찌개 레시피) 시 "찾을 수 없음" 응답 확인 | ✅ 완료 |
| 1 | RAG 평가 | `evaluate.py` — 규칙 기반 vs LLM-as-Judge 비교 평가 | ✅ 완료 |
| 2 | LangChain 마이그레이션 | 기존 Chroma 인덱스 재오픈 (재인덱싱 없이 전환) (`rag_lc.py`) | ✅ 완료 |
| 2 | 프롬프트 이식 | `ChatPromptTemplate` system/human 분리 | ✅ 완료 |
| 2 | LCEL 체인 조립 | `retriever \| format_docs`, `RunnablePassthrough` | ✅ 완료 |
| 2 | 동일 동작 검증 | rag.py 대비 동일 질문·동일 결과(RAG 정의, 환각방지) 확인 | ✅ 완료 |
| 3 | State/Node/Edge 설계 | `TypedDict` 기반 State, 단순 선형 그래프 | ✅ 완료 |
| 3 | judge 노드 + 조건부 라우팅 | LLM이 검색 필요 여부 판단 → `add_conditional_edges` 분기 (`rag_graph.py`) | ✅ 완료 |
| 3 | 판단 근거 실증 | "오늘 너무 힘들고 피곤했다" 같은 일지성 문장에서 검색 생략 확인 | ✅ 완료 |
| 3 | ReAct Agent 확장 | `bind_tools` + `ToolNode` + `tools_condition` (`rag_graph.py`) | ✅ 완료 |
| 3 | Agent 자율판단 검증 | 지식 질문 → 도구 호출 / 잡담 → 도구 미호출 분기 확인 | ✅ 완료 |
| 4 | FastAPI 배포 | REST API 래핑 | ✅ 완료  |
| 5 | Qwen 모델 LoRA·QLoRA Fine-Tuning (코랩으로 별도) | ✅ 완료  |
| 5 | Post-Training Quantization 적용 및 양자화 전후 성능·메모리 비교 (코랩으로 별도) | ✅ 완료  |
| 5 | GGUF 포맷 변환 및 Llama.cpp 추론 (코랩으로 별도) | ✅ 완료  |
| 6 | Unix 환경 프로세스·스레드·메모리 상태 분석 보고서 | ✅ 완료  |
| 6 | Wireshark를 통한 HTTP/HTTPS 통신 캡처 분석 | ✅ 완료  |
| 7 | Docker 컨테이너화 + Docker Compose 실행 | ✅ 완료  |
| 7 | AWS EC2 배포 (외부 접근 가능 구성) | ✅ 완료  |
| 8 | GitHub Actions 기반 CI/CD 파이프라인 구축 | ✅ 완료  |



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

- 실선(`-->`): 고정된 흐름
- 점선(`-.->`): `tools_condition`이 판단하는 조건부 분기
- `tools → agent`로 되돌아가는 순환 구조가 이 그래프를 Workflow가 아닌 **Agent**로 만드는 핵심 요소
---

## 검증 결과 요약

같은 두 질문을 3개 구현체(`rag.py` → `rag_lc.py` → `rag_graph.py`)에
동일하게 던져, **로직 전환 과정에서 동작이 깨지지 않았는지** 확인했습니다.

| 질문 | rag.py (순수) | rag_lc.py (LangChain) | rag_graph.py (Workflow) | rag_graph.py (Agent) |
|---|---|---|---|---|
| "RAG가 뭐야?" | ✅ 문서 근거 답변 | ✅ 동일 | ✅ judge=True → 검색 후 답변 | ✅ 도구 호출 후 답변 |
| 무관 질문 (김치찌개 / 일지성 문장) | ✅ "찾을 수 없음" | ✅ 동일 | ✅ judge=False → 검색 생략 | ✅ 도구 미호출, 자연스런 응답 |

---

## Function

### Main
- TIL 문서 기반 질의응답 (검색 → 생성)

### Sub
- 검색 필요 여부 자율 판단 (judge 노드 / Agent의 도구 선택)
- 문서 근거가 없을 시 환각 방지 응답 ("제 TIL에서 해당 정보를 찾을 수 없습니다")
- 답변 시 근거 문서 번호 인용
- 순수구현 / LangChain / LangGraph(Workflow) / LangGraph(Agent) 4단계 비교 구조 유지

---

## 설계 결정 & 근거

- **Chroma 컬렉션 재사용**: 인덱싱(299청크, 42분 소요)을 재실행하지 않기 위해, LangChain/LangGraph
  단계 전환 시 `Chroma(persist_directory=..., collection_name="til_rag")`로 기존 인덱스를 재오픈.
- **judge는 규칙이 아닌 LLM 호출**: "오늘 너무 힘들고 피곤했다"처럼 키워드·물음표·길이 같은 표면적
  규칙으로는 검색 필요 여부를 안정적으로 판단할 수 없는 실제 사례를 확인 → 의미 기반 판단이 필요하다고 결론.
- **judge의 LLM 호출은 라우팅 함수가 아닌 노드 안에서 실행**: 라우팅 함수는 판단 결과만 읽어 분기하고,
  무거운 연산(LLM 호출)은 노드 안에서 처리하는 것이 LangGraph의 권장 패턴.
- **judge+조건분기는 Workflow, bind_tools+ToolNode는 Agent**: 개발자가 실행 경로를 코드로 고정했는지(Workflow),
  LLM이 실행 중 다음 행동을 스스로 선택하는지(Agent)로 두 그래프를 구분해 설계.

---

## Version

### Patch Version (0.x.N) — Development Stage
- 0.1.0: 순수 RAG 구현 — TIL 42개, 299청크 인덱싱, 환각 방지 검증 (`rag.py`)
- 0.2.0: LangChain 마이그레이션 — LCEL 체인 (`rag_lc.py`)
- 0.3.0: LangGraph StateGraph — judge 노드 + 조건부 라우팅 (`rag_graph.py`)
- 0.3.1: LangGraph ReAct Agent — bind_tools + ToolNode (`rag_graph.py`) ← **now processing next: FastAPI**

### Minor Version (0.N.x) — LLM Provider
- 0.0.x: Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`) 기반 구현 ← **now processing**

---

## 실행 방법

```bash
# 0. 인덱싱 (최초 1회, TIL 문서 → Chroma)
uv run indexing.py

# 1. 순수 RAG 실행
uv run rag.py

# 2. RAG 평가 (규칙 기반 vs LLM-as-Judge)
uv run evaluate.py

# 3. LangChain 기반 RAG 실행
uv run rag_lc.py

# 4. LangGraph StateGraph 실행 (judge + 조건부 라우팅)
uv run rag_graph.py

# 5. LangGraph ReAct Agent 실행 (bind_tools + ToolNode)
uv run rag_graph.py

# 6. FastAPI 서버
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# 7. Docker로 실행
docker compose up -d
```


## 배포된 서버
```
http://32.236.37.151:8000/docs
```
- 인스턴스 중지 상태로 인한 재시작시 변결 될 예정
---

## Data Source

### 지식 베이스
1. 개인 TIL(Today I Learned) 마크다운 문서 42개
   - 부트캠프 학습 기록 (Python, ML/DL, RAG, LangChain, LangGraph 등)
   - `til_notes/` 디렉토리에 위치
   - 전처리 후 299개 청크로 분할, `chroma_db/`에 인덱싱

---

## 앞으로의 계획 (개인 프로젝트 외 커리큘럼)

- 프론트엔드 부분
- TIL -> 야구로 데이터 바꿔서 재학습 시킨 후 구조 다시 짜보기