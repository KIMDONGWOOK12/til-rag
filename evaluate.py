from dotenv import load_dotenv
import os
import time
import chromadb
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(
    name="til_rag", metadata={"hnsw:space":"cosine"}
)

# retreieve
def embed_texts(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001", contents= texts
    )
    return [e.values for e in result.embeddings]

def retrieve(question, n_results= 3):
    q_emb = embed_texts([question])[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results
    )
    return results["documents"][0]

# prompt
def generate(contexts, question):
    docs_text = ""
    for i, chunk in enumerate(contexts, 1):
        docs_text += f"\n[문서{i}]\n{chunk}\n"

    prompt = f"""
    당신은 개인 학습 기록인 TIL 기반 QA 시스템입니다. 
    아래 제공된 문서만을 근거로 질문에 한국어로 답하세요.
    문서에 답이 없으면 "제 TIL에서 해당 정보를 찾을 수 없습니다" 라고 답 하세요.
    답변 시 근거가 된 문서 번호를 함께 표시 하세요.
    답변은 3문장 이내로 작성하세요.
    {docs_text}
    질문 : {question}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text.strip()



eval_questions = [
    {
        "question": "랜덤포레스트가 무엇인가요?",
        "ground_truth": "의사결정 트리 기반 앙상블 학습 기법"
    },
    {
        "question": "RAG는 무엇인가요?",
        "ground_truth": "외부 데이터 저장소에서 관련 문서를 검색하고, 그 결과를 LLM 프롬프트에 삽입하여 근거 기반의 답변을 생성하는 아키텍처입니다."
    },
    {
        "question": "머신러닝은 무엇인가요?",
        "ground_truth": "경험을 통해 자동으로 개선하는 컴퓨터 알고리즘의 연구분야입니다."

    },
    {
        "question": "파인튜닝은 무엇인가요?",
        "ground_truth": "사전 훈련된 모델을 새로운 데이터셋에 맞춰 미세 조정하여 성능을 최적화하는 과정"

    },
    {
        "question": "데이터 전처리는 무엇인가요?",
        "ground_truth": "데이터를 분석 및 모델 학습을 위해 원본 데이터를 정제하고 변환하는 과정 입니다."
    }
]

eval_dataset = []

for item in eval_questions:
    #1. retreieval 실행
    contexts = retrieve(item["question"])

    time.sleep(3)

    # 2. generation 실행
    answer = generate(contexts, item["question"])

    time.sleep(3)

    #3. 평가 데이터셋 1건 와ㄴ성
    eval_dataset.append({
        "question": item["question"],
        "ground_truth" : item["ground_truth"],
        "contexts": contexts,
        "answer": answer
    })

    # 실행상태 황긴하기
    print(f"Q:{item['question']}")
    print(f"A:{answer[:80]}...")
    print()


# === 5개 질문 x 4개 지표 평가 실행 ===

# 질문별 평가 점수를 저장할 리스트입니다.
results = []


def simple_faithfulness(answer, contexts):
    """답변의 각 문장이 검색된 문맥에 포함되는지 확인합니다."""
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 5]
    if not sentences:
        return 1.0
    supported = 0
    for sent in sentences:
        for ctx in contexts:
            if sent in ctx:
                supported += 1
                break
    return supported / len(sentences)

def simple_answer_relevancy(answer, question):
    """답변에 질문의 핵심 키워드가 포함되는지 확인합니다."""
    keywords = [w for w in question.replace("?", "").split() if len(w) > 1]
    if not keywords:
        return 1.0
    found = sum(1 for kw in keywords if kw in answer)
    return found / len(keywords)


def simple_context_precision(question, contexts):
    """검색된 문맥 중 질문 키워드를 포함하는 비율을 계산합니다."""
    keywords = [w for w in question.replace("?", "").split() if len(w) > 1]
    if not keywords or not contexts:
        return 0.0
    relevant = 0
    for ctx in contexts:
        if any(kw in ctx for kw in keywords):
            relevant += 1
    return relevant / len(contexts)

def simple_context_recall(ground_truth, contexts):
    """정답의 핵심 키워드가 검색된 문맥에 포함되는 비율을 계산합니다."""
    keywords = [w for w in ground_truth.replace(".", "").split() if len(w) > 1]
    if not keywords:
        return 1.0
    ctx_text = " ".join(contexts)
    found = sum(1 for kw in keywords if kw in ctx_text)
    return found / len(keywords)


# eval_dataset에 들어 있는 각 질문-정답-문맥-답변 쌍을 하나씩 평가합니다.
for item in eval_dataset:
    # 4개 평가 함수를 실행하여 점수를 계산합니다.
    scores = {
        # 답변이 검색된 문맥에 얼마나 충실한지 평가합니다.
        "faithfulness": simple_faithfulness(item["answer"], item["contexts"]),

        # 답변이 질문 의도에 얼마나 잘 맞는지 평가합니다.
        "answer_relevancy": simple_answer_relevancy(item["answer"], item["question"]),

        # 검색된 문맥 중 질문과 실제로 관련 있는 비율을 평가합니다.
        "context_precision": simple_context_precision(item["question"], item["contexts"]),

        # 정답에 필요한 근거가 검색된 문맥에 얼마나 포함되었는지 평가합니다.
        "context_recall": simple_context_recall(item["ground_truth"], item["contexts"]),
    }

    # 질문과 4개 점수를 함께 results에 저장합니다.
    results.append({"question": item["question"], **scores})



# === 결과 출력 ===

# 평균을 계산할 지표 이름 목록입니다.
metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

print("=== RAG Evaluation 결과 ===")
print()

# 질문별 평가 결과를 순서대로 출력합니다.
for r in results:
    print(f"  질문: {r['question']}")
    print(f"  Faithfulness: {r['faithfulness']:.2f}  |  Answer Relevancy: {r['answer_relevancy']:.2f}  |  Context Precision: {r['context_precision']:.2f}  |  Context Recall: {r['context_recall']:.2f}")
    print()

# 평균 점수
# 5개 질문의 각 지표 평균을 계산합니다.
avg = {m: sum(r[m] for r in results) / len(results) for m in metrics}

print("--- 평균 ---")
print(f"  Faithfulness: {avg['faithfulness']:.2f}  |  Answer Relevancy: {avg['answer_relevancy']:.2f}  |  Context Precision: {avg['context_precision']:.2f}  |  Context Recall: {avg['context_recall']:.2f}")


# llm judge 함수
def llm_judge(metric, question, ground_truth, contexts, answer):
    ctx_text = "\n---\n".join(contexts)

    prompts = {
        "faithfulness": f"""다음 답변의 각 주장이 검색된 문맥에 의해 뒷받침되는지 평가하세요.
문맥에 없는 내용을 답변에서 주장하면 0점, 모든 주장이 문맥에 있으면 1점입니다.
0과 1 사이의 소수로 점수만 출력하세요.

[검색된 문맥]
{ctx_text}

[답변]
{answer}""",

        "answer_relevancy": f"""다음 답변이 질문의 의도에 얼마나 정확히 맞는지 평가하세요.
질문의 핵심을 정확히 답하면 1점, 전혀 관련 없으면 0점입니다.
0과 1 사이의 소수로 점수만 출력하세요.

[질문]
{question}

[답변]
{answer}""",

        "context_precision": f"""다음 검색된 문맥들 중에서 질문에 답하는 데 실제로 도움이 되는 문맥의 비율을 평가하세요.
모두 도움이 되면 1점, 모두 무관하면 0점입니다.
0과 1 사이의 소수로 점수만 출력하세요.

[질문]
{question}

[검색된 문맥]
{ctx_text}""",

        "context_recall": f"""다음 정답에 포함된 핵심 정보가 검색된 문맥 안에 얼마나 포함되어 있는지 평가하세요.
정답의 모든 핵심 정보가 문맥에 있으면 1점, 전혀 없으면 0점입니다.
0과 1 사이의 소수로 점수만 출력하세요.

[정답]
{ground_truth}

[검색된 문맥]
{ctx_text}"""
    }

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompts[metric]
    )
    try:
        return float(response.text.strip())
    except ValueError:
        return 0.0
    

print("\n=== LLM-as-a-Judge vs 규칙 기반 ===\n")

sample = eval_dataset[1]      # 비교하고 싶은 질문 인덱스
rule = results[1]             # 같은 인덱스의 규칙 기반 점수

llm_scores = {}
for m in metrics:
    llm_scores[m] = llm_judge(
        m, sample["question"], sample["ground_truth"],
        sample["contexts"], sample["answer"]
    )
    time.sleep(6)             # 분당 10회 제한 회피

print(f"질문: {sample['question']}")
print(f"[규칙 기반]  Faith: {rule['faithfulness']:.2f} | Relev: {rule['answer_relevancy']:.2f} | Prec: {rule['context_precision']:.2f} | Recall: {rule['context_recall']:.2f}")
print(f"[LLM Judge]  Faith: {llm_scores['faithfulness']:.2f} | Relev: {llm_scores['answer_relevancy']:.2f} | Prec: {llm_scores['context_precision']:.2f} | Recall: {llm_scores['context_recall']:.2f}")