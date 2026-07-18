from dotenv import load_dotenv
import os
import chromadb
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(                # 여기서 키 포인트는 persistentClient 같은 경로라서 인덱싱.py에서 저장한걸 여는것
    name = "til_rag", metadata={"hnsw:space":"cosine"}
)

# retreieve
def embed_texts(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001", contents=texts
    )
    return [e.values for e in result.embeddings]

def retrieve(question, n_results=3):
    q_emb = embed_texts([question])[0]          # 질문은 벡터로 해버리기
    results = collection.query(
        query_embeddings=[q_emb], 
        n_results=n_results
    )
    return results["documents"][0]


# prompt + generate
def generate(contexts, question):
    docs_text = "" 
    for i, chunk in enumerate(contexts, 1):
        docs_text += f"\n[문서{i}]\n{chunk}\n"


    prompt = f"""당신은 개인 학습 기록인 TIL 기반 QA 시스템입니다. 
    아래 제공된 문서만을 근거로 질문에 한국어로 답하세요.
    문서에 답이 없으면 "제 TIL에서 해당 정보를 찾을 수 없습니다" 라고 답 하세요.
    답변 시 근거가 된 문서 번호를 함께 표시 하세요.
    답변은 3문장 이내로 작성하세요.
    {docs_text}
    질문 : {question}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text.strip()

question = "김치찌개 맛있게 끓이는 법 알려줘"

contexts = retrieve(question)
print("=====search chunk===")
for i, c in enumerate(contexts, 1):
    print(f"[{i}]{c[:80]}...")

answer = generate(contexts, question)
print("\n===답변===")
print(answer)