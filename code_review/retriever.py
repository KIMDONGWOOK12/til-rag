from dotenv import load_dotenv
load_dotenv()

import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

vectorstore = Chroma(
    persist_directory="./code_review_db",
    embedding_function=embeddings,
    collection_name="code_review",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(f"[사례{i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1))


def get_relevant_context(query: str, k: int = 3, threshold: float = 0.30) -> str:
    """유사도 점수가 threshold보다 낮은(=충분히 비슷한) 문서만 컨텍스트로 사용.
    Chroma의 similarity_search_with_score는 거리(distance) 기준이라
    값이 작을수록 더 유사함."""
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    relevant = [doc for doc, score in docs_with_scores if score < threshold]

    if not relevant:
        return ""

    return "\n\n".join(f"[사례{i}]\n{doc.page_content}" for i, doc in enumerate(relevant, 1))
