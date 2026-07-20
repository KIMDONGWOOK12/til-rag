from dotenv import load_dotenv
load_dotenv()

import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="til_rag",
)

# retriever (rag.py의 retrieve() 대체)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 문서 포맷 (rag.py의 [문서{i}] 로직 대체)
def format_docs(docs):
    return "\n".join(f"[문서{i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1))

# 프롬프트 (rag.py의 f-string 대체)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 개인 학습 기록인 TIL 기반 QA 시스템입니다.\n"
     "아래 제공된 문서만을 근거로 질문에 한국어로 답하세요.\n"
     "문서에 답이 없으면 \"제 TIL에서 해당 정보를 찾을 수 없습니다\"라고 답하세요.\n"
     "답변 시 근거가 된 문서 번호를 함께 표시하세요.\n"
     "답변은 3문장 이내로 작성하세요.\n\n"
     "{context}"),
    ("human", "질문: {question}"),
])

# 모델 (rag.py의 client.models.generate_content 대체)
model = ChatGoogleGenerativeAI(
    model=os.environ["GOOGLE_MODEL"],
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

# 체인 조립
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "김치찌개 맛있게 끓이는 법 알려줘"
    print(rag_chain.invoke(question))