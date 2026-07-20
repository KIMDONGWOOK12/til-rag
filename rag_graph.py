from dotenv import load_dotenv
load_dotenv()

import os
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph

embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    google_api_key = os.environ["GOOGLE_API_KEY"],
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function= embeddings,
    collection_name="til_rag"
)

retreiever = vectorstore.as_retriever(search_kwargs={"k":3})

def format_docs(docs):
    return "\n".join(f"[문서{i}]\n{doc.page_content}" for i , doc in enumerate(docs, 1))

prompt = ChatPromptTemplate([
    ("system",
     "당신은 개인 학습 기록인 TIL 기반 QA 시스템입니다.\n"

    "아래 제공된 문서만을 근거로 질문에 한국어로 답하세요.\n"

    "문서에 답이 없으면 \"제 TIL에서 해당 정보를 찾을 수 없습니다\"라고 답하세요.\n"

    "답변 시 근거가 된 문서 번호를 함께 표시하세요.\n"
	
    "답변은 3문장 이내로 작성하세요.\n\n"

    "{context}"),

    ("human", "질문: {question}")
])

model = ChatGoogleGenerativeAI(
    model = os.environ["GOOGLE_MODEL"],
    google_api_key = os.environ["GOOGLE_API_KEY"],
)

class State(TypedDict):
    question : str
    context : str
    answer : str
    needs_search : bool

def judge(state: State) -> dict:
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "사용자의 입력이 TIL(학습 기록) 문서를 검색해야 답할 수 있는 '지식 질문'인지,\n"
         "아니면 잡담·일지·감정 표현처럼 검색이 필요 없는 말인지 판단하세요.\n"
         "검색이 필요하면 'yes', 필요 없으면 'no'라고만 답하세요."),
        ("human", "{question}"),
    ])
    chain = judge_prompt | model | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    need_search = "yes" in result.lower()
    return {"needs_search": need_search}

def route_after_judge(state: State) -> str:
    if state["needs_search"]:
        return "search"
    return "generate"

def search(state : State) -> dict:
    docs = retreiever.invoke(state["question"])
    return {"context": format_docs(docs)}


def generate(state : State) -> dict:
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer" : answer}

builder = StateGraph(State)

builder.add_node("judge", judge)
builder.add_node("search", search)
builder.add_node("generate", generate)
builder.add_edge(START, "judge")
builder.add_conditional_edges(
    "judge", route_after_judge,
    {
        "search" : "search",
        "generate" : "generate"
    },
)
builder.add_edge("search", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

if __name__ == "__main__":
    result1 = graph.invoke({"question": "RAG가 뭐야?", "context": "", "answer": "", "needs_search": False})
    print("[검색 필요]", result1["answer"])

    result2 = graph.invoke({"question": "오늘 너무 힘들다", "context": "", "answer": "", "needs_search": False})
    print("[검색 불필요]", result2["answer"])