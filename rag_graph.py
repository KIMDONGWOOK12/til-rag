from dotenv import load_dotenv
load_dotenv()

import os
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage


embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    google_api_key = os.environ["GOOGLE_API_KEY"],
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="til_rag"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

def format_docs(docs):
    return "\n".join(f"[문서{i}]\n{doc.page_content}"for i, doc in enumerate(docs, 1))

#search 도구

@tool
def search_til(query: str) -> str:
    """개인 학습 기록(TIL) 문서에서 정보를 검색합니다. 지식 질문에만 사용하세요."""
    docs = retriever.invoke(query)
    return format_docs(docs)

tools = [search_til]

# model에 도구 연결
model = ChatGoogleGenerativeAI(
    model = os.environ["GOOGLE_MODEL"],
    google_api_key = os.environ["GOOGLE_API_KEY"],
)
model_with_tools = model.bind_tools(tools)  

SYSTEM_PROMPT = (
    "당신은 개인 학습 기록인 TIL 기반 QA 시스템입니다.\n"
    "지식 질문이면 search_til 도구로 TIL 문서를 검색해 그 내용만 근거로 답하세요.\n"
    "문서에 답이 없으면 \"제 TIL에서 해당 정보를 찾을 수 없습니다\"라고 답하세요.\n"
    "잡담이나 감정 표현처럼 검색이 필요 없는 말에는 도구를 쓰지 말고 자연스럽게 응답하세요.\n"
    "답변 시 근거가 된 문서 번호를 함께 표시하고, 3문장 이내로 작성하세요."
)

def agent(state: MessagesState) -> dict:
    messages = state["messages"]
    if not any(m.type == "system" for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = model_with_tools.invoke(messages)
    return {"messages" : [response]}

builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile()

if __name__ == "__main__":
    result1 = graph.invoke({"messages" : [HumanMessage(content="RAG가 뭐야")]})
    print("[검색 필요]", result1["messages"][-1].content)

    result2 = graph.invoke({"messages" : [HumanMessage(content="이번주 비가 너무 많이오네")]})
    print("[검색 필요]", result2["messages"][-1].content)

    '''mermaid_code = graph.get_graph().draw_mermaid()
    print("\n=== Mermaid 다이어그램 코드 ===")
    print(mermaid_code)'''