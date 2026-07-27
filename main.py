from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from rag_graph import graph

app = FastAPI()

@app.get("/")
def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")

class Question(BaseModel):
    question : str

@app.post("/ask")
def ask(q: Question):
    result = graph.invoke({"messages": [HumanMessage(content=q.question)]})
    last_message = result["messages"][-1].content

    if isinstance(last_message, list):
        answer = "".join(part.get("text", "")for part in last_message if isinstance(part, dict))

    else:
        answer = last_message

    return {"answer":answer}