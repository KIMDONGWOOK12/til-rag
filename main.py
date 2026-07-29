from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from rag_graph import graph

app = FastAPI()

TIL_NOTES_DIR = Path(__file__).parent / "til_notes"

@app.get("/")
def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")

@app.get("/docs/{filename}")
def get_til_doc(filename: str):
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")

    file_path = TIL_NOTES_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="해당 문서를 찾을 수 없습니다.")

    return PlainTextResponse(file_path.read_text(encoding="utf-8"))

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