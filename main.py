import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from rag_graph import graph

# code_review/rag_review.py가 같은 폴더의 retriever.py를 `from retriever import ...`
# 형태로 불러오기 때문에, 패키지 임포트(from code_review.rag_review import ...) 대신
# code_review 폴더 자체를 sys.path에 얹어 최상위 모듈처럼 불러온다.
CODE_REVIEW_DIR = Path(__file__).parent / "code_review"
sys.path.insert(0, str(CODE_REVIEW_DIR))
from code_review.review_graph import review_code_with_judge as review_code

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

class CodeReviewRequest(BaseModel):
    code: str

@app.post("/review")
def review(req: CodeReviewRequest):
    try:
        review_text = review_code(req.code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"코드 리뷰 서버에 연결할 수 없습니다: {e}")

    return {"review": review_text}