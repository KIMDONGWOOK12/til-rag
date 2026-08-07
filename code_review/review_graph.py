import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END

from retriever import get_relevant_context

# ── vLLM 클라이언트 (학습시킨 리뷰 생성 모델) ─────────────────
VLLM_URL = "https://democrat-tiring-greedily.ngrok-free.dev/v1"
MODEL_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-merged-v7"
vllm_client = OpenAI(base_url=VLLM_URL, api_key="not-needed")

# ── Anthropic 클라이언트 (판단 전용, 안정적인 모델) ────────────────
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MAX_ATTEMPTS = 3


class ReviewState(TypedDict):
    code: str
    review: str
    attempts: int
    is_relevant: bool


def generate_review(state: ReviewState) -> dict:
    """vLLM으로 코드 리뷰를 생성하는 노드."""
    code = state["code"]
    context = get_relevant_context(code)

    if context:
        prompt = (
            "다음 코드를 리뷰해줘:\n\n"
            + "```\n" + code + "\n```\n\n"
            + "참고로, 아래는 과거에 있었던 유사한 코드 리뷰 사례들이야. "
            + "이 사례들의 리뷰 스타일과 관점만 참고하고, "
            + "반드시 위에서 제시한 코드에 대해서만 리뷰해:\n\n"
            + context
        )
    else:
        prompt = "다음 코드를 리뷰해줘:\n\n```\n" + code + "\n```"

    prompt += "\n\n반드시 한국어로 답변해줘."

    response = vllm_client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    review = response.choices[0].message.content

    return {
        "review": review,
        "attempts": state.get("attempts", 0) + 1,
    }


def judge_review(state: ReviewState) -> dict:
    """Claude로 생성된 리뷰가 실제로 입력 코드에 대한 리뷰인지 판단하는 노드."""
    code = state["code"]
    review = state["review"]

    judge_prompt = f"""다음은 코드와, 그 코드에 대해 어떤 AI가 생성한 리뷰입니다.

[코드]
{code}

[생성된 리뷰]
{review}

이 리뷰가 실제로 위 코드의 내용을 근거로 작성된 것인지 판단해주세요.
다음과 같은 경우는 부적절한 리뷰입니다:
- 코드와 무관한 다른 코드나 주제를 언급함
- 코드에 존재하지 않는 변수/함수/사용자명을 언급함
- 리뷰 내용이 지나치게 짧거나 실질적인 내용이 없음
- 코드를 그대로 복사만 하고 개선점을 제시하지 않음

적절하면 "적절" 한 단어만, 부적절하면 "부적절" 한 단어만 답하세요."""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = response.content[0].text.strip()
    is_relevant = "적절" in verdict and "부적절" not in verdict

    return {"is_relevant": is_relevant}


def route_after_judge(state: ReviewState) -> str:
    if state["is_relevant"]:
        return "translate"      # end 대신 translate로 수정
    if state["attempts"] >= MAX_ATTEMPTS:
        return "fallback"
    return "retry"


def fallback_response(state: ReviewState) -> dict:
    # MAX_ATTEMPTS만큼 시도해도 적절한 리뷰를 만들지 못했을 때의 안전한 응답
    return {
        "review": "죄송합니다. 이 코드에 대한 신뢰할 만한 리뷰를 생성하지 못했습니다. "
                   "다른 방식으로 코드를 다시 붙여넣어 주시면 감사하겠습니다."
    }

def translate_review(state: ReviewState) -> dict:
    """적절하다고 판단된 리뷰를 자연스러운 한국어로 다듬는 노드."""
    review = state["review"]

    translate_prompt = f"""다음은 코드 리뷰 내용입니다. 이 내용을 자연스러운 한국어로
정리해서 다시 작성해주세요. 코드 블록(```)이 있다면 그대로 유지하고,
설명 부분만 한국어로 작성하세요. 원래 내용의 의미는 그대로 유지해야 합니다.

[리뷰 내용]
{review}

한국어로 정리된 리뷰만 출력하세요."""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": translate_prompt}],
    )
    return {"review": response.content[0].text.strip()}

builder = StateGraph(ReviewState)
builder.add_node("generate", generate_review)
builder.add_node("judge", judge_review)
builder.add_node("translate", translate_review)
builder.add_node("fallback", fallback_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "judge")
builder.add_conditional_edges(
    "judge", route_after_judge,
    {"retry": "generate", "translate": "translate", "fallback": "fallback"}
)
builder.add_edge("translate", END)
builder.add_edge("fallback", END)

graph = builder.compile()


def review_code_with_judge(code: str) -> str:
    """외부에서 호출하는 진입점 함수. main.py에서 이걸 사용."""
    result = graph.invoke({"code": code, "attempts": 0})
    return result["review"]


if __name__ == "__main__":
    test_code = '''
def get_user(id):
    users = db.query("SELECT * FROM users WHERE id = " + id)
    return users
'''
    print(review_code_with_judge(test_code))