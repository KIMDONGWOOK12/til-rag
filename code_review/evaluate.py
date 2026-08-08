"""
코드리뷰 모델(v5, v6, v7)의 답변 품질을 Gemini로 채점하는 평가 스크립트.
til-rag의 evaluate.py(LLM-as-Judge 패턴)를 코드리뷰 도메인에 맞게 재구성.
"""

import os
import json
from google import genai
from dotenv import load_dotenv
from anthropic import Anthropic



load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# 형이 이미 테스트했던 3개 케이스 + 각 버전의 실제 출력을 기록
test_results = [
    {
        "category": "보안(SQL 인젝션)",
        "code": "def get_user(id):\n    users = db.query(\"SELECT * FROM users WHERE id = \" + id)\n    return users",
        "v5_review": "죄송합니다. 이 코드에 대한 신뢰할 만한 리뷰를 생성하지 못했습니다.",
        "v6_review": "아니요, 이 코드는 안전하지 않습니다. SQL Injection 공격에 취약합니다. 매개변수화된 쿼리를 사용해야 합니다.",
        "v7_review": "다음 코드는 보안상 문제가 있습니다. SQL 인젝션 공격에 취약합니다. 매개변수화된 쿼리를 사용하여...",
    },
    {
        "category": "버그(제로 나누기)",
        "code": "def calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)",
        "v5_review": "빈 리스트에 대한 예외 처리를 추가하여 잘못된 입력값에 대해 명확한 오류 메시지를 반환합니다.",
        "v6_review": "아니요, 그럴 필요는 없습니다. 다만 이 함수의 코드를 더 간결하게 리팩토링할 수 있는 방법이 있는지 살펴봐주세요.",
        "v7_review": "죄송합니다. 이 코드에 대한 신뢰할 만한 리뷰를 생성하지 못했습니다.",
    },
    {
        "category": "정상 코드",
        "code": "def add(a: int, b: int) -> int:\n    return a + b",
        "v5_review": "docstring을 추가하여 함수의 목적과 매개변수를 명확히 하는 것을 제안합니다.",
        "v6_review": "죄송합니다. 이 코드에 대한 신뢰할 만한 리뷰를 생성하지 못했습니다.",
        "v7_review": "docstring을 추가하여 함수의 목적과 매개변수, 반환값을 명시하는 것을 제안합니다.",
    },
]


def judge_with_gemini(code: str, review: str) -> dict:
    prompt = f"""다음은 코드와, 어떤 AI가 생성한 코드 리뷰입니다.

[코드]
{code}

[생성된 리뷰]
{review}

이 리뷰를 1~5점으로 평가해주세요. 평가 기준:
- 코드의 실제 문제를 정확히 지적했는가
- 코드와 무관한 내용은 없는가
- 제안이 실질적이고 구체적인가

다음 JSON 형식으로만 답하세요: {{"score": N, "reason": "이유"}}"""

    response = client.models.generate_content(
        model="gemini-3.5-flash", contents=prompt
    )
    text = response.text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "reason": text}


def judge_with_claude(code: str, review: str) -> dict:
    prompt = f"""다음은 코드와, 어떤 AI가 생성한 코드 리뷰입니다.

[코드]
{code}

[생성된 리뷰]
{review}

이 리뷰를 1~5점으로 평가해주세요. 평가 기준:
- 코드의 실제 문제를 정확히 지적했는가
- 코드와 무관한 내용은 없는가
- 제안이 실질적이고 구체적인가

다음 JSON 형식으로만 답하세요: {{"score": N, "reason": "이유"}}"""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "reason": text}


def run_evaluation():
    all_scores = {"v5": {"gemini": [], "claude": []}, "v6": {"gemini": [], "claude": []}, "v7": {"gemini": [], "claude": []}}

    for case in test_results:
        print(f"\n=== {case['category']} ===")
        for version in ["v5", "v6", "v7"]:
            review = case[f"{version}_review"]
            gemini_result = judge_with_gemini(case["code"], review)
            claude_result = judge_with_claude(case["code"], review)
            if gemini_result.get("score") is not None:
                all_scores[version]["gemini"].append(gemini_result["score"])
            if claude_result.get("score") is not None:
                all_scores[version]["claude"].append(claude_result["score"])
            print(f"{version} - Gemini: {gemini_result.get('score')}점 / Claude: {claude_result.get('score')}점")

    print("\n=== 버전별 평균 점수 ===")
    for version, scores in all_scores.items():
        g_avg = sum(scores["gemini"]) / len(scores["gemini"]) if scores["gemini"] else 0
        c_avg = sum(scores["claude"]) / len(scores["claude"]) if scores["claude"] else 0
        print(f"{version}: Gemini {g_avg:.2f}점 / Claude {c_avg:.2f}점")
        
if __name__ == "__main__":
    run_evaluation()
