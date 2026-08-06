from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from retriever import get_relevant_context

VLLM_URL = "https://democrat-tiring-greedily.ngrok-free.dev/v1"
MODEL_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-merged-v4"

client = OpenAI(base_url=VLLM_URL, api_key="not-needed")

def review_code(code: str) -> str:
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
        prompt = (
            "다음 코드를 리뷰해줘:\n\n"
            + "```\n" + code + "\n```"
        )

    prompt += "\n\n반드시 한국어로 답변해줘."

    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_code = '''
def get_user(id):
    users = db.query("SELECT * FROM users WHERE id = " + id)
    return users
'''
    print(review_code(test_code))
