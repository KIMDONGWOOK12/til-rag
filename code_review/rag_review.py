from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from retriever import retriever, format_docs

VLLM_URL = "https://democrat-tiring-greedily.ngrok-free.dev/v1"
MODEL_PATH = "/content/drive/MyDrive/code_review_qlora/qwen-code-review-merged"

client = OpenAI(base_url=VLLM_URL, api_key="not-needed")

def review_code(code: str) -> str:
    docs = retriever.invoke(code)
    context = format_docs(docs)

    prompt = (
        "아래는 과거의 유사한 코드 리뷰 사례입니다:\n\n"
        + context
        + "\n\n이 사례들을 참고해서, 다음 코드를 리뷰해줘:\n\n"
        + "```\n" + code + "\n```"
    )

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