from github import Github
import json
import os
from dotenv import load_dotenv

load_dotenv()
g = Github(os.environ["GITHUB_TOKEN"])

repo_names = [
    # 기존 5개
    "langchain-ai/langgraph",
    "langchain-ai/langchain",
    "pallets/flask",
    "psf/requests",
    "tiangolo/fastapi",
    # 추가 10개
    "django/django",                    # 웹 프레임워크
    "pandas-dev/pandas",                 # 데이터 처리
    "numpy/numpy",                       # 수치 연산
    "scikit-learn/scikit-learn",         # ML
    "python-poetry/poetry",              # 패키지 관리
    "sqlalchemy/sqlalchemy",             # ORM
    "celery/celery",                     # 비동기 작업 큐
    "pallets/click",                     # CLI 도구
    "huggingface/transformers",          # NLP/딥러닝
    "encode/django-rest-framework",      # REST API
]

dataset = []
PR_LIMIT_PER_REPO = 500
MAX_CODE_LENGTH = 1000  # 이 이상이면 잘라내기

for repo_name in repo_names:
    repo = g.get_repo(repo_name)
    checked = 0
    repo_collected = 0
    repo_truncated = 0

    prs = repo.get_pulls(state="closed", sort="created", direction="asc")
    for pr in prs:
        if not pr.merged:
            continue
        checked += 1
        if checked > PR_LIMIT_PER_REPO:
            break

        try:
            for comment in pr.get_review_comments():
                body = comment.body.strip()
                if len(body) < 30:
                    continue
                diff = comment.diff_hunk
                if not diff or len(diff) < 20:
                    continue

                if len(diff) > MAX_CODE_LENGTH:
                    diff = diff[:MAX_CODE_LENGTH] + "\n... (생략)"
                    repo_truncated += 1

                dataset.append({
                    "repo": repo_name,
                    "type": "review_comment",
                    "file": comment.path,
                    "code": diff,
                    "review": body,
                })
                repo_collected += 1
        except Exception as e:
            print(f"{repo_name} PR #{pr.number} 스킵: {e}")

    print(f"{repo_name}: merge된 PR {checked}개에서 코멘트 {repo_collected}개 수집 (그중 {repo_truncated}개 잘림)")

print(f"\n전체 합계: {len(dataset)}개")

with open("code_review/data/sample_data.json", "w") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
