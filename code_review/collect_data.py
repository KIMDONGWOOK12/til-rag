from github import Github
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
g = Github(os.environ["GITHUB_TOKEN"])

repo_names = [
    "langchain-ai/langgraph",
    "langchain-ai/langchain",
    "pallets/flask",
    "psf/requests",
    "tiangolo/fastapi",
]

dataset = []
PR_LIMIT_PER_REPO = 500

for repo_name in repo_names:
    repo = g.get_repo(repo_name)
    checked = 0
    repo_collected = 0

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
                # 너무 짧거나 (LGTM류) 코드가 없으면 제외
                if len(body) < 30:
                    continue
                if not comment.diff_hunk or len(comment.diff_hunk) < 20:
                    continue
                dataset.append({
                    "repo": repo_name,
                    "type": "review_comment",
                    "file": comment.path,
                    "code": comment.diff_hunk,
                    "review": body,
                })
                repo_collected += 1
        except Exception as e:
            print(f"{repo_name} PR #{pr.number} 스킵: {e}")

    print(f"{repo_name}: merge된 PR {checked}개에서 코멘트 {repo_collected}개 수집")

print(f"\n전체 합계: {len(dataset)}개")

with open("code_review/data/sample_data.json", "w") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
