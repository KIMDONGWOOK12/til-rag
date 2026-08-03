from github import Github
import json
import os
from dotenv import load_dotenv

load_dotenv()
g = Github(os.environ["GITHUB_TOKEN"])

repo_names = ["langchain-ai/langgraph", "langchain-ai/langchain"]
dataset = []

for repo_name in repo_names:
    repo = g.get_repo(repo_name)
    checked = 0
    repo_collected = 0

    prs = repo.get_pulls(state="closed", sort="created", direction="asc")
    for pr in prs:
        if not pr.merged:
            continue
        checked += 1
        if checked > 300:
            break

        try:
            # 코드 라인에 달린 리뷰 코멘트
            for comment in pr.get_review_comments():
                if len(comment.body) > 20:
                    dataset.append({
                        "repo": repo_name,
                        "type": "review_comment",
                        "file": comment.path,
                        "code": comment.diff_hunk,
                        "review": comment.body,
                    })
                    repo_collected += 1

            # PR 전체에 대한 일반 코멘트
            for comment in pr.get_issue_comments():
                if len(comment.body) > 20:
                    dataset.append({
                        "repo": repo_name,
                        "type": "issue_comment",
                        "file": None,
                        "code": None,
                        "review": comment.body,
                    })
                    repo_collected += 1
        except Exception as e:
            print(f"{repo_name} PR #{pr.number} 스킵: {e}")

    print(f"{repo_name}: merge된 PR {checked}개에서 코멘트 {repo_collected}개 수집")

print(f"\n전체 합계: {len(dataset)}개")

with open("code_review/data/sample_data.json", "w") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)