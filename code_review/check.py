from github import Github
import os
from dotenv import load_dotenv

load_dotenv()
g = Github(os.environ["GITHUB_TOKEN"])
repo = g.get_repo("langchain-ai/langchain")

# 정렬 기준을 바꿔서, 코멘트 많았을 것 같은 오래된 시기도 확인
prs = repo.get_pulls(state="closed", sort="created", direction="asc")  # 오래된 것부터
checked = 0
total_review = 0
total_issue = 0
for pr in prs:
    if not pr.merged:
        continue
    checked += 1
    if checked > 300:   # 오래된 PR 300개까지 스캔
        break
    rc = pr.get_review_comments().totalCount
    ic = pr.get_issue_comments().totalCount
    total_review += rc
    total_issue += ic

print(f"확인한 PR 수: {checked}")
print(f"review_comments 합계: {total_review}")
print(f"issue_comments 합계: {total_issue}")