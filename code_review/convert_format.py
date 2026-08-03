import json
with open("code_review/data/sample_data.json", "r") as f:
    raw_data = json.load(f)

training_data = []

for item in raw_data:
    if item["type"] == "review_comment" and item["code"]:
        # 형식은 이제 코드 라인에 달린 리뷰를 코드를 보고 리뷰하라는 뜻
        user_msg = f"다음 코드를 리뷰해줘:\n\n```\n{item['code']}\n```"
    elif item['type'] == "issue_comment":
        continue
    else:
        continue

    training_data.append({
        "message":[
            {"role":"user","content" :user_msg},
            {"role":"assistant", "content": item["review"]},
        ]
    })

print(f"학습용 데이터 {len(training_data)} 개 준비 완료")

with open("code_review/data/training_data.json1","w") as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")