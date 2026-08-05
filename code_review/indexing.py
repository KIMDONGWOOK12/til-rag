from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import chromadb
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def embed_texts(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001", contents=texts
    )
    return [e.values for e in result.embeddings]

with open("code_review/data/sample_data.json", "r") as f:
    data = json.load(f)

client_db = chromadb.PersistentClient(path="./code_review_db")
collection = client_db.get_or_create_collection(
    name="code_review", metadata={"hnsw:space": "cosine"}
)

already_saved = collection.count()
print(f"이미 저장된 개수: {already_saved}, 전체: {len(data)}")

batch_size = 20
for i in range(already_saved, len(data), batch_size):
    batch = data[i:i+batch_size]
    texts = [f"[코드]\n{item.get('code','')}\n\n[리뷰]\n{item['review']}" for item in batch]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[f"item_{i+j}" for j in range(len(batch))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "repo": item.get("repo") or "",
                "file": item.get("file") or "",
            }
            for item in batch
        ],
    )
    print(f"[{i+len(batch)}/{len(data)}] 저장 완료")
    time.sleep(20)

print(f"총 저장된 항목: {collection.count()}")
