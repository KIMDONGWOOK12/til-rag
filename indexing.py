from dotenv import load_dotenv
import os
import glob
import re
import unicodedata
import chromadb
import time
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# 마크다운 파일은 텍스트 기반 형식이라 그대로 읽음
def load_markdown(filepath):
    # UTF-8 인코딩으로 파일 전체를 읽어 문자열로 반환하기
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
    

# 텍스트 전처리 함수 정의 (Text Preprocessing)
def preprocess(text):
    #유니코드 정규화 + 불필요한 문자 제거 + 연속 공백 및 줄바꿈 정리해버리기
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]","",text)  # 화면에 잘 보이지 않는 제어 문자를 제거
    text = re.sub(r"\n{3,}", "\n\n", text)                  # 3줄 이상 연속된 줄바꿈을 2줄로 줄여서 단락 구조만 남기기
    text = re.sub(r" {2,}", " ", text)                      # 연속 공백은 1줄로 줄여서 텍스트를 이뽀게 해버리기

    return text.strip() # 양 옆 공백이나 특정 문자 제거

def chunk_text(text, chunk_size=500, chunk_overlap=50):

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size                # 현재 청크의 끝 위치를 계산
        chunks.append(text[start:end])           # 시작부터 마지막 전까지 잘라서 청크로 저장
        start += chunk_size - chunk_overlap     # 다음 청크 시작 위치로 간다음에 이지ㅔ overlap만큼 겹치도록 하기

    return chunks


# embedding gemini
def embed_texts(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001", contents=texts
    )
    return [e.values for e in result.embeddings]


files = glob.glob("til_notes/*.md")
print(f"파일 수 : {len(files)}")

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(
    name="til_rag", metadata={"hnsw:space":"cosine"}
)

for filepath in files:
    raw = load_markdown(filepath)       # 로딩
    cleaned = preprocess(raw)           # 전처리
    chunks = chunk_text(cleaned)        # chunking
    embeddings = embed_texts(chunks)    # Embedding

    filename = os.path.basename(filepath)  # AI 쓴 부분 이해 노력중
    collection.add(
        ids=[f"{filename}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filename} for i in range(len(chunks))],
    )
    print(f"[완료] {filename} → {len(chunks)}개 청크")
    time.sleep(20)
print(f"\n총 저장된 청크: {collection.count()}")
