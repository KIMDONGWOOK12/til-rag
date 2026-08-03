from dotenv import load_dotenv
load_dotenv()

import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

vectorstore = Chroma(
    persist_directory="./code_review_db",
    embedding_function=embeddings,
    collection_name="code_review",
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

def format_docs(docs):
    return "\n\n".join(f"[사례{i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1))