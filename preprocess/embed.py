import os
import json
import uuid
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import re


from dotenv import load_dotenv
load_dotenv()


# --- Config ---
class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    JSON_FOLDER = "./final_operations"  # Folder containing your JSON files
    NAMESPACE = "common"
    INDEX_NAME = "career-counseling-documents"

def remove_markdown(text):
    # Remove **bold**, *italic*, __underline__, etc.
    text = re.sub(r"(\*{1,2}|_{1,2})(.*?)\1", r"\2", text)
    # Remove markdown headings like ##, ###, etc.
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove inline links [text](url)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    # Remove leftover backticks
    text = re.sub(r"`+", "", text)
    return text.strip()

# --- Initialize Clients ---
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(Config.INDEX_NAME)
embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL, api_key=Config.OPENAI_API_KEY)

# --- Load and Embed ---
for filename in os.listdir(Config.JSON_FOLDER):
    if filename.endswith(".json"):
        with open(os.path.join(Config.JSON_FOLDER, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Metadata shared across all chunks from this file
        metadata = {
            "filename": data["filename"],
            "keywords": data["keywords"],
            "topics": data["topics"]
        }
        
        for i, chunk in enumerate(data.get("summarized_chunks", [])):
            clean_chunk = remove_markdown(chunk)
            vector_id = str(uuid.uuid4())
            vector_values = embeddings.embed_query(clean_chunk)

            index.upsert(
                vectors=[
                    {
                        "id": vector_id,
                        "values": vector_values,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "text": clean_chunk
                        }
                    }
                ],
                namespace=Config.NAMESPACE
            )

print("✅ All JSON chunks embedded into Pinecone successfully!")
