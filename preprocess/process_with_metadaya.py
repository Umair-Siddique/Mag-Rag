import os
import json
import faiss
import numpy as np
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Setup embeddings for semantic chunking
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
semantic_chunker = SemanticChunker(
    openai_embeddings,
    breakpoint_threshold_type="gradient",
    min_chunk_size=500,
    breakpoint_threshold_amount=75.0
)

# Setup embeddings for FAISS indexing
embeddings_model = OllamaEmbeddings(model='mxbai-embed-large:335m')
dimension = 1024
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
groq_client = Groq(api_key= os.getenv('GROQ_API_KEY'))

# Recursive Chunking Function (for fallback)
def chunk_text_recursively(text, chunk_size=2500, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# Semantic Chunking Function
def semantic_chunk_text(text):
    return semantic_chunker.create_documents([text])

# Embed and store chunks in FAISS
def embed_and_store_chunks(chunks, embeddings_model, faiss_index):
    vectors, ids = [], []
    for idx, chunk in enumerate(chunks):
        vector = embeddings_model.embed_query(chunk)
        vectors.append(vector)
        ids.append(idx)
    vectors = np.array(vectors).astype('float32')
    faiss_index.add_with_ids(vectors, np.array(ids))

# Generate metadata using RAG
def generate_metadata_rag(text, faiss_index, embeddings_model, groq_client):
    chunks = semantic_chunk_text(text)
    text_chunks = [chunk.page_content for chunk in chunks]
    embed_and_store_chunks(text_chunks, embeddings_model, faiss_index)

    joined_chunks = "\n\n".join(text_chunks[:8])

    prompt = f"""
You are provided with several chunks of text from a document.

Your task:
- Analyze and summarize the document briefly.
- Provide 3 to 5 clear and distinct short topics/themes discussed in the document (each topic should be 2-4 words).
- List 5 to 10 concise keywords highly relevant to the document (each keyword should ideally be a single word or a short phrase of maximum 3 words).

Provide your output strictly in this JSON format:
{{
  "topics": ["short topic1", "short topic2", "..."],
  "keywords": ["short keyword1", "short keyword2", "..."]
}}

Text chunks:
{joined_chunks}

Return JSON only. Do not add explanations or additional formatting.
"""

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=512,
        stream=False
    )

    response_content = completion.choices[0].message.content.strip()

    # Remove markdown backticks if present
    if response_content.startswith("```"):
        response_content = response_content.strip("```").strip()

    try:
        metadata = json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"Metadata JSON parse error: {e}")
        print("Raw LLM response:", response_content)
        metadata = {"topics": [], "keywords": []}

    return metadata, text_chunks

# Process all files in the provided folder
input_folder = r"extracted_data\extracted_text_insights"
output_folder = "final_insights"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.txt'):
        input_file = os.path.join(input_folder, filename)

        with open(input_file, 'r', encoding='utf-8') as f:
            text_content = f.read()

        metadata, chunks = generate_metadata_rag(text_content, faiss_index, embeddings_model, groq_client)
        
        # Save metadata and each chunk separately
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "filename": filename,
                "chunk_id": idx,
                "content": chunk,
                "metadata": metadata
            }

            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_chunk{idx}_metadata.json")

            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(chunk_metadata, json_file, ensure_ascii=False, indent=4)

        print(f"✅ Metadata and chunks stored for '{filename}'")
