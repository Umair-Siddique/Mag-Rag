# import os
# import uuid
# import re
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # --- Configuration Loading ---
# load_dotenv()

# class Config:
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# # --- Constants ---
# PINECONE_INDEX_NAME = "career-counseling-documents"
# NAMESPACE = "insights"
# BATCH_SIZE = 5

# # --- Path Correction ---
# try:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)
# except NameError:
#     project_root = os.path.abspath('.')

# DATA_PATH = os.path.join(project_root, "extracted_text_insights")

# # --- NEW: Helper function to create safe, ASCII-only IDs ---
# def sanitize_for_pinecone_id(text: str) -> str:
#     """Encodes text to ASCII, ignoring errors, to create a safe ID."""
#     return text.encode('ascii', 'ignore').decode('ascii')

# def initialize_services():
#     """Initializes and returns Pinecone and OpenAI clients."""
#     print("Initializing services...")
#     pinecone = Pinecone(api_key=Config.PINECONE_API_KEY)
#     embeddings_model = OpenAIEmbeddings(
#         api_key=Config.OPENAI_API_KEY,
#         model=Config.EMBEDDING_MODEL
#     )
#     print("✅ Services initialized successfully.")
#     return pinecone, embeddings_model

# def process_and_upload():
#     """
#     Processes files by batching texts for embedding and upserting to Pinecone.
#     Skips any batch that encounters an error.
#     """
#     try:
#         pinecone, embeddings_model = initialize_services()
#         index = pinecone.Index(PINECONE_INDEX_NAME)
#         print(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}'...")
#         print(index.describe_index_stats())

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000,
#             chunk_overlap=200,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )

#         if not os.path.isdir(DATA_PATH):
#             print(f"❌ Error: Data directory not found at '{DATA_PATH}'")
#             return

#         files_to_process = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
#         if not files_to_process:
#             print(f"No .txt files found in '{DATA_PATH}'.")
#             return
        
#         print(f"Found {len(files_to_process)} files to process...")
        
#         text_batch = []
#         source_batch = []
#         total_chunks_processed = 0

#         # --- MODIFIED: Removed tqdm for cleaner output ---
#         for filename in files_to_process:
#             file_path = os.path.join(DATA_PATH, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#             except Exception as e:
#                 print(f"⚠️ Could not read file {filename}. Skipping. Error: {e}")
#                 continue

#             chunks = text_splitter.split_text(content)

#             for chunk_text in chunks:
#                 text_batch.append(chunk_text)
#                 source_batch.append(filename)

#                 if len(text_batch) >= BATCH_SIZE:
#                     # --- MODIFIED: Added robust error handling per batch ---
#                     try:
#                         embeddings = embeddings_model.embed_documents(text_batch)
#                         vectors_to_upsert = []
#                         for i, text in enumerate(text_batch):
#                             sanitized_source = sanitize_for_pinecone_id(source_batch[i])
#                             vector_id = f"{sanitized_source}-{uuid.uuid4()}"
#                             vectors_to_upsert.append({
#                                 "id": vector_id,
#                                 "values": embeddings[i],
#                                 "metadata": {"source": source_batch[i], "text": text}
#                             })
                        
#                         index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
#                         total_chunks_processed += len(vectors_to_upsert)

#                     except Exception as e:
#                         print(f"❌ Failed to process a batch. Skipping. Error: {e}")
#                     finally:
#                         # Clear batch to proceed to the next one, regardless of success or failure
#                         text_batch.clear()
#                         source_batch.clear()
        
#         # --- Process the final remaining batch ---
#         if text_batch:
#             try:
#                 embeddings = embeddings_model.embed_documents(text_batch)
#                 vectors_to_upsert = []
#                 for i, text in enumerate(text_batch):
#                     sanitized_source = sanitize_for_pinecone_id(source_batch[i])
#                     vector_id = f"{sanitized_source}-{uuid.uuid4()}"
#                     vectors_to_upsert.append({
#                         "id": vector_id,
#                         "values": embeddings[i],
#                         "metadata": {"source": source_batch[i], "text": text}
#                     })

#                 index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
#                 total_chunks_processed += len(vectors_to_upsert)

#             except Exception as e:
#                 print(f"❌ Failed to process the final batch. Error: {e}")

#         print("\n--- ✅ Pipeline finished successfully! ---")
#         print(f"Total chunks processed and upserted: {total_chunks_processed}")
#         print("Final index stats:")
#         print(index.describe_index_stats())

#     except Exception as e:
#         print(f"❌ A critical error occurred during the pipeline execution: {e}")

# if __name__ == "__main__":
#     process_and_upload()



import os
import json
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# --- Configuration Loading ---
load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# --- Pinecone Initialization ---
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index("career-counseling-documents")

# --- Embedding Initialization ---
embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL, api_key=Config.OPENAI_API_KEY)

# --- Embedding Pipeline ---
def embed_and_store_documents(folder_path, namespace):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                content = data['content']
                metadata = data.get('metadata', {})

                # Generate embedding
                embedding_vector = embeddings.embed_documents([content])[0]

                # Prepare the record for Pinecone
                record = {
                    'id': str(uuid.uuid4()),
                    'values': embedding_vector,
                    'metadata': {
                        'filename': data['filename'],
                        'content': content,
                        'topics': metadata.get('topics', []),
                        'keywords': metadata.get('keywords', [])
                    }
                }

                # Store embedding and metadata in Pinecone within the specified namespace
                index.upsert(vectors=[record], namespace=namespace)

                print(f"Embedded and stored: {filename} in namespace: {namespace}")


# Example usage
if __name__ == "__main__":
    embed_and_store_documents("final_insights", "insights")