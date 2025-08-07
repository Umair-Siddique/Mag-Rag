import os
import json
from flask import Blueprint, request, jsonify, current_app
from openai import OpenAI
from langchain_experimental.text_splitter import SemanticChunker
import re

upload_document_bp = Blueprint('document', __name__)

PINECONE_INDEX = "career-counseling-documents"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    text = re.sub(r'[^\w\s\.,\'"\!\?\-\(\)\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def authenticate(token):
    try:
        user = current_app.supabase.auth.get_user(token)
        return user.user is not None
    except Exception:
        return False

# Generate Metadata with GPT-4 Turbo
def generate_metadata(chunks):
    joined_chunks = "\n\n".join(chunks[:14])
    prompt = f"""
You are provided with several chunks of text from a document.

Your task:
- Analyze and summarize the document briefly.
- Provide 3 to 5 clear and distinct short topics/themes discussed in the document (each topic should be 2-3 words).
- List 5 to 10 concise keywords highly relevant to the document (each keyword should ideally be a single word or a short phrase of maximum 2 words).

Provide your output strictly in this JSON format:
{{
  "topics": ["short topic1", "short topic2", "..."],
  "keywords": ["short keyword1", "short keyword2", "..."]
}}

Text chunks:
{joined_chunks}

Return JSON only. Do not add explanations or additional formatting.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5
    )

    return json.loads(response.choices[0].message.content)

# Flask route
@upload_document_bp.route('/upload', methods=['POST'])
def upload_and_embed():
    token = request.headers.get('Authorization')
    if not token or not authenticate(token):
        return jsonify({"error": "Unauthorized access."}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    namespace = request.form.get('namespace', 'default_namespace')

    if file.filename == '' or not file.filename.endswith('.txt'):
        return jsonify({"error": "Invalid file type. Please upload a .txt file."}), 400

    try:
        raw_text = file.read().decode('utf-8')
        processed_text = preprocess_text(raw_text)

        semantic_chunker = SemanticChunker(
            current_app.embeddings_model,
            breakpoint_threshold_type="gradient",
            min_chunk_size=650,
            breakpoint_threshold_amount=77.5
        )

        semantic_chunks = semantic_chunker.create_documents([processed_text])
        chunks = [chunk.page_content for chunk in semantic_chunks]

        metadata = generate_metadata(chunks)

        index = current_app.pinecone.Index(PINECONE_INDEX)

        for idx, chunk in enumerate(chunks):
            embedding_vector = current_app.embeddings_model.embed_documents([chunk])[0]

            record = {
                'id': f"{file.filename}_chunk_{idx}",
                'values': embedding_vector,
                'metadata': {
                    'filename': file.filename,
                    'topics': metadata['topics'],
                    'keywords': metadata['keywords'],
                    'content': chunk
                }
            }

            index.upsert(vectors=[record], namespace=namespace)

        return jsonify({
            "message": f"File '{file.filename}' successfully uploaded and processed."
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
