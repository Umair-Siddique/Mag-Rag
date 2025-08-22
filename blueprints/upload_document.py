import os
import json
import uuid
from flask import Blueprint, request, jsonify, current_app
from openai import OpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

upload_document_bp = Blueprint('document', __name__)

PINECONE_INDEX = "career-counseling-documents"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
TOKENS_PER_CHUNK = 2500
TOKENS_OVERLAP = 250
SEPARATORS = [
    "\n\n", "\r\n\r\n",       # paragraphs
    "\n", "\r\n",             # lines
    ". ", "! ", "? ",         # sentence ends (with trailing space)
    ".", "!", "?",            # sentence ends (no space)
]

# Enhanced preprocessing (similar to pipeline.py)
def preprocess_text(text):
    """
    Enhanced text preprocessing similar to pipeline.py
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove references/citations like [1], [2,3], etc.
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    
    # Remove standalone numeric lines (often page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove sequences of dots, underscores, or hyphens longer than 3
    text = re.sub(r'(\.\s*){3,}', ' ', text)
    text = re.sub(r'_{3,}', ' ', text)
    text = re.sub(r'-{3,}', ' ', text)
    
    # Replace ligatures and special quotes with ASCII characters
    replacements = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff',
        '"': '"', '"': '"', ''': "'", ''': "'"
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    
    # Remove unwanted special characters except punctuation
    text = re.sub(r'[^\w\s\.,\'"\!\?\-\(\)\[\]]', '', text)
    
    # Normalize whitespace and tabs to single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize multiple newlines (more than 2) to just 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove isolated short numeric or special-character lines (common OCR issues)
    text = re.sub(r'^\W{1,3}\s*$', '', text, flags=re.MULTILINE)
    
    # Strip lines and remove empty lines
    lines = (line.strip() for line in text.split('\n'))
    non_empty_lines = [line for line in lines if line.strip()]
    
    return '\n'.join(non_empty_lines)

def make_recursive_splitter(chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """
    Create LangChain's token-aware recursive splitter
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        separators=SEPARATORS,
        keep_separator=True,
        strip_whitespace=False,
    )

def make_semantic_splitter() -> SemanticChunker:
    """
    Semantic chunking for the FINAL summary
    """
    return SemanticChunker(
        current_app.embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=83.0
    )

# LLM summarization (similar to process_with_metadaya.py)
SUMMARIZE_PROMPT = """
Summarize the provided text only, cleaning and condensing to make it clear and readable without changing meaning or adding interpretations. Keep the original tone and language (no translation). Retain all meaningful details: facts, figures, dates, durations, names, quotes, citations/section numbers, requirements/conditions/thresholds, examples, personas, brand/competitor names, key phrases/slogans. Remove only noise: contact info, headers/footers, page numbers, disclaimers, tracking IDs, random numeric strings, decorative characters, ads, navigation/UI, obvious OCR artifacts. Preserve the source order and hierarchy; keep headings/subheadings when present; use brief labels only if needed. Convert tables to compact key:value lists and preserve every row. Keep acronyms and terminology as-is; preserve numbers and units exactly. Condense repetition; if unsure, keep it. Fix broken hyphenations, spacing, and OCR issues. Output only the cleaned, faithful summary as plain text—no markdown, no preamble or postscript, no extra words or blank lines.
"""

def summarize_chunk_with_openai(chunk: str) -> str:
    """
    Summarize a single chunk using OpenAI
    """
    user_content = f"{SUMMARIZE_PROMPT}\n\n=== TEXT START ===\n{chunk}\n=== TEXT END ==="
    
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
        max_tokens=1200,
    )
    
    result = response.choices[0].message.content.strip()
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    return result

def summarize_document(text: str):
    """
    Two-stage summarization process:
    1) Split original text into chunks and summarize each
    2) Concatenate summaries into full summary
    """
    splitter = make_recursive_splitter(TOKENS_PER_CHUNK, TOKENS_OVERLAP)
    chunks = splitter.split_text(text)
    
    summaries = []
    for chunk in chunks:
        cleaned = summarize_chunk_with_openai(chunk)
        summaries.append(cleaned)
    
    concatenated = "\n\n".join(summaries).strip()
    concatenated = re.sub(r'[ \t]+', ' ', concatenated)
    concatenated = re.sub(r'\n{3,}', '\n\n', concatenated).strip()
    
    return summaries, concatenated

# Enhanced metadata generation (similar to process_with_metadaya.py)
def generate_metadata_rag(text, chunks):
    """
    Generate metadata from first 8 summarized chunks
    """
    sample_chunks = chunks[:8] if chunks else [text]
    joined_chunks = "\n\n".join(sample_chunks)
    
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
    
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=512,
    )
    
    response_content = response.choices[0].message.content.strip()
    
    # Remove markdown fences
    response_content = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        response_content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    
    # Helper to coerce/clean the structure
    def _coerce(md: dict) -> dict:
        topics = md.get("topics", [])
        keywords = md.get("keywords", [])
        if not isinstance(topics, list):
            topics = []
        if not isinstance(keywords, list):
            keywords = []
        topics = [str(t).strip() for t in topics if isinstance(t, (str, int, float)) and str(t).strip()]
        keywords = [str(k).strip() for k in keywords if isinstance(k, (str, int, float)) and str(k).strip()]
        return {"topics": topics[:5], "keywords": keywords[:10]}
    
    # Parse JSON with fallback
    metadata = {"topics": [], "keywords": []}
    try:
        metadata = _coerce(json.loads(response_content))
    except Exception:
        m = re.search(r"\{.*\}", response_content, flags=re.DOTALL)
        if m:
            try:
                metadata = _coerce(json.loads(m.group(0)))
            except Exception:
                pass
    
    return metadata

def authenticate(token):
    try:
        user = current_app.supabase.auth.get_user(token)
        return user.user is not None
    except Exception:
        return False

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
        # Step 1: Extract and preprocess text
        raw_text = file.read().decode('utf-8')
        processed_text = preprocess_text(raw_text)
        
        # Step 2: Two-stage summarization process
        per_chunk_summaries, full_summary = summarize_document(processed_text)
        
        # Step 3: Generate metadata from first 8 summarized chunks
        metadata = generate_metadata_rag(processed_text, per_chunk_summaries)
        
        # Step 4: Create semantic chunks from the full summary
        sem_splitter = make_semantic_splitter()
        semantic_summary_chunks = sem_splitter.split_text(full_summary)
        
        # Step 5: Embed semantic chunks into Pinecone
        index = current_app.pinecone.Index(PINECONE_INDEX)
        
        for idx, chunk in enumerate(semantic_summary_chunks):
            embedding_vector = current_app.embeddings_model.embed_documents([chunk])[0]
            
            record = {
                'id': f"{file.filename}_semantic_chunk_{idx}_{str(uuid.uuid4())[:8]}",
                'values': embedding_vector,
                'metadata': {
                    'filename': file.filename,
                    'topics': metadata.get('topics', []),
                    'keywords': metadata.get('keywords', []),
                    'chunk_index': idx,
                    'content': chunk,
                    'total_chunks': len(semantic_summary_chunks)
                }
            }
            
            index.upsert(vectors=[record], namespace=namespace)

        return jsonify({
            "message": f"File '{file.filename}' successfully uploaded and processed.",
            "details": {
                "original_chunks": len(per_chunk_summaries),
                "semantic_chunks": len(semantic_summary_chunks),
                "topics": metadata.get('topics', []),
                "keywords": metadata.get('keywords', [])
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
