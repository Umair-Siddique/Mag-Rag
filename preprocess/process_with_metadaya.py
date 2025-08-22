import os
import re
import json
from typing import List, Tuple

import numpy as np
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

# --- Config ---
TOKENS_PER_CHUNK = 2500
TOKENS_OVERLAP   = 250  # small overlap to avoid boundary loss
MODEL_NAME       = "meta-llama/llama-4-scout-17b-16e-instruct"

INPUT_FOLDER  = r"extracted_data\extracted_text_operations"
OUTPUT_FOLDER = "final_operations"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GROQ_API_KEY=os.getenv('GROQ_API_KEY')
OPEN_AI_API=os.getenv('OPENAI_API_KEY')

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Simple recursive splitter (LangChain only; no fallback) ---
SEPARATORS = [
    "\n\n", "\r\n\r\n",       # paragraphs
    "\n", "\r\n",             # lines
    ". ", "! ", "? ",         # sentence ends (with trailing space)
    ".", "!", "?",            # sentence ends (no space)
]

def make_recursive_splitter(chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """
    Always use LangChain's token-aware recursive splitter.
    NOTE: Requires tiktoken under the hood; no fallback paths.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        separators=SEPARATORS,
        keep_separator=True,       # keep separators so nothing is dropped
        strip_whitespace=False,    # do not trim; preserve exact content
    )

def make_semantic_splitter() -> SemanticChunker:
    """
    Semantic chunking for the FINAL summary.
    Adjust thresholds as needed.
    """
    return SemanticChunker(
        OpenAIEmbeddings(api_key=OPEN_AI_API),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=83.0
    )

# ---------- LLM summarization ----------
SUMMARIZE_PROMPT = """
Summarize the provided text only, cleaning and condensing to make it clear and readable without changing meaning or adding interpretations. Keep the original tone and language (no translation). Retain all meaningful details: facts, figures, dates, durations, names, quotes, citations/section numbers, requirements/conditions/thresholds, examples, personas, brand/competitor names, key phrases/slogans. Remove only noise: contact info, headers/footers, page numbers, disclaimers, tracking IDs, random numeric strings, decorative characters, ads, navigation/UI, obvious OCR artifacts. Preserve the source order and hierarchy; keep headings/subheadings when present; use brief labels only if needed. Convert tables to compact key:value lists and preserve every row. Keep acronyms and terminology as-is; preserve numbers and units exactly. Condense repetition; if unsure, keep it. Fix broken hyphenations, spacing, and OCR issues. Output only the cleaned, faithful summary as plain text—no markdown, no preamble or postscript, no extra words or blank lines.
"""

def generate_metadata_rag(text, faiss_index, embeddings_model, groq_client, chunks):
    """
    Produce short topics and keywords from a sample of the document's chunks.
    Uses the first ~8 per-chunk summaries.
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

    completion = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=512,
        stream=False
    )

    response_content = completion.choices[0].message.content.strip()

    # Remove markdown fences (``` or ```json ... ```)
    response_content = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        response_content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Helper to coerce/clean the structure and trim lengths
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

    # Parse JSON strictly, with a fallback that extracts the first {...} block
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


def summarize_chunk_with_groq(chunk: str) -> str:
    user_content = f"{SUMMARIZE_PROMPT}\n\n=== TEXT START ===\n{chunk}\n=== TEXT END ==="
    completion = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
        max_completion_tokens=1200,
        stream=False,
    )
    result = completion.choices[0].message.content.strip()
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    return result

# ---------- Processing pipeline ----------
def summarize_document(text: str) -> Tuple[List[str], str]:
    """
    1) Split ORIGINAL text with LangChain RecursiveCharacterTextSplitter (~2500 tokens, 250 overlap).
    2) Summarize each original chunk ONCE -> per_chunk_summaries.
    3) Concatenate per_chunk_summaries into a single full_summary (no further summarization).
    """
    splitter = make_recursive_splitter(TOKENS_PER_CHUNK, TOKENS_OVERLAP)
    chunks = splitter.split_text(text)

    summaries: List[str] = []
    for ch in chunks:
        cleaned = summarize_chunk_with_groq(ch)
        summaries.append(cleaned)

    concatenated = "\n\n".join(summaries).strip()
    concatenated = re.sub(r'[ \t]+', ' ', concatenated)
    concatenated = re.sub(r'\n{3,}', '\n\n', concatenated).strip()

    return summaries, concatenated




# ---------- Main loop ----------
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(".txt"):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    with open(input_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    per_chunk_summaries, full_summary = summarize_document(original_text)

    # Save ONE final file with the concatenated summary
    base = os.path.splitext(filename)[0]
    final_txt_path = os.path.join(OUTPUT_FOLDER, f"{base}__summary.txt")
    with open(final_txt_path, "w", encoding="utf-8") as out:
        out.write(full_summary)

    # --- NEW: Semantic chunking for the FINAL SUMMARY (store these in JSON) ---
    sem_splitter = make_semantic_splitter()
    semantic_summary_chunks = sem_splitter.split_text(full_summary)

    # Also store a human-readable chunk list (optional, unchanged behavior)
    chunk_list_path = os.path.join(OUTPUT_FOLDER, f"{base}__summary_chunklist.txt")
    with open(chunk_list_path, "w", encoding="utf-8") as cf:
        total = len(semantic_summary_chunks)
        for i, ch in enumerate(semantic_summary_chunks, start=1):
            cf.write(f"=== SEMANTIC SUMMARY CHUNK {i}/{total} ===\n{ch}\n\n")

    # --- Single JSON output (filename, topics, keywords, summarized_chunks = semantic chunks) ---
    meta = generate_metadata_rag(
        original_text,          # original (fallback inside the function)
        None,                   # faiss_index (unused)
        None,                   # embeddings_model (unused)
        groq_client,            # Groq client
        per_chunk_summaries     # first ~8 summarized chunks used inside
    )

    unified_json = {
        "filename": filename,
        "keywords": meta.get("keywords", []),
        "topics": meta.get("topics", []),
        # IMPORTANT: these are SEMANTIC chunks of the FINAL SUMMARY
        "summarized_chunks": semantic_summary_chunks
    }

    unified_path = os.path.join(OUTPUT_FOLDER, f"{base}__meta.json")
    with open(unified_path, "w", encoding="utf-8") as jf:
        json.dump(unified_json, jf, ensure_ascii=False, indent=2)

    print(f"✅ Summarized & cleaned: '{filename}' → {final_txt_path}")
    print(f"🧠 Semantic chunk list saved: '{filename}' → {chunk_list_path}")
    print(f"🗂️ JSON saved: '{filename}' → {unified_path}")
