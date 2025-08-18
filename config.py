import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    OPEN_AI_KEY=os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY=os.getenv("CLAUDE_API_KEY")
    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
    MISTRAL_API_KEY=os.getenv('MISTRAL_API_KEY')