from supabase import create_client
from pinecone import Pinecone
import anthropic
from langchain_openai.embeddings import OpenAIEmbeddings
from config import Config
from mistralai import Mistral

def init_supabase(app):
    app.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

def init_pinecone(app):
    app.pinecone = Pinecone(api_key=Config.PINECONE_API_KEY)

def init_anthropic(app):
    app.anthropic = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)

def init_mistral(app):
    app.mistral = Mistral(api_key=Config.MISTRAL_API_KEY)

def init_openai_embeddings(app):
    app.embeddings_model = OpenAIEmbeddings(
        api_key=Config.OPEN_AI_KEY,
        model=Config.EMBEDDING_MODEL
    )
