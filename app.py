from flask import Flask
from config import Config
from dotenv import load_dotenv
from extensions import init_supabase, init_pinecone, init_anthropic, init_openai_embeddings
from blueprints.auth import auth_bp
from blueprints.chat import chat_bp 
from flask_cors import CORS
from blueprints.sys_prompt import sys_prompt_bp
from blueprints.upload_document import upload_document_bp

def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config.from_object(Config)
    
    CORS(app, supports_credentials=True, origins=['http://localhost:5173'])

    # Initialize extensions
    init_supabase(app)
    init_pinecone(app)
    init_anthropic(app)
    init_openai_embeddings(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(sys_prompt_bp, url_prefix="/prompt")
    app.register_blueprint(upload_document_bp,url_prefix="/document")

    return app