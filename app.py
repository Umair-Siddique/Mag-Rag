from flask import Flask
from config import Config
from dotenv import load_dotenv
from extensions import init_supabase, init_pinecone, init_anthropic, init_openai_embeddings,init_mistral
from blueprints.auth import auth_bp
from blueprints.chat import chat_bp 
from flask_cors import CORS
from blueprints.sys_prompt import sys_prompt_bp
from blueprints.upload_document import upload_document_bp
from blueprints.retriever import retriever_bp

def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config.from_object(Config)
    
    CORS(app, supports_credentials=True, origins=['http://localhost:5173', 'https://blue-line-beta.vercel.app/'])

    # Initialize extensions
    init_supabase(app)
    init_pinecone(app)
    init_anthropic(app)
    init_openai_embeddings(app)
    init_mistral(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(sys_prompt_bp, url_prefix="/prompt")
    app.register_blueprint(upload_document_bp,url_prefix="/document")
    app.register_blueprint(retriever_bp,url_prefix="/retriever")

    return app

# Create the app instance for gunicorn
app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)