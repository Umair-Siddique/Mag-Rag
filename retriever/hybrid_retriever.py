import os
from typing import List
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone and LLM
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4-turbo")

index_name = "your-pinecone-index-name"
pinecone_index = pc.Index(index_name)

# Function to dynamically extract metadata (topics or keywords) from user query using LLM
def extract_metadata_from_query(query: str) -> List[str]:
    prompt_template = ChatPromptTemplate.from_template(
        """Given the following user query, identify relevant metadata topics or keywords from these lists:
        Topics: ["Mediterranean Lifestyle", "Wellness Culture", "Community Driven", "Blue Zone Approach", "Authentic Wellness"]
        Keywords: ["Forum", "Wellness", "Mediterranean", "Community", "Blue Zone", "Authenticity", "Inclusivity", "Relaxation", "Social Connection", "Longevity"]

        User query: '{query}'

        Return only a Python list of relevant metadata."""
    )

    response = llm.invoke(prompt_template.format(query=query))
    metadata = eval(response.content.strip())
    return metadata

# Function to retrieve documents based on metadata
def retrieve_docs_by_metadata(query: str, top_k: int = 5):
    # Extract relevant metadata dynamically
    metadata = extract_metadata_from_query(query)

    # Define Pinecone metadata filter
    metadata_filter = {
        "$or": [
            {"topics": {"$in": metadata}},
            {"keywords": {"$in": metadata}}
        ]
    }

    # Perform metadata-only search in Pinecone
    results = pinecone_index.query(
        namespace="",
        vector=[],  # No vector provided as we are only using metadata search
        top_k=top_k,
        filter=metadata_filter,
        include_metadata=True
    )

    return results

# Example usage
if __name__ == "__main__":
    user_query = "I'm interested in relaxation and community wellness."

    retrieved_docs = retrieve_docs_by_metadata(user_query)

    for idx, match in enumerate(retrieved_docs.matches):
        print(f"Document {idx + 1}:\n{match.metadata}\n---")
