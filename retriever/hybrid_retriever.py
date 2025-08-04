import os
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
import json

load_dotenv()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv('OPENAI_API_KEY'),
    model='text-embedding-3-large'
)

# Define Pinecone index name
index_name = "career-counseling-documents"

# Initialize Pinecone Vector Store directly with index name
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model,
    text_key="content"  # Specify the correct metadata key here
)


# Dense vector retrieval
def dense_search(query: str, namespace: str, top_k: int = 8):
    query_embedding = embeddings_model.embed_query(query)

    results = vectorstore._index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    documents = [match['metadata'] for match in results.get('matches', [])]
    return documents

# Self-query metadata retrieval without explicit limit
def self_query_retriever(query: str, namespace: str, llm_for_retriever):
    metadata_field_info = [
        AttributeInfo(name="topics", description="Topics or categories the document covers", type="string"),
        AttributeInfo(name="filename", description="The name of the file containing the document", type="string")
    ]

    document_content_description = "Documents containing insights, analyses, and information for career counseling."

    retriever = SelfQueryRetriever.from_llm(
        llm=llm_for_retriever,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=PineconeTranslator(),
        search_kwargs={"namespace": namespace},
        verbose=True
    )

    print("Retriever initialized successfully.")
    docs = retriever.get_relevant_documents(query)
    return docs

# Namespace selection via LLM
def decide_namespace(query: str):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    prompt = f"""
    Decide the appropriate namespace for retrieving external data based on the user's query.

    Available Namespaces:
    - "brand-positioning": queries about brand identity, cultural metaphors, archetypes, and brand positioning.
    - "insights": queries about consumer behavior, societal trends, wellness, community, or cultural insights.

    User query: "{query}"

    Respond with ONLY one word: "insights" or "insights".
    """

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    namespace = completion.choices[0].message.content.strip().lower()
    return namespace

# Example usage
def get_combined_context(query, namespace, llm_for_retriever):
    dense_docs = dense_search(query, namespace)
    metadata_docs = self_query_retriever(query, namespace, llm_for_retriever)

    combined_context = "\n\n".join(
        [doc.get("content", "") for doc in dense_docs] +
        [doc.page_content for doc in metadata_docs]
    )
    
    return combined_context

# Stream response from Claude
SYSTEM_PROMPT = "You are an expert assistant providing detailed, precise, and professional answers."

def stream_llm_response(query, context):
    user_prompt = f"""CONTEXT:\n{context}\n\nUSER QUERY:\n{query}\n\nINSTRUCTIONS TO YOU:\n- Use only the CONTEXT to answer unless the query explicitly asks for outside knowledge.\n- If multiple parts of the CONTEXT conflict, note the conflict and explain the safest/most conservative interpretation.\n- If the answer requires assumptions beyond what's in CONTEXT, label them clearly as assumptions.\n- Provide the answer and, if relevant, a brief list of which parts of CONTEXT you used\n"""

    api_url = 'https://api.anthropic.com/v1/messages'
    api_key = os.getenv('CLAUDE_API_KEY')

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": True
    }

    response = requests.post(api_url, headers=headers, json=data, stream=True)

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line.startswith('data:'):
                json_data = decoded_line[5:].strip()
                if json_data:
                    event_data = json.loads(json_data)
                    if event_data['type'] == 'content_block_delta':
                        text_delta = event_data['delta']['text']
                        print(text_delta, end='', flush=True)
    print()

if __name__ == "__main__":
    query = input("Enter your query: ")
    llm_for_retriever = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4")
    namespace = decide_namespace(query)
    context = get_combined_context(query, namespace, llm_for_retriever)
    stream_llm_response(query, context)
