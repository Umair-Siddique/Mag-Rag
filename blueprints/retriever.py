import os
from typing import TypedDict, List, Literal
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

load_dotenv()

app = Flask(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
embeddings_model = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model='text-embedding-3-large')
index = pc.Index("career-counseling-documents")

class GraphState(TypedDict):
    query: str
    method: Literal["web", "pinecone", "chat_history"]
    namespace: str
    retrieved_documents: List[str]
    web_search_results: str
    final_answer: str

# Decide retrieval method based on query
def decide_method(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""
    Decide retrieval method based on query:
    - Use "web" for recent external trends or events.
    - Use "pinecone" for brand or consumer insights.
    - Use "chat_history" for conversational or follow-up queries.

    Query: {query}
    Respond ONLY with "web", "pinecone", or "chat_history".
    """
    completion = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    method = completion.choices[0].message.content.strip().lower()
    return {**state, "method": method}

# Decide namespace for Pinecone
def decide_namespace(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""
    Choose namespace:
    - "brand-positioning" for queries about branding, metaphors, archetypes.
    - "insights" for consumer behavior, trends, or cultural insights.

    Query: {query}
    Respond ONLY with "brand-positioning" or "insights".
    """
    completion = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    namespace = completion.choices[0].message.content.strip().lower()
    return {**state, "namespace": namespace}

# Pinecone retrieval
def retrieve_pinecone(state: GraphState) -> GraphState:
    print(f"Retrieving from Pinecone namespace: {state['namespace']}")
    query_embedding = embeddings_model.embed_query(state["query"])
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=state["namespace"])
    docs = [match['metadata'].get('content', '') for match in results.get('matches', [])]
    return {**state, "retrieved_documents": docs, "web_search_results": ""}


def retrieve_web(state: GraphState) -> GraphState:
    print("Retrieving from Web using Tavily")
    search = TavilySearch(max_results=3)
    results = search.invoke(state["query"])
    docs = [r["content"] for r in results.get("results", [])]
    print(f"Web returned {len(docs)} hits")
    formatted = "\n\n".join(docs)
    return {
        **state,
        "retrieved_documents": docs,
        "web_search_results": formatted
    }


# Placeholder for chat history retrieval
def retrieve_chat_history(state: GraphState) -> GraphState:
    print("Retrieving from Chat History")
    chat_history = "Previous conversation context placeholder."
    return {**state, "retrieved_documents": [chat_history], "web_search_results": ""}

# Route decision after method
def route_method(state: GraphState):
    if state["method"] == "pinecone":
        return "decide_namespace"
    if state["method"] == "web":
        return "web_retrieve"
    return "chat_history_retrieve"

def generate_answer(state: GraphState) -> GraphState:
    # build a prompt that includes the retrieved docs
    docs_blob = "\n\n---\n\n".join(state["retrieved_documents"])
    prompt = f"""
You are an expert assistant.  A user asked:
{state['query']}

Here are the relevant documents we found:
{docs_blob}

Please synthesize an answer to the user’s question, grounding your answer in the above documents. 
If anything is uncertain, say “I’m not sure” rather than hallucinating.
"""
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = completion.choices[0].message.content.strip()
    return {**state, "final_answer": answer}


# Build graph
builder = StateGraph(GraphState)
builder.add_node("decide_method", decide_method)
builder.add_node("decide_namespace", decide_namespace)
builder.add_node("pinecone_retrieve", retrieve_pinecone)
builder.add_node("web_retrieve", retrieve_web)
builder.add_node("chat_history_retrieve", retrieve_chat_history)
builder.set_entry_point("decide_method")

builder.add_conditional_edges("decide_method", route_method)
builder.add_edge("decide_namespace", "pinecone_retrieve")
builder.add_node("generate_answer", generate_answer)

builder.add_edge("pinecone_retrieve", "generate_answer")
builder.add_edge("web_retrieve",      "generate_answer")
builder.add_edge("chat_history_retrieve", "generate_answer")
builder.add_edge("generate_answer", END)


graph = builder.compile()

@app.route('/query', methods=['POST'])
def query_api():
    data = request.json
    query = data.get("query")

    initial_state = {"query": query}
    final_state   = graph.invoke(initial_state)

    response = {
        "method": final_state["method"],
        "namespace": final_state.get("namespace"),
        "retrieved_documents": final_state.get("retrieved_documents", []),
        "web_search_results":    final_state.get("web_search_results", ""),
        "final_answer":          final_state.get("final_answer", "")
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)