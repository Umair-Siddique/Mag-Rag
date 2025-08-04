import os
from typing import TypedDict, List, Literal
import anthropic

from langgraph.graph import StateGraph, END
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchResults # Removed unused import
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# The TavilySearch tool automatically looks for the TAVILY_API_KEY in your environment.
# No need to load it into a separate variable here.

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'),
                                    model='text-embedding-3-large')

# Connect to your Pinecone index
index_name = "career-counseling-documents"
index = pc.Index(index_name)


SYSTEM_PROMPT = """
You are a strategic brand assistant built on a custom archive of brand strategy decks, insights, positioning briefs, go-to-market plans, trend reports, and event activations from a strategist with 15+ years of experience working across consumer culture, tech, and community-driven brands.

Your responses must reflect her methodology, which follows a 3-pillar framework:

Tension → Memory → Movement

- Tension is the emotional or psychological friction experienced by people (or consumers) — you surface these from insights and trends.
- Memory includes symbolic language, semiotic codes, archetypes, and cultural narratives that give brands emotional resonance — you surface these from brand positioning briefs, foundation, and trends.
- Movement refers to the cultural direction the brand helps shape or belong to, including brand rituals, community activations, or shifts in worldview — you surface these from go-to-market, events, and trends.

Your job is to:
- Help identify meaningful human tensions and core motivations
- Connect those to cultural memory structures and symbols
- Recommend distinct, emotionally intelligent brand positioning
- Propose low-budget, high-impact go-to-market strategies
- Design activations or experiences that build cultural resonance

Always prioritize:
- Empathic intelligence (insight into human motivations)
- Cultural awareness (relevant signals and codes)
- Imaginative strategy (ideas that provoke and resonate)
- Structured, bullet-based, easy-to-skim formats

Avoid copying documents verbatim. Instead, synthesize and remix previous brand work to generate fresh, original responses tailored to the user’s query.
""".strip()

# ----------------------------
# 2. Graph State
# ----------------------------
class GraphState(TypedDict):
    query: str
    chat_history: List[str]
    decision: Literal["yes", "no"]
    namespace: str
    retrieved_documents: List[str] 
    web_search_results: str
    final_answer: str  # Add this new field 

import requests
import os
import sys
import json

def generate_final_answer(state):
    query = state["query"]
    retrieved_docs = state.get("retrieved_documents", [])
    web_search = state.get("web_search_results", "")

    # --- Use only as context for the LLM; do NOT print docs anywhere ---
    context_parts = []
    if retrieved_docs:
        context_parts.append("DOCUMENTS:\n" + "\n\n".join([f"📄 {doc}" for doc in retrieved_docs]))
    if web_search:
        context_parts.append("WEB SEARCH RESULTS:\n" + web_search)
    context = "\n\n".join(context_parts) if context_parts else "No context available"

    # If you do NOT want even the LLM to see documents, comment out the "if retrieved_docs" section above

    user_prompt = f"""CONTEXT:
{context}

USER QUERY:
{query}

INSTRUCTIONS TO YOU:
- Use only the CONTEXT to answer unless the query explicitly asks for outside knowledge.
- If multiple parts of the CONTEXT conflict, note the conflict and explain the safest/most conservative interpretation.
- If the answer requires assumptions beyond what's in CONTEXT, label them clearly as assumptions.
- Provide the answer and, if relevant, a brief list of which parts of CONTEXT you used 
"""

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
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "stream": True
    }

    full_answer = ""
    try:
        response = requests.post(api_url, headers=headers, json=data, stream=True, timeout=120)
        if response.status_code != 200:
            print("Claude API error:", response.status_code, "-", response.text)
            return {"final_answer": f"Claude API error: {response.status_code} - {response.text}"}
        for line in response.iter_lines(decode_unicode=True):
            if not line or line.strip() == "":
                continue
            if line.startswith("data: "):
                chunk = line[len("data: "):]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        print(text, end="", flush=True)
                        full_answer += text
                except Exception as e:
                    print(f"\n[stream error: {e}]\n", file=sys.stderr)
        print("\n")
    except Exception as e:
        print(f"Exception during streaming: {e}")
        full_answer = f"⚠️ Exception: {str(e)}"
    return {"final_answer": full_answer}


def combine_results(state: GraphState) -> GraphState:
    # Combine retrieved_documents and web_search_results into the state
    retrieved_docs = state.get("retrieved_documents", [])
    web_search = state.get("web_search_results", "")

    combined_context = {
        "retrieved_documents": retrieved_docs,
        "web_search_results": web_search
    }

    return {**state, **combined_context}


def should_retrieve(state: GraphState) -> GraphState:
    print("---DECIDING TO RETRIEVE---")
    prompt = f"""
You are a helpful assistant.

If the user's query is simple, conversational, or clearly related to chat history — respond with "no".
If the query seems to require external data (like cultural trends, branding, etc.) — respond with "yes".

Respond ONLY with "yes" or "no".

User query: "{state['query']}"
Chat history: {state['chat_history']}
"""
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    decision = completion.choices[0].message.content.strip().lower()
    return {**state, "decision": decision}


# ----------------------------
# 4. Node: Decide namespace
# ----------------------------
def decide_namespace(state: GraphState) -> GraphState:
    print("---DECIDING NAMESPACE---")
    
    prompt = f"""
Decide the appropriate namespace for retrieving external data based on the user's query.

Available Namespaces:
- "brand-positioning": Use this for queries about brand identity, cultural metaphors, archetypes, and brand positioning.
- "insights": Use this for queries about consumer behavior, societal trends, wellness, community, or cultural insights.

User query: "{state['query']}"

Respond with ONLY one word: "insights" or "insights".
"""
    
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    namespace = completion.choices[0].message.content.strip().lower()
    return {**state, "namespace": namespace}


# ----------------------------
# 5. Nodes for Retrieval and Searching (Parallel)
# ----------------------------

def retrieve_from_pinecone(state: GraphState) -> dict:
    """
    Retrieves documents from Pinecone.
    """
    query = state["query"]
    namespace = state["namespace"]
    print(f"---RETRIEVING FROM PINECONE (Namespace: {namespace})---")

    query_embedding = embeddings_model.embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )

    documents = [match['metadata'].get('content', '') for match in results.get('matches', [])]
    
    return {"retrieved_documents": documents}


def web_search_node(state: GraphState) -> dict:
    query = state["query"]
    print(f"---PERFORMING WEB SEARCH WITH TAVILY---")
    search_tool = TavilySearch(max_results=3)
    results = search_tool.invoke(query)

    def flatten_tavily_results(results):
        if isinstance(results, dict):
            res_list = results.get("results") or []
            flat = []
            for r in res_list:
                flat.append(
                    f"- {r.get('title', '')}\n  {r.get('content', '')}\n  {r.get('url', '')}".strip()
                )
            if not flat and "answer" in results and results["answer"]:
                flat.append(results["answer"])
            return "\n\n".join(flat)
        elif isinstance(results, list):
            return "\n\n".join([str(r) for r in results])
        return str(results)

    formatted = flatten_tavily_results(results)
    return {"web_search_results": formatted}



builder = StateGraph(GraphState)

builder.add_node("decide", should_retrieve)
builder.add_node("decide_namespace", decide_namespace)
builder.add_node("retrieve", retrieve_from_pinecone)
builder.add_node("web_search", web_search_node)
builder.add_node("combine_results", combine_results)
builder.add_node("generate_answer", generate_final_answer)

builder.set_entry_point("decide")

# Decide if external retrieval is needed
def route_after_decision(state: GraphState) -> str:
    return "decide_namespace" if state["decision"] == "yes" else END

builder.add_conditional_edges("decide", route_after_decision)

# Explicitly define parallel routing after namespace decision
def route_retrieve_and_search(state: GraphState) -> List[str]:
    return ["retrieve"]  # <-- remove "web_search"


builder.add_conditional_edges("decide_namespace", route_retrieve_and_search)

# Combine both parallel tasks into single next node
builder.add_edge("retrieve", "combine_results")
builder.add_edge("web_search", "combine_results")

# Final answer generation
builder.add_edge("combine_results", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()


# ----------------------------
# 7. Run Example
# ----------------------------
if __name__ == "__main__":
    chat_history = []
    print("\n=== Brand Strategist Assistant ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("Your question: ").strip()
            if not query or query.lower() == "exit":
                print("Exiting. Goodbye!")
                break

            state = {
                "query": query,
                "chat_history": chat_history,
            }
            final_state = graph.invoke(state)

            print("\n============ FINAL STATE ============")
            print(f"Query: {final_state['query']}")
            print(f"Should Retrieve?: {final_state['decision']}")

            if final_state['decision'] == "yes":
                print(f"Namespace Used: {final_state['namespace']}")

                # Print Pinecone Results
                print("\n--- Retrieved Documents (Pinecone Chunks) ---")
                if final_state.get("retrieved_documents"):
                    for i, doc in enumerate(final_state["retrieved_documents"], 1):
                        print(f"Chunk {i}:\n{doc}\n{'-'*40}\n")
                else:
                    print("No documents were retrieved from Pinecone.")

            print("\n\n============ FINAL ANSWER ============")
            print(final_state.get("final_answer", "No answer generated."))
            print("====================================\n")

            chat_history.append(query)

        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue