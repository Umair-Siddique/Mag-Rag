import os
from typing import TypedDict, List, Literal
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from uuid import uuid4
from datetime import datetime, timezone
from postgrest.exceptions import APIError
import json
from flask import Response, stream_with_context

# Initialize blueprint
retriever_bp = Blueprint('retriever', __name__)

# Initialize clients
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embeddings_model = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model='text-embedding-3-large')

# Initialize LangChain components
llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-4"
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
)

class GraphState(TypedDict):
    query: str
    method: Literal["web", "pinecone", "chat_history"]
    namespace: str
    retrieved_documents: List[str]
    web_search_results: str
    final_answer: str
    conversation_id: str
    user_id: str
    chat_history: List[str]

# Helper function for Server-Sent Events
def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def get_user_system_prompt(user_id: str) -> str:
    """
    Get the user-specific system prompt from Supabase, or fall back to global default.
    """
    try:
        supabase = current_app.supabase

        # First, check if the user has a custom system prompt
        prompt_resp = (
            supabase.table('user_system_prompts')
            .select('system_prompt')
            .eq('user_id', user_id)
            .execute()
        )

        if prompt_resp.data:
            return prompt_resp.data[0]['system_prompt']

        # If no custom prompt, get the global default and cache it for the user
        global_resp = (
            supabase.table('global_system_prompt')
            .select('config_value')
            .eq('config_key', 'system_prompt')
            .execute()
        )

        if global_resp.data:
            default_prompt = global_resp.data[0]['config_value']
            # Cache the default prompt for this user
            supabase.table('user_system_prompts').insert(
                {'user_id': user_id, 'system_prompt': default_prompt}
            ).execute()
            return default_prompt

        # Fallback to hardcoded prompt if database is not configured
        return """
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

Avoid copying documents verbatim. Instead, synthesize and remix previous brand work to generate fresh, original responses tailored to the users query.
        """

    except Exception as e:
        current_app.logger.error(f"Error fetching system prompt for user {user_id}: {e}")
        return """
You are a strategic brand assistant. Please provide helpful, insightful responses based on the available context and user queries.
        """

def get_chat_history_context(conversation_id: str, user_id: str) -> List[str]:
    """
    Use LangChain's memory management to retrieve and format chat history
    """
    if not conversation_id:
        return []
    
    try:
        supabase = current_app.supabase
        
        # Get the last 10 messages from Supabase
        res = (
            supabase
            .table('messages')
            .select('content, sender, created_at')
            .eq('conversation_id', conversation_id)
            .order('created_at', desc=False)
            .limit(10)
            .execute()
        )

        rows = res.data or []
        
        # Convert to LangChain message format
        messages = []
        for row in rows:
            if row['sender'] == 'user':
                messages.append(HumanMessage(content=row['content']))
            elif row['sender'] == 'assistant':
                messages.append(AIMessage(content=row['content']))
        
        # Update memory with conversation history
        memory.chat_memory.messages = messages
        
        # Get formatted history from LangChain memory
        history = memory.load_memory_variables({})
        chat_history = history.get('chat_history', [])
        
        # Format for display
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Assistant: {msg.content}")
        
        return formatted_history[-10:]  # Return last 10 messages
        
    except Exception as e:
        current_app.logger.error(f"Error fetching chat history context: {e}")
        return []

# Authentication helper functions
def _extract_access_token() -> str | None:
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header.split(' ', 1)[1].strip() or None
    return None

def authenticate_user() -> object | None:
    access_token = _extract_access_token()
    if not access_token:
        return None

    supabase = current_app.supabase
    try:
        return supabase.auth.get_user(access_token).user
    except Exception:
        return None

def verify_conversation_ownership(conversation_id: str, user_id: str) -> bool:
    try:
        supabase = current_app.supabase
        res = (
            supabase.table('conversations')
            .select('user_id')
            .eq('id', conversation_id)
            .single()
            .execute()
        )
        return res.data and res.data.get('user_id') == user_id
    except Exception:
        return False

# Graph functions
def decide_method(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""
    Decide retrieval method based on query:
    - Use "web" Only when user specifically mentioned to retrieve from web.
    - Use "pinecone" for Every general query where User ask about some topic or a thing and It did not mentioned to search from web.

    Query: {query}
    Respond ONLY with "web" or "pinecone".
    """
    completion = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    method = completion.choices[0].message.content.strip().lower()
    return {**state, "method": method}

def decide_namespace(state: GraphState) -> GraphState:
    query = state["query"]

    prompt = """
    Based on the query, choose the most appropriate namespace from these 4 options:

    1. Memory: For queries about memory structures, iconic elements, functional elements, emotional elements, brand architecture, branding, brand strategy, positioning, metaphors, archetypes, semiotics, brand foundations, brand briefs, cultural positioning, brand narratives, manifestos, values, and brand voice.

    2. Tension: For queries about consumer behavior, market trends, empathic insights, seven centers, cultural insights, cultural strategy, cultural currency, tensions, consumer research, red door, market analysis, behavioral patterns, cultural shifts, and trend analysis, cultural trends.

    3. Movement: For queries about events, brand activations, launch events, community building, cultural programs, IRL experiences, event design, experiential marketing, go-to-market plans, content strategy, experience strategy, lifecycle marketing, campaigns, social media, influencers or brand ambassadors, communications plans, connections strategy, media planning, paid media, owned media, earned media.

    4. common: For queries about operations, team building, frameworks, templates, brand strategy, methodologies, workshops, general business operations.

    Query: {query}

    Respond ONLY with one of these exact namespaces: Memory, Tension, Movement, or common (no quotes, no punctuation, no extra words).
    """.format(query=query)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    namespace = completion.choices[0].message.content.strip()
    namespace = namespace.strip('"').strip("'").lower()

    # Map client-facing namespaces back to Pinecone ones
    namespace_map = {
        "memory": "brand-positioning",
        "tension": "insights",
        "movement": "events",
        "common": "common"
    }

    namespace = namespace_map.get(namespace, "common")

    return {**state, "namespace": namespace}


def retrieve_pinecone(state: GraphState) -> GraphState:
    print(f"Retrieving from Pinecone namespace: {state['namespace']}")
    query_embedding = embeddings_model.embed_query(state["query"])
    
    # Use the Pinecone instance from extensions.py
    index = current_app.pinecone.Index("career-counseling-documents")
    
    # Get all available namespaces
    all_namespaces = ["brand-positioning", "insights", "events", "common"]
    other_namespaces = [ns for ns in all_namespaces if ns != state["namespace"]]
    
    # Search from specific namespace
    specific_results = index.query(
        vector=query_embedding, 
        top_k=5, 
        include_metadata=True, 
        namespace=state["namespace"]
    )
    
    # Search from all other namespaces combined
    other_results = index.query(
        vector=query_embedding, 
        top_k=5, 
        include_metadata=True, 
        namespace=",".join(other_namespaces)
    )
    
    # Combine and deduplicate results
    all_matches = []
    
    if specific_results.get('matches'):
        all_matches.extend(specific_results['matches'])
    
    if other_results.get('matches'):
        all_matches.extend(other_results['matches'])
    
    # Remove duplicates based on metadata text
    seen_texts = set()
    unique_matches = []
    for match in all_matches:
        text = match['metadata'].get('text', '')
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_matches.append(match)
    
    docs = [match['metadata'].get('text', '') for match in unique_matches[:5]]
    
    return {**state, "retrieved_documents": docs, "web_search_results": ""}

def retrieve_web(state: GraphState) -> GraphState:
    print("Retrieving from Web using Tavily")
    search = TavilySearch(max_results=5)
    results = search.invoke(state["query"])
    docs = [r["content"] for r in results.get("results", [])]
    print(f"Web returned {len(docs)} hits")
    formatted = "\n\n".join(docs)
    return {
        **state,
        "retrieved_documents": docs,
        "web_search_results": formatted
    }

def retrieve_chat_history(state: GraphState) -> GraphState:
    print("Retrieving from Chat History")
    conv_id = state.get("conversation_id")

    if not conv_id:
        return {
            **state,
            "retrieved_documents": ["No conversation_id provided; cannot load chat history."],
            "web_search_results": ""
        }

    try:
        # Use LangChain memory to get chat history
        chat_history = get_chat_history_context(conv_id, state.get("user_id"))
        
        if not chat_history:
            return {
                **state,
                "retrieved_documents": ["No chat history found."],
                "web_search_results": ""
            }

        return {
            **state,
            "retrieved_documents": chat_history,
            "web_search_results": ""
        }

    except Exception as e:
        return {
            **state,
            "retrieved_documents": [f"Error fetching chat history: {e}"],
            "web_search_results": ""
        }

def route_method(state: GraphState):
    if state["method"] == "pinecone":
        return "decide_namespace"
    if state["method"] == "web":
        return "web_retrieve"
    return "chat_history_retrieve"

def generate_answer(state: GraphState) -> GraphState:
    # Always get chat history context first using LangChain memory
    chat_history_context = get_chat_history_context(state.get("conversation_id"), state.get("user_id"))
    
    # Combine chat history with retrieved documents
    all_context = []
    
    # Add chat history context first (most recent conversations)
    if chat_history_context:
        all_context.extend(chat_history_context)
    
    # Add retrieved documents from the main method
    if state.get("retrieved_documents"):
        all_context.extend(state["retrieved_documents"])
    
    # If no main documents retrieved, still use chat history
    if not all_context:
        all_context = chat_history_context
    
    docs_blob = f"User (current query): {state['query']}\n\n---\n\n" + "\n\n---\n\n".join(all_context[:8])

    # Get the user ID from the conversation to fetch their system prompt
    user_id = None
    if state.get("conversation_id"):
        user_id = get_conversation_user_id(state["conversation_id"])
    
    if user_id:
        system_prompt = get_user_system_prompt(user_id)
    else:
        system_prompt = """
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

Avoid copying documents verbatim. Instead, synthesize and remix previous brand work to generate fresh, original responses tailored to the users query.
        """
    
    user_prompt = f"""
A user asked:
{state['query']}

Here are the most relevant messages or context:
{docs_blob}
"""

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
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
builder.add_edge("web_retrieve", "generate_answer")
builder.add_edge("chat_history_retrieve", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()

# Utility functions for message storage
def get_conversation_user_id(conversation_id: str) -> str | None:
    if not conversation_id:
        return None
    try:
        supabase = current_app.supabase
        res = (
            supabase.table('conversations')
            .select('user_id')
            .eq('id', conversation_id)
            .single()
            .execute()
        )
        return (res.data or {}).get('user_id')
    except Exception as e:
        current_app.logger.warning(f"Could not resolve user_id for conversation {conversation_id}: {e}")
        return None

def store_message(conversation_id, sender, content, user_id=None):
    if not (conversation_id and content):
        return
    try:
        supabase = current_app.supabase

        uid = user_id or get_conversation_user_id(conversation_id)
        if not uid:
            current_app.logger.error(
                f"messages.user_id is NOT NULL; no user_id available "
                f"(conversation_id={conversation_id}). Skipping insert."
            )
            return

        payload = {
            'id': str(uuid4()),
            'conversation_id': conversation_id,
            'user_id': uid,
            'sender': sender,
            'content': content,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        supabase.table('messages').insert(payload).execute()
        
        # Update LangChain memory with new message
        if sender == 'user':
            memory.chat_memory.add_user_message(content)
        elif sender == 'assistant':
            memory.chat_memory.add_ai_message(content)
            
    except APIError as e:
        current_app.logger.exception(f"Failed to store message: {e}")

@retriever_bp.route('/query', methods=['POST'])
def query_api():
    try:
        user = authenticate_user()
        if not user:
            return jsonify({'error': 'Invalid or missing token'}), 401

        data = request.get_json() or {}
        query = data.get("query")
        conversation_id = data.get("conversation_id")
        model_id = data.get("model_id", "claude-sonnet-4-20250514")

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        if conversation_id and not verify_conversation_ownership(conversation_id, user.id):
            return jsonify({'error': 'Unauthorized access to conversation'}), 403

        if conversation_id:
            store_message(conversation_id, "user", query, user_id=user.id)

        app = current_app._get_current_object()

        def generate():
            # 1) Decide method first so we can show a status line BEFORE retrieval.
            state: GraphState = {
                "query": query,
                "method": "web",
                "namespace": "",
                "retrieved_documents": [],
                "web_search_results": "",
                "final_answer": "",
                "conversation_id": conversation_id,
                "user_id": user.id,
                "chat_history": []
            }
            method_state = decide_method(state)
            method = method_state["method"]

            namespace = None
            if method == "pinecone":
                ns_state = decide_namespace(method_state)
                namespace = ns_state["namespace"]
                yield sse("status", f"Retrieving from Pinecone from {namespace}")
                pine_state = retrieve_pinecone({**method_state, "namespace": namespace})
                retrieved_docs = pine_state["retrieved_documents"]
                web_results = ""
            elif method == "web":
                yield sse("status", "Retrieving from web")
                web_state = retrieve_web(method_state)
                retrieved_docs = web_state["retrieved_documents"]
                web_results = web_state["web_search_results"]
            else:
                yield sse("status", "Retrieving from chat history")
                hist_state = retrieve_chat_history(method_state)
                retrieved_docs = hist_state["retrieved_documents"]
                web_results = ""

            # 3) Build prompts with LangChain memory context
            docs_blob = f"User (current query): {query}\n\n---\n\n" + "\n\n---\n\n".join(retrieved_docs[:8])
            system_prompt = get_user_system_prompt(user.id)
            user_prompt = f"A user asked:\n{query}\n\nHere are the most relevant messages or context:\n{docs_blob}"

            # 4) Stream the model with token events
            yield sse("status", "Generating answer…")

            full_answer_chunks = []

            for token in stream_llm_response_sse(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_id=model_id,
                app=app
            ):
                full_answer_chunks.append(token["data"])
                yield sse("token", token["data"])

            # 5) Store & finish
            full_answer = "".join(full_answer_chunks).strip()
            if conversation_id and full_answer:
                store_message(conversation_id, "assistant", full_answer, user_id=user.id)

            yield sse("done", {
                "method": method,
                "namespace": namespace,
                "has_web_results": bool(web_results)
            })

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
        return Response(stream_with_context(generate()), headers=headers)

    except Exception as e:
        current_app.logger.exception(f"Error in query processing: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def stream_llm_response_sse(prompt: str, system_prompt: str, model_id: str, app):
    """
    Yields dicts like {"event": "token", "data": "<delta>"} for SSE formatting.
    """
    if model_id == "claude-sonnet-4-20250514":
        response_stream = app.anthropic.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            max_tokens=3500,
            temperature=0.7,
            stream=True
        )
        for event in response_stream:
            if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                yield {"event": "token", "data": event.delta.text}

    elif model_id == "mistral-large-latest":
        try:
            response_stream = app.mistral.chat.stream(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3500,
                temperature=0.7
            )
            for chunk in response_stream:
                delta = getattr(chunk.data.choices[0].delta, "content", None)
                if delta:
                    yield {"event": "token", "data": delta}
        except Exception as e:
            current_app.logger.error(f"Mistral API error: {e}")
            yield {"event": "token", "data": f"\n[Provider error: {str(e)}]"}

    elif model_id.startswith("gpt-4o"):
        try:
            response_stream = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=3500,
                temperature=0.7,
                stream=True,
            )
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    yield {"event": "token", "data": chunk.choices[0].delta.content}
        except Exception as e:
            current_app.logger.error(f"OpenAI API error: {e}")
            yield {"event": "token", "data": f"\n[Provider error: {str(e)}]"}
    else:
        raise ValueError(f"Unsupported model_id provided: {model_id}")
