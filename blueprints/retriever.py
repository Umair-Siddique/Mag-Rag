import os
from typing import TypedDict, List, Literal
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
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


class GraphState(TypedDict):
    query: str
    method: Literal["web", "pinecone", "chat_history"]
    namespace: str
    retrieved_documents: List[str]
    web_search_results: str
    final_answer: str
    conversation_id: str
    user_id: str  # Add user_id to the state

# --- Add this tiny helper somewhere near the top ---
def sse(event: str, data) -> str:
    # Formats a Server-Sent Event: event: <type>\n data: <json>\n\n
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
        # Return fallback prompt on error
        return """
You are a strategic brand assistant. Please provide helpful, insightful responses based on the available context and user queries.
        """


# Authentication helper functions
def _extract_access_token() -> str | None:
    """
    Pull the JWT from the Authorization header:
        Authorization: Bearer <token>
    Return None if the header is missing or malformed.
    """
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header.split(' ', 1)[1].strip() or None
    return None

def authenticate_user() -> object | None:
    """
    Authenticate the incoming request by validating the JWT against Supabase.
    Returns the Supabase user object on success, otherwise None.
    """
    access_token = _extract_access_token()
    if not access_token:
        return None

    supabase = current_app.supabase
    try:
        return supabase.auth.get_user(access_token).user
    except Exception:
        return None

def verify_conversation_ownership(conversation_id: str, user_id: str) -> bool:
    """
    Verify that the conversation belongs to the authenticated user.
    """
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
    - Use "web" for recent external trends or events.
    - Use "pinecone" for brand or consumer insights.
    - Use "chat_history" for conversational or follow-up queries.

    Query: {query}
    Respond ONLY with "web", "pinecone", or "chat_history".
    """
    completion = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    method = completion.choices[0].message.content.strip().lower()
    return {**state, "method": method}

def decide_namespace(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = """
    Based on the query, choose the most appropriate namespace from these 4 options:

    1. brand-positioning: For queries about branding, brand strategy, positioning, metaphors, archetypes, semiotics, brand foundations, brand briefs, memory structures, cultural positioning, brand narratives, manifestos, values, and brand voice.

    2. insights: For queries about consumer behavior, market trends, cultural insights, tensions, consumer research, market analysis, behavioral patterns, cultural shifts, and trend analysis.

    3. events: For queries about events, activations, launch events, community building, cultural programs, IRL experiences, event design, and experiential marketing.

    4. common: For queries about operations, go-to-market strategies, business models, scaling, investment pitches, launch campaigns, DTC strategies, influencer marketing, and general business operations.

    Query: {query}
    
    Respond ONLY with one of these exact namespaces: brand-positioning, insights, events, or common (no quotes, no punctuation, no extra words).
    """.format(query=query)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    namespace = completion.choices[0].message.content.strip().lower()
    namespace = namespace.strip('"').strip("'")

    # Update validation to include all 4 namespaces
    if namespace not in ("brand-positioning", "insights", "events", "common"):
        namespace = "common"  # Default fallback

    return {**state, "namespace": namespace}

def retrieve_pinecone(state: GraphState) -> GraphState:
    print(f"Retrieving from Pinecone namespace: {state['namespace']}")
    query_embedding = embeddings_model.embed_query(state["query"])
    # Use the Pinecone instance from extensions.py
    index = current_app.pinecone.Index("career-counseling-documents")
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=state["namespace"])
    docs = [match['metadata'].get('text', '') for match in results.get('matches', [])]
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
    supabase = current_app.supabase
    conv_id = state.get("conversation_id")

    if not conv_id:
        return {
            **state,
            "retrieved_documents": ["No conversation_id provided; cannot load chat history."],
            "web_search_results": ""
        }

    try:
        # Get the last 10 messages (5 user-assistant pairs) in chronological order
        res = (
            supabase
            .table('messages')
            .select('content, sender, created_at')
            .eq('conversation_id', conv_id)
            .order('created_at', desc=False)  # Chronological order (oldest first)
            .limit(10)  # Get 10 messages to ensure we have 5 complete conversations
            .execute()
        )

        rows = res.data or []
        
        # Group messages into conversations (user + assistant pairs)
        conversations = []
        i = 0
        while i < len(rows) - 1:  # Need at least 2 messages for a conversation
            if rows[i]['sender'] == 'user' and rows[i + 1]['sender'] == 'assistant':
                # Found a complete conversation pair
                conversation = f"User: {rows[i]['content']}\nAssistant: {rows[i + 1]['content']}"
                conversations.append(conversation)
                i += 2  # Skip both messages
            else:
                i += 1  # Skip incomplete conversation
        
        # Take only the last 5 conversations and reverse order (most recent first)
        last_5_conversations = conversations[-5:] if len(conversations) > 5 else conversations
        last_5_conversations.reverse()  # Most recent conversation on top
        
        # If no complete conversations found, return individual messages
        if not last_5_conversations:
            formatted = [f"{row['sender'].title()}: {row['content']}" for row in rows[-5:]]
            formatted.reverse()  # Most recent on top
        else:
            formatted = last_5_conversations

        return {
            **state,
            "retrieved_documents": formatted,
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
    history_docs = state["retrieved_documents"]
    docs_blob = f"User (current query): {state['query']}\n\n---\n\n" + "\n\n---\n\n".join(history_docs)

    # Get the user ID from the conversation to fetch their system prompt
    user_id = None
    if state.get("conversation_id"):
        user_id = get_conversation_user_id(state["conversation_id"])
    
    # If we can't get user_id from conversation, we'll use a default prompt
    if user_id:
        system_prompt = get_user_system_prompt(user_id)
    else:
        # Fallback to default prompt if no user_id available
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
            }
            method_state = decide_method(state)                         # uses OpenAI
            method = method_state["method"]

            namespace = None
            if method == "pinecone":
                ns_state = decide_namespace(method_state)               # uses OpenAI
                namespace = ns_state["namespace"]
                yield sse("status", f"Retrieving from Pinecone from {namespace}")
                # 2) Retrieve from Pinecone
                pine_state = retrieve_pinecone({**method_state, "namespace": namespace})
                retrieved_docs = pine_state["retrieved_documents"]
                web_results = ""
            elif method == "web":
                yield sse("status", "Retrieving from web")
                # 2) Retrieve from Web
                web_state = retrieve_web(method_state)
                retrieved_docs = web_state["retrieved_documents"]
                web_results = web_state["web_search_results"]
            else:
                yield sse("status", "Retrieving from chat history")
                hist_state = retrieve_chat_history(method_state)
                retrieved_docs = hist_state["retrieved_documents"]
                web_results = ""

            # 3) Build prompts
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
                # token already formatted as `event: token`
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

        # NOTE: Use SSE content type and no buffering
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # for nginx, if applicable
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

    elif model_id.startswith("gpt-"):
        with client.chat.completions.stream(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3500,
            temperature=0.7,
        ) as stream:
            for event in stream:
                try:
                    delta = event.choices[0].delta.content
                    if delta:
                        yield {"event": "token", "data": delta}
                except Exception:
                    continue
    else:
        raise ValueError(f"Unsupported model_id provided: {model_id}")
