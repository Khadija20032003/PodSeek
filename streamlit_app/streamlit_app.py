import streamlit as st
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- Environment & Paths ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ES_HOST, EMBEDDING_MODEL_NAME
from es_search.search import search, format_time

# --- Styling ---
st.set_page_config(page_title="PodSeek", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    /* Main container styling */
    .main { background-color: #f8f9fa; }
    
    /* Clean Header */
    .stHeading h1 { font-weight: 800; color: #1E1E1E; }
    
    /* Source Card Styling */
    .source-card {
        background-color: white;
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 5px solid #6c63ff;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Chunk Text Styling for longer text readability */
    .chunk-text {
        padding-left: 15px;
        padding-right: 15px;
        margin-bottom: 25px;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #4a4a4a;
        border-left: 2px solid #e0e0e0;
    }
    
    /* Pill buttons for suggestions */
    div.stButton > button {
        border-radius: 20px;
        padding: 5px 20px;
        border: 1px solid #6c63ff;
        color: #6c63ff;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #6c63ff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    st.error("GROQ_API_KEY is missing. Check your .env file.")
    st.stop()

@st.cache_resource
def get_es_client(): return Elasticsearch(ES_HOST)

@st.cache_resource
def get_query_embedder(): return SentenceTransformer(EMBEDDING_MODEL_NAME)

es_client = get_es_client()
if not es_client.ping():
    st.error(f"Cannot connect to Elasticsearch at {ES_HOST}")
    st.stop()

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    llm_model = st.selectbox("LLM Model", [
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "groq/compound-mini"
    ])
    top_k = st.slider("Context Chunks", 1, 20, 5)
    
    with st.expander("Advanced ES Settings"):
        enable_knn = st.toggle("Hybrid Search", value=True)
        knn_k = st.number_input("kNN k", 1, 200, 50)
        fuzziness = st.selectbox("Fuzziness", ["AUTO", "Off"])
        include_parent = st.toggle("Parent Context", value=True)

# --- UI Header ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🎙️ PodSeek</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>AI-powered podcast search and synthesis</p>", unsafe_allow_html=True)

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_source_chunks(results):
    if not results: return
    with st.expander(f"📚 View {len(results)} Sources"):
        for idx, result in enumerate(results):
            src = result["_source"]
            
            # 1. Display the metadata card
            st.markdown(f"""
            <div class="source-card">
                <strong>{idx+1}. {src.get('show_name')}</strong><br>
                <small>Episode: {src.get('episode_name')}</small><br>
                <code>{format_time(src.get('start_time', 0))} - {format_time(src.get('end_time', 0))}</code>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Display the full chunk text underneath
            chunk_text = src.get("text", "No text available.")
            st.markdown(f'<div class="chunk-text">{chunk_text}</div>', unsafe_allow_html=True)

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🗣️" if msg["role"]=="user" else "🎙️"):
        st.markdown(msg["content"])
        if "results" in msg:
            render_source_chunks(msg["results"])

# Suggested Queries (Only if no chat)
if not st.session_state.messages:
    st.write("---")
    cols = st.columns(3)
    suggestions = ["Are autonomous cars safe?", "History of Isaac Newton", "Intermittent fasting tips"]
    for i, q in enumerate(suggestions):
        if cols[i].button(q, use_container_width=True):
            st.session_state.initial_query = q
            st.rerun()

# --- Input & Processing ---
user_query = st.chat_input("Ask about a podcast topic...")
if "initial_query" in st.session_state:
    user_query = st.session_state.initial_query
    del st.session_state.initial_query

if user_query:
    st.chat_message("user", avatar="🗣️").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant", avatar="🎙️"):
        with st.status("Searching transcripts...", expanded=False) as status:
            search_response = search(
                es_client, user_query, top_k=top_k, 
                embedder=get_query_embedder(), enable_knn=enable_knn
            )
            hits = search_response.get("hits", [])
            status.update(label=f"Found {len(hits)} relevant segments", state="complete")

        if not hits:
            resp = "No results found in the podcast database."
            st.warning(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
        else:
            llm = ChatGroq(api_key=api_key, model_name=llm_model, streaming=True)
            
            # Format context for the LLM
            context = "\n".join([f"- {r['_source'].get('text')}" for r in hits])
            full_prompt = f"Question: {user_query}\nContext: {context}\nAnswer:"
            
            response_placeholder = st.empty()
            full_response = response_placeholder.write_stream(llm.stream(full_prompt))
            
            render_source_chunks(hits)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "results": hits
            })