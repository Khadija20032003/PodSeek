import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import config and es_search
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ES_HOST
from es_search.search import search, format_time
from elasticsearch import Elasticsearch
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "").strip() 

if not api_key:
    st.error("GROQ_API_KEY is missing from environment. Please add it to your .env file.")
    st.stop()

# --- STREAMLIT UI ---
st.set_page_config(page_title="PodSeek", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
/* Make font sizes bigger across the app */
html, body, [class*="css"] {
    font-size: 16px !important;
}
.stMarkdown p, .stMarkdown div, .stChatMessage {
    font-size: 1.15rem !important;
}

/* Force all buttons to act as inline inline-block tags instead of grid columns */
div.stButton {
    display: inline-block !important;
    width: auto !important;
    margin-right: 8px !important;
    margin-bottom: 8px !important;
}
div.stButton > button {
    border-radius: 20px !important; /* give them a 'pill' / 'tag' look */
    padding: 2px 14px !important;
    border: 1px solid #d1d5db !important;
}
</style>
""", unsafe_allow_html=True)

# --- ELASTICSEARCH CONNECTION ---
@st.cache_resource
def get_es_client():
    return Elasticsearch(ES_HOST)

es_client = get_es_client()

# --- THE LLM PROMPT TEMPLATE ---
rag_prompt = PromptTemplate.from_template("""
You are a helpful podcast assistant. Answer the user's question by synthesizing ALL relevant information from the provided podcast transcripts. 

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided transcripts.
2. If the transcripts do not contain the answer, say "I cannot find the answer in the current podcast database."
3. You MUST cite EVERY podcast ID and timestamp that contributed to your answer. If multiple podcasts discuss the topic, you must include information from all of them.

User Question: {question}

Podcast Transcripts:
{context}

Answer:
""")

st.title("🎙️ PodSeek")
st.markdown("Search the database to find exactly where a topic was discussed.")

# Settings moved properly to the top, styled differently
col_settings, col_empty = st.columns([1, 3])
with col_settings:
    with st.expander("Search & AI Parameters", expanded=False):
        top_k_slider = st.slider("Podcast Chunks to Retrieve", 1, 20, 5)
        llm_model = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "groq/compound-mini", "openai/gpt-oss-120b", "openai/gpt-oss-20b"], 0)

# Initialize LLM based on settings (temperature is fixed to 0.0 for correctness)
llm = ChatGroq(
    temperature=0.0,
    groq_api_key=api_key,
    model_name=llm_model
)

# Custom Avatars for the Chat UI
USER_AVATAR = "🗣️"
BOT_AVATAR = "🎙️"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "results" in message and message["results"]:
            with st.expander(f"View {len(message['results'])} source chunks", expanded=False):
                for idx, result in enumerate(message['results']):
                    src = result["_source"]
                    score = result.get("_score", 0)
                    episode = src.get("episode_name", "Unknown Episode")
                    show = src.get("show_name", "Unknown Show")
                    start = format_time(src.get("start_time", 0))
                    end = format_time(src.get("end_time", 0))
                    highlights = result.get("highlight", {}).get("text", [])
                    display_text = "... " + " ... ".join(highlights) + " ..." if highlights else src.get("text", "")[:300] + "..."
                    
                    st.markdown(f"**{idx+1}. {show}** - *{episode}*")
                    st.markdown(f"⏱️ `{start} - {end}` &nbsp;&nbsp;🎯 `Score: {score:.2f}`")
                    st.caption(display_text)
                    st.divider()

# Suggested Queries Container
if not st.session_state.messages:
    st.markdown("##### Try asking about:")
    suggested_queries = ["Are autonomous cars safe?", "Who is Isaac Newton?", "What is intermittent fasting?"]
    for query in suggested_queries:
        if st.button(query):
            st.session_state.initial_query = query
            st.rerun()

# Determine user input source
user_query = st.chat_input("Ask PodSeek")
if "initial_query" in st.session_state:
    user_query = st.session_state.initial_query
    del st.session_state.initial_query

# Main Chat Logic
if user_query:
    # Display user message in chat message container
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Assistant Response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        status_placeholder = st.empty()
        
        with status_placeholder.status("Searching podcasts...", expanded=True) as status:
            st.write("📡 Connecting to Elasticsearch database...")
            search_response = search(es_client, user_query, top_k=top_k_slider)
            search_results = search_response.get("hits", [])
            
            if not search_results:
                st.write("⚠️ No matching chunks found.")
                status.update(label="No results found.", state="error")
                found_initial = False
            else:
                st.write(f"📥 Retrieved {len(search_results)} relevant chunks.")
                status.update(label="Found chunks, preparing context...", state="running")
                
                # 2. Format Context
                context_string = ""
                for result in search_results:
                    src = result["_source"]
                    episode = src.get("episode_name", "Unknown Episode")
                    show = src.get("show_name", "Unknown Show")
                    start = format_time(src.get("start_time", 0))
                    end = format_time(src.get("end_time", 0))
                    text_content = src.get("parent_text", src.get("text", "")) 
                    
                    context_string += f"\n- [{show} - {episode} | {start}-{end}]: {text_content}"
                
                # 3. Create prompt
                st.write("🧠 Synthesizing information with AI...")
                formatted_prompt = rag_prompt.format(question=user_query, context=context_string)
                status.update(label="Ready! Generating reply...", state="complete", expanded=False)
                found_initial = True

        if not found_initial:
            # Replaces the expandable status with a generic non-expandable error text
            status_placeholder.error("I'm sorry, no relevant podcast chunks were found in the database. Please try another topic.")
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, no relevant podcast chunks were found in the database. Please try another topic.", "results": []})
        else:
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response instead of waiting for full generation
            for chunk in llm.stream(formatted_prompt):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            # Remove blinking cursor after completion
            response_placeholder.markdown(full_response)
            
            suggestion = search_response.get("suggestion")
            if suggestion and suggestion.lower() != user_query.lower():
                st.info(f"💡 Did you mean: **{suggestion}**?")

            # Only show chunks if the LLM didn't completely fail to find the answer
            if "i cannot find the answer" in full_response.lower():
                show_chunks = False
                # Replaces the status with a non-expandable warning!
                status_placeholder.warning("The topic was not discussed in the retrieved chunks.")
            else:
                show_chunks = True

            # Save the assistant response to state history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "results": search_results if show_chunks else []
            })

            # Present expandable chunks for the immediate result
            if show_chunks:
                with st.expander(f"View {len(search_results)} source chunks", expanded=False):
                    for idx, result in enumerate(search_results):
                        src = result["_source"]
                        score = result.get("_score", 0)
                        episode = src.get("episode_name", "Unknown Episode")
                        show = src.get("show_name", "Unknown Show")
                        start = format_time(src.get("start_time", 0))
                        end = format_time(src.get("end_time", 0))
                        highlights = result.get("highlight", {}).get("text", [])
                        display_text = "... " + " ... ".join(highlights) + " ..." if highlights else src.get("text", "")[:300] + "..."
                        
                        st.markdown(f"**{idx+1}. {show}** - *{episode}*")
                        st.markdown(f"⏱️ `{start} - {end}` &nbsp;&nbsp;🎯 `Score: {score:.2f}`")
                        st.caption(display_text)
                        st.divider()