import streamlit as st
import os
import sys
import time
from pathlib import Path

# Prevent Hugging Face from hanging the thread during sentence-transformer network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ES_HOST, EMBEDDING_MODEL_NAME
from es_search.search import search, format_time

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    st.error("GROQ_API_KEY is missing from environment. Please add it to your .env file.")
    st.stop()


@st.cache_resource
def get_es_client() -> Elasticsearch:
    return Elasticsearch(ES_HOST)


@st.cache_resource
def get_query_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


es_client = get_es_client()
if not es_client.ping():
    st.error(f"Cannot connect to Elasticsearch at {ES_HOST}")
    st.stop()


rag_prompt = PromptTemplate.from_template(
    """
You are a helpful podcast assistant. Answer the user's question by synthesizing information ONLY from the provided podcast transcript chunks.

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided transcripts.
2. If the transcripts do not contain the answer, say "I cannot find the answer in the current podcast database."
3. You MUST cite EVERY podcast show/episode and timestamp that contributed to your answer.
4. Be concise: keep the answer under ~8 short sentences OR up to 6 bullet points. No long introductions.

User Question: {question}

Podcast Transcripts:
{context}

Answer:
"""
)


st.set_page_config(page_title="PodSeek", page_icon="🎙️", layout="centered")

st.markdown(
    """
<style>
/* Make font sizes bigger across the app */
html, body, [class*="css"] {
    font-size: 16px !important;
}
.stMarkdown p, .stMarkdown div, .stChatMessage {
    font-size: 1.15rem !important;
}

/* Force all buttons to act as inline inline-block tags instead of grid columns */
div.stButton, div.stButton > button {
    display: inline-block !important;
    width: auto !important;
}
div.stButton {
    margin-right: 8px !important;
    margin-bottom: 8px !important;
}
div.stButton > button {
    border-radius: 20px !important;
    padding: 2px 14px !important;
    border: 1px solid #d1d5db !important;
}

div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    text-align: center !important;
    align-items: center !important;
    justify-content: center !important;
    display: flex !important;
    flex-direction: column !important;
}

div[data-testid="column"] > div[data-testid="stVerticalBlock"] > div {
    width: 100% !important;
}

div.element-container:has(div.stButton) {
    display: inline-block !important;
    width: auto !important;
}

.stExpander {
    text-align: left;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center;'>🎙️ PodSeek</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Search the database to find exactly where a topic was discussed.</p>",
    unsafe_allow_html=True,
)


default_top_k = 5
default_llm_model = "llama-3.1-8b-instant"
default_enable_knn = True
default_knn_k = 50
default_num_candidates = 100
default_window_size = 100
default_rank_constant = 60
default_fuzziness = "AUTO"
default_include_parent_text = True
default_include_category_boost = True
default_include_title_boost = True
default_category_filter = ""
default_show_filter = ""

col_empty1, col_settings, col_empty2 = st.columns([1, 2, 1])
with col_settings:
    with st.expander("Search & AI Parameters", expanded=False):
        top_k = st.slider("Podcast Chunks to Retrieve", 1, 20, default_top_k)
        llm_model = st.selectbox(
            "LLM Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "groq/compound-mini",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ],
            0,
        )

        with st.expander("Advanced Elasticsearch settings", expanded=False):
            enable_knn = st.toggle("Hybrid (BM25 + Vector)", value=default_enable_knn)
            knn_k = st.slider("kNN k", min_value=1, max_value=200, value=default_knn_k, step=1, disabled=not enable_knn)
            num_candidates = st.slider(
                "kNN num_candidates",
                min_value=10,
                max_value=500,
                value=default_num_candidates,
                step=10,
                disabled=not enable_knn,
            )
            window_size = st.slider("RRF window_size", min_value=10, max_value=300, value=default_window_size, step=10)
            rank_constant = st.slider("RRF rank_constant", min_value=1, max_value=100, value=default_rank_constant, step=1)
            fuzziness_mode = st.selectbox("Fuzziness", options=["AUTO", "Off"], index=0)
            fuzziness = None if fuzziness_mode == "Off" else "AUTO"

            include_parent_text = st.toggle("Use parent context", value=default_include_parent_text)
            include_category_boost = st.toggle("Boost category field", value=default_include_category_boost)
            include_title_boost = st.toggle("Boost show/episode fields", value=default_include_title_boost)

            st.subheader("Filters")
            category_filter = st.text_input("Category (exact)", value=default_category_filter)
            show_filter = st.text_input("Show name (exact)", value=default_show_filter)


llm = ChatGroq(
    temperature=0.0,
    groq_api_key=api_key,
    model_name=llm_model,
    streaming=True,
)


USER_AVATAR = "🗣️"
BOT_AVATAR = "🎙️"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "results" in message and message["results"]:
            with st.expander(f"View {len(message['results'])} source chunks", expanded=False):
                for idx, result in enumerate(message["results"]):
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


if not st.session_state.messages:
    st.markdown("<h5 style='text-align: center;'>Try asking about:</h5>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    suggested_queries = ["Are autonomous cars safe?", "Who is Isaac Newton?", "What is intermittent fasting?"]
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        for q in suggested_queries:
            if st.button(q):
                st.session_state.initial_query = q
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


user_query = st.chat_input("Ask PodSeek")
if "initial_query" in st.session_state:
    user_query = st.session_state.initial_query
    del st.session_state.initial_query


def _format_context(results: list[dict]) -> str:
    context_string = ""
    for result in results:
        src = result.get("_source", {})
        episode = src.get("episode_name", "Unknown Episode")
        show = src.get("show_name", "Unknown Show")
        start = format_time(src.get("start_time", 0))
        end = format_time(src.get("end_time", 0))
        text_content = src.get("parent_text", src.get("text", ""))
        context_string += f"\n- [{show} - {episode} | {start}-{end}]: {text_content}"
    return context_string


if user_query:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        status_placeholder = st.empty()

        with status_placeholder.status("Searching podcasts...", expanded=True) as status:
            st.write("📡 Querying Elasticsearch...")
            embedder = get_query_embedder()
            search_response = search(
                es_client,
                user_query,
                top_k=top_k,
                category=category_filter or None,
                show_name=show_filter or None,
                embedder=embedder,
                enable_knn=enable_knn,
                knn_k=knn_k,
                num_candidates=num_candidates,
                window_size=window_size,
                rank_constant=rank_constant,
                fuzziness=fuzziness,
                include_parent_text=include_parent_text,
                include_category_boost=include_category_boost,
                include_title_boost=include_title_boost,
            )
            search_results = search_response.get("hits", []) or []

            if not search_results:
                st.write("⚠️ No matching chunks found.")
                status.update(label="No results found.", state="error")
                found_initial = False
                formatted_prompt = None
            else:
                st.write(f"📥 Retrieved {len(search_results)} relevant chunks.")
                status.update(label="Found chunks, preparing context...", state="running")
                context_string = _format_context(search_results)
                formatted_prompt = rag_prompt.format(question=user_query, context=context_string)
                st.write("🧠 Synthesizing information with AI...")
                status.update(label="Ready! Generating reply...", state="complete", expanded=False)
                found_initial = True

        if not found_initial:
            msg = "I'm sorry, no relevant podcast chunks were found in the database. Please try another topic."
            status_placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "results": []})
        else:
            response_placeholder = st.empty()
            t_llm0 = time.perf_counter()
            t_first_token = {"t": None}

            def _stream_text():
                for chunk in llm.stream(formatted_prompt):
                    chunk_text = getattr(chunk, "content", None)
                    if not chunk_text:
                        continue
                    if t_first_token["t"] is None:
                        t_first_token["t"] = time.perf_counter()
                    yield chunk_text

            with st.spinner("Generating answer..."):
                full_response = response_placeholder.write_stream(_stream_text())

            t_done = time.perf_counter()
            ttft_ms = ((t_first_token["t"] - t_llm0) * 1000.0) if t_first_token["t"] else 0.0
            total_ms = (t_done - t_llm0) * 1000.0
            print(f"[Latency] LLM stream TTFT_ms={ttft_ms:.1f} total_ms={total_ms:.1f}")

            query_suggestions = search_response.get("query_suggestions", []) or []
            if query_suggestions:
                with st.expander("Suggested queries", expanded=False):
                    for i, s in enumerate(query_suggestions[:3]):
                        label = s if len(s) <= 80 else (s[:77] + "...")
                        if st.button(label, key=f"query_suggestion_{i}"):
                            st.session_state.initial_query = s
                            st.rerun()

            if "i cannot find the answer" in str(full_response).lower():
                show_chunks = False
                status_placeholder.warning("The topic was not discussed in the retrieved chunks.")
            else:
                show_chunks = True

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": str(full_response),
                    "results": search_results if show_chunks else [],
                }
            )

            if show_chunks:
                with st.expander(f"View {len(search_results)} source chunks", expanded=False):
                    for idx, result in enumerate(search_results):
                        src = result.get("_source", {})
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