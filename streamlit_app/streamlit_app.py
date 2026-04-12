import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import sys
import time
from pathlib import Path
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("GROQ_API_KEY").strip() 

llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    streaming=True,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ES_HOST, EMBEDDING_MODEL_NAME
from es_search.search import search as hybrid_search
from es_search.search import format_time
import es_search.search as es_search_module

@st.cache_resource
def get_es_client() -> Elasticsearch:
    return Elasticsearch(ES_HOST)

@st.cache_resource
def get_query_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def elasticsearch_search(user_query: str, top_k: int = 5, category: str = None, show_name: str = None, embedder=None, enable_knn: bool = True, knn_k: int = 50, num_candidates: int = 100, window_size: int = 100, rank_constant: int = 60, fuzziness: str = "AUTO", include_parent_text: bool = True, include_category_boost: bool = True, include_title_boost: bool = True):
    es = get_es_client()
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {ES_HOST}")
    if embedder is None:
        embedder = get_query_embedder()
    result = hybrid_search(
        es,
        user_query,
        top_k=top_k,
        category=category,
        show_name=show_name,
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
    return result

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

# --- STREAMLIT UI ---
st.set_page_config(page_title="PodSeek", page_icon="🎙️")
st.title("🎙️ PodSeek")
st.write("Search the database to find exactly where a topic was discussed.")

top_k = 5
enable_knn = True
knn_k = 50
num_candidates = 100
window_size = 100
rank_constant = 60
fuzziness = "AUTO"
include_parent_text = True
include_category_boost = True
include_title_boost = True
category_filter = ""
show_filter = ""

with st.sidebar.expander("Search settings", expanded=False):
    top_k = st.slider("Results", min_value=1, max_value=20, value=top_k, step=1)

    enable_knn = st.toggle("Hybrid (BM25 + Vector)", value=enable_knn)
    knn_k = st.slider("kNN k", min_value=1, max_value=200, value=knn_k, step=1, disabled=not enable_knn)
    num_candidates = st.slider(
        "kNN num_candidates",
        min_value=10,
        max_value=500,
        value=num_candidates,
        step=10,
        disabled=not enable_knn,
    )

    window_size = st.slider(
        "RRF window_size",
        min_value=10,
        max_value=300,
        value=window_size,
        step=10,
    )
    rank_constant = st.slider(
        "RRF rank_constant",
        min_value=1,
        max_value=100,
        value=rank_constant,
        step=1,
    )

    fuzziness_mode = st.selectbox("Fuzziness", options=["AUTO", "Off"], index=0)
    fuzziness = None if fuzziness_mode == "Off" else "AUTO"

    include_parent_text = st.toggle("Use parent context", value=include_parent_text)
    include_category_boost = st.toggle("Boost category field", value=include_category_boost)
    include_title_boost = st.toggle("Boost show/episode fields", value=include_title_boost)

    st.subheader("Filters")
    category_filter = st.text_input("Category (exact)", value=category_filter)
    show_filter = st.text_input("Show name (exact)", value=show_filter)

# The Search Bar
if "query" not in st.session_state:
    st.session_state.query = ""

with st.form("search_form"):
    query = st.text_input("What topic are you looking for? (e.g., 'Higgs Boson')", key="query")
    submitted = st.form_submit_button("Search Podcasts")

# The Search Button Action
if submitted:
    if query:

        # 1) Elasticsearch phase: show results ASAP
        with st.spinner("Searching Elasticsearch..."):
            try:
                es_result = elasticsearch_search(
                    query,
                    top_k=top_k,
                    category=category_filter or None,
                    show_name=show_filter or None,
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
                timings = es_result.get("timings", {}) or {}
                query_suggestions = es_result.get("query_suggestions", []) or []
                print(f"[Debug] es_search.search loaded from: {getattr(es_search_module, '__file__', 'unknown')}")
                print(f"[Latency] Raw hybrid timings dict: {timings}")
                print(
                    "[Latency] Hybrid search timings (ms): "
                    f"embed={timings.get('embed_ms', 0.0):.3f} "
                    f"lexical_es={timings.get('lexical_es_ms', 0.0):.3f} "
                    f"knn_es={timings.get('knn_es_ms', 0.0):.3f} "
                    f"rrf={timings.get('rrf_ms', 0.0):.3f} "
                    f"total={timings.get('total_ms', 0.0):.3f}"
                )

                hits = es_result.get("hits", [])
                search_results = []
                for hit in hits:
                    src = hit.get("_source", {})
                    start = format_time(float(src.get("start_time", 0.0)))
                    end = format_time(float(src.get("end_time", 0.0)))
                    show_name = src.get("show_name", "Unknown")
                    episode_name = src.get("episode_name", "Unknown")
                    podcast_id = f"{show_name} — {episode_name}"

                    highlight_texts = hit.get("highlight", {}).get("text", [])
                    if highlight_texts:
                        snippet = " ".join(str(h) for h in highlight_texts)
                    else:
                        snippet = str(src.get("text", ""))

                    search_results.append(
                        {
                            "podcast_id": podcast_id,
                            "start": start,
                            "end": end,
                            "text": snippet,
                        }
                    )
            except Exception as e:
                st.error(str(e))
                search_results = []
                query_suggestions = []

        if query_suggestions:
            with st.expander("Suggested queries", expanded=False):
                for i, s in enumerate(query_suggestions[:3]):
                    label = s if len(s) <= 80 else (s[:77] + "...")
                    if st.button(label, key=f"query_suggestion_{i}"):
                        st.session_state.query = s
                        st.rerun()

        st.subheader("Raw Audio Chunks Found")
        if search_results:
            for result in search_results:
                st.info(
                    f"**{result['podcast_id']}** ({result['start']} - {result['end']})\n\n{result['text']}"
                )
        else:
            st.warning("No Elasticsearch results found.")

        # 2) LLM phase: stream answer after ES results are visible
        st.subheader("Podcast Results:")
        answer_placeholder = st.empty()

        context_string = ""
        for result in search_results:
            context_string += f"\n- [{result['podcast_id']} | {result['start']}-{result['end']}]: {result['text']}"

        formatted_prompt = rag_prompt.format(question=query, context=context_string)

        try:
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
                final_text = answer_placeholder.write_stream(_stream_text())

            t_done = time.perf_counter()
            ttft_ms = ((t_first_token["t"] - t_llm0) * 1000.0) if t_first_token["t"] else 0.0
            total_ms = (t_done - t_llm0) * 1000.0
            print(f"[Latency] LLM stream TTFT_ms={ttft_ms:.1f} total_ms={total_ms:.1f}")
        except Exception:
            t_llm0 = time.perf_counter()
            response = llm.invoke(formatted_prompt)
            t_done = time.perf_counter()
            total_ms = (t_done - t_llm0) * 1000.0
            answer_placeholder.write(response.content)
            print(f"[Latency] LLM invoke total_ms={total_ms:.1f}")
    else:
        st.warning("Please enter a search query first.")