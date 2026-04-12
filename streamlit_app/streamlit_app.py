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

def elasticsearch_search(user_query: str, top_k: int = 5):
    es = get_es_client()
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {ES_HOST}")
    embedder = get_query_embedder()
    result = hybrid_search(es, user_query, top_k=top_k, embedder=embedder)
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

# The Search Bar
query = st.text_input("What topic are you looking for? (e.g., 'Higgs Boson')")

# The Search Button Action
if st.button("Search Podcasts"):
    if query:
        # 1) Elasticsearch phase: show results ASAP
        with st.spinner("Searching Elasticsearch..."):
            try:
                es_result = elasticsearch_search(query, top_k=5)
                timings = es_result.get("timings", {}) or {}
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