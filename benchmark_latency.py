"""
benchmark_latency.py — Headless latency benchmark for PodSeek.

Measures average processing times across 20 test queries:
  1. Retrieval phase: query encoding + Elasticsearch hybrid search
  2. Generation phase: LLM streaming (TTFT + total) for two Groq models

Usage:
    python benchmark_latency.py
"""

import os
import sys
import time
import statistics
from pathlib import Path

# --- Environment (match streamlit_app.py) ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ES_HOST, EMBEDDING_MODEL_NAME
from es_search.search import search, format_time

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    sys.exit("ERROR: GROQ_API_KEY is missing. Add it to your .env file.")

# --- Clients ---
print("Initializing Elasticsearch client...")
es_client = Elasticsearch(ES_HOST)
if not es_client.ping():
    sys.exit(f"ERROR: Cannot connect to Elasticsearch at {ES_HOST}")

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- RAG prompt (same as streamlit_app.py) ---
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


# --- Test queries ---
TEST_QUERIES = [
    "What are the benefits of intermittent fasting?",
    "Are autonomous cars safe?",
    "Who is Isaac Newton and what did he discover?",
    "How does cryptocurrency mining work?",
    "What causes climate change?",
    "How does the human immune system function?",
    "What is quantum computing?",
    "How do vaccines work?",
    "What are the risks of artificial intelligence?",
    "How does meditation affect the brain?",
    "What is the history of the internet?",
    "How do black holes form?",
    "What are the effects of sleep deprivation?",
    "How does blockchain technology work?",
    "What is the role of DNA in genetics?",
    "How do electric cars compare to gas cars?",
    "What are the principles of machine learning?",
    "How does the stock market work?",
    "What is the impact of social media on mental health?",
    "How do renewable energy sources compare?",
]

# --- Default hybrid search settings (from streamlit_app.py) ---
SEARCH_DEFAULTS = dict(
    top_k=5,
    enable_knn=True,
    knn_k=50,
    num_candidates=100,
    window_size=100,
    rank_constant=10,
    fuzziness="AUTO",
    include_parent_text=True,
    include_category_boost=False,
    include_title_boost=False,
)

LLM_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

SLEEP_BETWEEN_LLM_CALLS = 1.0  # seconds


# ============================================================
# Phase 1: Retrieval benchmark
# ============================================================
def run_retrieval_benchmark():
    print("\n" + "=" * 70)
    print("  PHASE 1: Retrieval Latency Benchmark (20 queries)")
    print("=" * 70)

    embed_times = []
    lex_es_times = []
    knn_es_times = []
    rrf_times = []
    search_total_times = []
    retrieval_total_times = []  # embed + search_total

    formatted_prompts = []  # store for phase 2

    for i, query in enumerate(TEST_QUERIES):
        print(f"  [{i+1:2d}/20] \"{query}\"")

        t_retrieval_start = time.perf_counter()

        search_response = search(
            es_client,
            query,
            embedder=embedder,
            **SEARCH_DEFAULTS,
        )

        t_retrieval_end = time.perf_counter()

        timings = search_response.get("timings", {})
        hits = search_response.get("hits", []) or []

        embed_ms = timings.get("embed_ms", 0.0)
        lex_ms = timings.get("lexical_es_ms", 0.0)
        knn_ms = timings.get("knn_es_ms", 0.0)
        rrf_ms = timings.get("rrf_ms", 0.0)
        search_total_ms = timings.get("total_ms", 0.0)
        retrieval_total_ms = (t_retrieval_end - t_retrieval_start) * 1000.0

        embed_times.append(embed_ms)
        lex_es_times.append(lex_ms)
        knn_es_times.append(knn_ms)
        rrf_times.append(rrf_ms)
        search_total_times.append(search_total_ms)
        retrieval_total_times.append(retrieval_total_ms)

        # Build the formatted prompt for LLM phase
        if hits:
            context_string = _format_context(hits)
            formatted_prompts.append(
                rag_prompt.format(question=query, context=context_string)
            )
        else:
            formatted_prompts.append(None)

        print(
            f"         embed={embed_ms:.1f}ms  "
            f"lex={lex_ms:.1f}ms  knn={knn_ms:.1f}ms  "
            f"rrf={rrf_ms:.1f}ms  total={search_total_ms:.1f}ms  "
            f"retrieval={retrieval_total_ms:.1f}ms  "
            f"hits={len(hits)}"
        )

    return {
        "embed": embed_times,
        "lex_es": lex_es_times,
        "knn_es": knn_es_times,
        "rrf": rrf_times,
        "search_total": search_total_times,
        "retrieval_total": retrieval_total_times,
        "prompts": formatted_prompts,
    }


# ============================================================
# Phase 2: LLM generation benchmark
# ============================================================
def run_llm_benchmark(prompts: list):
    results = {}

    for model_name in LLM_MODELS:
        print(f"\n  Benchmarking LLM: {model_name}")
        print("  " + "-" * 50)

        llm = ChatGroq(
            temperature=0.0,
            groq_api_key=api_key,
            model_name=model_name,
            streaming=True,
        )

        ttft_list = []
        total_list = []

        for i, prompt in enumerate(prompts):
            if prompt is None:
                print(f"    [{i+1:2d}/20] SKIPPED (no search results)")
                continue

            print(f"    [{i+1:2d}/20] Generating...", end="", flush=True)

            t_llm0 = time.perf_counter()
            t_first_token = None
            full_response = ""

            for chunk in llm.stream(prompt):
                chunk_text = getattr(chunk, "content", None)
                if not chunk_text:
                    continue
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                full_response += chunk_text

            t_done = time.perf_counter()

            ttft_ms = ((t_first_token - t_llm0) * 1000.0) if t_first_token else 0.0
            total_ms = (t_done - t_llm0) * 1000.0

            ttft_list.append(ttft_ms)
            total_list.append(total_ms)

            print(
                f"  TTFT={ttft_ms:.1f}ms  total={total_ms:.1f}ms  "
                f"tokens~{len(full_response.split())}"
            )

            time.sleep(SLEEP_BETWEEN_LLM_CALLS)

        results[model_name] = {
            "ttft": ttft_list,
            "total": total_list,
        }

    return results


# ============================================================
# Reporting
# ============================================================
def _avg(lst):
    return statistics.mean(lst) if lst else 0.0


def _std(lst):
    return statistics.stdev(lst) if len(lst) > 1 else 0.0


def print_report(retrieval: dict, llm_results: dict):
    print("\n\n" + "=" * 70)
    print("  BENCHMARK RESULTS — Averages over 20 queries")
    print("=" * 70)

    print("\n--- Retrieval Phase ---")
    print(f"  Query Encoding Time (ms):      {_avg(retrieval['embed']):.1f}  "
          f"(σ={_std(retrieval['embed']):.1f})")
    print(f"  Lexical ES Search (ms):        {_avg(retrieval['lex_es']):.1f}  "
          f"(σ={_std(retrieval['lex_es']):.1f})")
    print(f"  kNN ES Search (ms):            {_avg(retrieval['knn_es']):.1f}  "
          f"(σ={_std(retrieval['knn_es']):.1f})")
    print(f"  RRF Fusion (ms):               {_avg(retrieval['rrf']):.1f}  "
          f"(σ={_std(retrieval['rrf']):.1f})")
    print(f"  ES Search Total (ms):          {_avg(retrieval['search_total']):.1f}  "
          f"(σ={_std(retrieval['search_total']):.1f})")
    print(f"  Total Retrieval (ms):          {_avg(retrieval['retrieval_total']):.1f}  "
          f"(σ={_std(retrieval['retrieval_total']):.1f})")

    print("\n--- Generation Phase ---")
    for model_name, data in llm_results.items():
        ttft = data["ttft"]
        total = data["total"]
        print(f"\n  Model: {model_name}")
        print(f"    Time to First Token (ms):   {_avg(ttft):.1f}  "
              f"(σ={_std(ttft):.1f})")
        print(f"    Total Generation Time (ms):  {_avg(total):.1f}  "
              f"(σ={_std(total):.1f})")

    # LaTeX-ready table
    print("\n\n" + "=" * 70)
    print("  LaTeX-Ready Summary (copy-paste)")
    print("=" * 70)
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l r r}")
    print(r"\toprule")
    print(r"Metric & Mean (ms) & Std (ms) \\")
    print(r"\midrule")
    print(
        f"Query Encoding & {_avg(retrieval['embed']):.1f} & {_std(retrieval['embed']):.1f} \\\\"
    )
    print(
        f"Lexical ES Search & {_avg(retrieval['lex_es']):.1f} & {_std(retrieval['lex_es']):.1f} \\\\"
    )
    print(
        f"kNN ES Search & {_avg(retrieval['knn_es']):.1f} & {_std(retrieval['knn_es']):.1f} \\\\"
    )
    print(
        f"RRF Fusion & {_avg(retrieval['rrf']):.1f} & {_std(retrieval['rrf']):.1f} \\\\"
    )
    print(
        f"Total Retrieval & {_avg(retrieval['retrieval_total']):.1f} & {_std(retrieval['retrieval_total']):.1f} \\\\"
    )
    print(r"\midrule")
    for model_name, data in llm_results.items():
        label = model_name.replace("_", r"\_")
        ttft = data["ttft"]
        total = data["total"]
        print(
            f"TTFT ({label}) & {_avg(ttft):.1f} & {_std(ttft):.1f} \\\\"
        )
        print(
            f"Total Gen ({label}) & {_avg(total):.1f} & {_std(total):.1f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Average latency across 20 test queries (hybrid RRF retrieval + Groq LLM generation)}")
    print(r"\label{tab:latency-benchmark}")
    print(r"\end{table}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    retrieval = run_retrieval_benchmark()
    llm_results = run_llm_benchmark(retrieval["prompts"])
    print_report(retrieval, llm_results)
