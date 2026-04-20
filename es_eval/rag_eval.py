"""
rag_eval.py — Evaluate PodSeek's RAG pipeline.

Two evaluation modes:
  1. Retrieval metrics (ground truth from dataset.json)
     - Hit@K:         Was any correct chunk in the top-K results?
     - MRR:           Mean reciprocal rank of the first correct hit
     - Precision@K:   Fraction of top-K results that are correct
     - Context Recall: Of all ground-truth chunks, what fraction was retrieved?

  2. RAG quality (RAGAS, Groq as judge)
     - Faithfulness:        Is the answer grounded in retrieved chunks?
     - Answer Relevancy:    Does the answer address the question?
     - Factual Correctness: Is the answer correct vs. the known reference?

Usage:
    python rag_eval.py --dataset dataset.json
    python rag_eval.py --dataset dataset.json --top 10
    python rag_eval.py --dataset dataset.json --skip-ragas
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from elasticsearch import Elasticsearch
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "es_search"))
from config import ES_HOST, ES_INDEX
from search import search, format_time

load_dotenv()


# ---------------------------------------------------------------------------
# Load ground-truth dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list:
    """
    Load dataset.json and flatten into evaluation cases.
    Each case has: question, title, reference_answer,
                   expected_elastic_ids, expected_file_ids,
                   expected_show_names, expected_episode_names,
                   expected_chunk_texts
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for entry in data:
        true_chunks = entry.get("true_chunks", [])
        cases.append({
            "question": entry["question"],
            "title": entry.get("title", ""),
            "reference_answer": entry.get("abstract", ""),
            "expected_elastic_ids": {c["elastic_id"] for c in true_chunks},
            "expected_file_ids": {c["file_id"] for c in true_chunks},
            "expected_show_names": {c["show_name"].strip().lower() for c in true_chunks},
            "expected_episode_names": {c["episode_name"].strip().lower() for c in true_chunks},
            "expected_chunk_texts": [c["text"] for c in true_chunks],
            "n_expected": len(true_chunks),
        })

    print(f"Loaded {len(cases)} evaluation cases from {path}")
    return cases


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def is_correct_hit(hit: dict, case: dict) -> bool:
    """Check if a retrieved chunk belongs to the expected ground-truth set."""
    hit_id = hit.get("_id", "")
    if hit_id in case["expected_elastic_ids"]:
        return True

    src = hit.get("_source", {})
    file_id = src.get("file_id", "").strip()
    if file_id in case["expected_file_ids"]:
        return True

    show = src.get("show_name", "").strip().lower()
    episode = src.get("episode_name", "").strip().lower()
    if show in case["expected_show_names"] and episode in case["expected_episode_names"]:
        return True

    return False


def reciprocal_rank(hits: list, case: dict) -> float:
    for rank, hit in enumerate(hits, 1):
        if is_correct_hit(hit, case):
            return 1.0 / rank
    return 0.0


def hit_at_k(hits: list, case: dict, k: int) -> bool:
    return any(is_correct_hit(h, case) for h in hits[:k])


def precision_at_k(hits: list, case: dict, k: int) -> float:
    correct = sum(1 for h in hits[:k] if is_correct_hit(h, case))
    return correct / k if k > 0 else 0.0


def context_recall(hits: list, case: dict) -> float:
    """Of all ground-truth chunks, what fraction appeared in the retrieved results?"""
    n_expected = case.get("n_expected", 0)
    if n_expected == 0:
        return 0.0
    n_found = sum(1 for h in hits if is_correct_hit(h, case))
    return n_found / n_expected


def run_retrieval_eval(es: Elasticsearch, cases: list, top_k: int) -> list:
    print(f"\n{'='*60}")
    print(f"  RETRIEVAL EVALUATION  ({len(cases)} questions, top_k={top_k})")
    print(f"{'='*60}")

    results = []
    for i, case in enumerate(cases):
        q = case["question"]
        print(f"  [{i+1}/{len(cases)}] {q[:70]}...")

        try:
            result = search(es, q, top_k=top_k)
            hits = result["hits"]
        except Exception as e:
            print(f"    ERROR: {e}")
            hits = []

        rr = reciprocal_rank(hits, case)
        h1 = hit_at_k(hits, case, k=1)
        h3 = hit_at_k(hits, case, k=3)
        hk = hit_at_k(hits, case, k=top_k)
        pk = precision_at_k(hits, case, k=top_k)
        cr = context_recall(hits, case)

        correct_ranks = [
            rank for rank, h in enumerate(hits, 1)
            if is_correct_hit(h, case)
        ]

        results.append({
            **case,
            "hits": hits,
            "rr": rr,
            "hit@1": h1,
            "hit@3": h3,
            f"hit@{top_k}": hk,
            f"p@{top_k}": pk,
            "context_recall": cr,
            "correct_ranks": correct_ranks,
        })

        status = f"rank {correct_ranks[0]}" if correct_ranks else "NOT FOUND"
        print(f"    {status}  |  RR={rr:.3f}  |  P@{top_k}={pk:.3f}  |  Recall={cr:.3f}")

    return results


def print_retrieval_summary(results: list, top_k: int):
    n = len(results)
    if n == 0:
        return

    mrr = sum(r["rr"] for r in results) / n
    h1 = sum(r["hit@1"] for r in results) / n
    h3 = sum(r["hit@3"] for r in results) / n
    hk = sum(r[f"hit@{top_k}"] for r in results) / n
    pk = sum(r[f"p@{top_k}"] for r in results) / n
    cr = sum(r["context_recall"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL RESULTS  (n={n})")
    print(f"{'='*60}")
    print(f"  MRR              {mrr:.4f}")
    print(f"  Hit@1            {h1:.4f}")
    print(f"  Hit@3            {h3:.4f}")
    print(f"  Hit@{top_k:<2}           {hk:.4f}")
    print(f"  Precision@{top_k:<2}     {pk:.4f}")
    print(f"  Context Recall   {cr:.4f}")
    print(f"{'='*60}")

    return {"mrr": mrr, "hit_at_1": h1, "hit_at_3": h3,
            f"hit_at_{top_k}": hk, f"p_at_{top_k}": pk,
            "context_recall": cr}


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------

def generate_answer(llm, question: str, hits: list) -> str:
    context_string = "\n\n".join(
        f"[Chunk {i+1} | Show: {hit['_source'].get('show_name', 'Unknown')} | "
        f"Episode: {hit['_source'].get('episode_name', 'Unknown')} | "
        f"Time: {format_time(hit['_source'].get('start_time', 0))}-"
        f"{format_time(hit['_source'].get('end_time', 0))}]: "
        f"{hit['_source'].get('parent_text') or hit['_source'].get('text', '')}"
        for i, hit in enumerate(hits)
    )

    prompt = f"""You are a podcast search assistant. Answer the user's question 
using ONLY the provided podcast transcripts.

RULES:
1. Every claim in your answer MUST be directly stated in the transcripts. Do not infer, generalize, or add background knowledge.
2. If the transcripts only partially answer the question, answer with only what is supported and say "The available podcasts only cover the following aspects of this topic."
3. If the transcripts do not contain the answer at all, say "I could not find relevant information in the podcast database."
4. Do not add disclaimers, recommendations, or meta-commentary like "I recommend exploring other resources."
5. For each point you make, cite the source in this format: (Source: [show name], [episode name], [timestamp])

User Question: {question}

Podcast Transcripts:
{context_string}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_eval(llm, cases: list, top_k: int, es: Elasticsearch) -> tuple:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.metrics.collections import FactualCorrectness
    from ragas.llms import LangchainLLMWrapper, llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_huggingface import HuggingFaceEmbeddings
    from groq import Groq

    print(f"\n{'='*60}")
    print(f"  RAGAS EVALUATION  ({len(cases)} questions)")
    print(f"{'='*60}")

    rag_results = []
    for i, case in enumerate(cases):
        q = case["question"]
        print(f"  [{i+1}/{len(cases)}] Generating answer: {q[:55]}...")
        try:
            result = search(es, q, top_k=top_k)
            hits = result["hits"]
            contexts = [
                h["_source"].get("parent_text") or h["_source"].get("text", "")
                for h in hits
            ]
            answer = generate_answer(llm, q, hits)
        except Exception as e:
            print(f"    ERROR: {e}")
            answer, contexts = "Error generating answer.", []

        rag_results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "reference": case.get("reference_answer", ""),
        })

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # LangchainLLMWrapper for Faithfulness + ResponseRelevancy
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embs = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # llm_factory with Groq client for FactualCorrectness
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_client = Groq(api_key=groq_api_key)
    factual_llm = llm_factory(
        "llama-3.1-8b-instant",
        provider="groq",
        client=groq_client,
    )

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embs)
    relevancy_metric.strictness = 1
    factual_metric = FactualCorrectness(llm=factual_llm)

    run_config = RunConfig(max_workers=1, timeout=120, max_retries=3)

    print("\n  Running RAGAS evaluation (using Groq as judge)...")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[faithfulness_metric, relevancy_metric, factual_metric],
        run_config=run_config,
    )

    df = result.to_pandas()
    print(f"\n  Available RAGAS columns: {list(df.columns)}")

    def _safe_mean(col_name):
        if col_name in df.columns:
            return round(float(df[col_name].mean()), 4)
        return float("nan")

    scores = {
        "faithfulness": _safe_mean("faithfulness"),
        "answer_relevancy": _safe_mean("answer_relevancy"),
        "factual_correctness": _safe_mean("factual_correctness"),
    }

    print(f"\n{'='*60}")
    print(f"  RAGAS RESULTS")
    print(f"{'='*60}")
    print(f"  Faithfulness:          {scores['faithfulness']}")
    print(f"    -> Score > 0.8 = good, the LLM is not hallucinating")
    print()
    print(f"  Answer Relevancy:      {scores['answer_relevancy']}")
    print(f"    -> Score > 0.75 = good, answers are on-topic")
    print()
    print(f"  Factual Correctness:   {scores['factual_correctness']}")
    print(f"    -> Score > 0.7 = good, answers match the reference")
    print(f"{'='*60}")

    return scores, rag_results, df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate PodSeek RAG pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset.json with ground-truth chunks")
    parser.add_argument("--top", type=int, default=5,
                        help="Top-K chunks to retrieve per question (default: 5)")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Only run retrieval metrics, skip RAGAS (faster, no API key needed)")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output results JSON path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")

    es = Elasticsearch(ES_HOST, request_timeout=120)
    if not es.ping():
        sys.exit("Cannot connect to Elasticsearch. Is Docker running?")
    print(f"Connected to Elasticsearch at {ES_HOST}")

    cases = load_dataset(dataset_path)
    if not cases:
        sys.exit("No evaluation cases found in dataset.")

    # --- Retrieval evaluation ---
    retrieval_results = run_retrieval_eval(es, cases, top_k=args.top)
    retrieval_scores = print_retrieval_summary(retrieval_results, top_k=args.top)

    output = {
        "config": {"top_k": args.top, "n_cases": len(cases)},
        "retrieval": retrieval_scores,
        "per_question": [
            {
                "title": r.get("title", ""),
                "question": r["question"],
                "rr": r["rr"],
                "hit@1": r["hit@1"],
                "hit@3": r["hit@3"],
                "context_recall": r["context_recall"],
                "correct_ranks": r["correct_ranks"],
            }
            for r in retrieval_results
        ],
        "ragas": None,
    }

    # --- RAGAS evaluation ---
    if not args.skip_ragas:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("\nWARNING: GROQ_API_KEY not found -- skipping RAGAS.")
        else:
            llm = ChatGroq(
                temperature=0,
                groq_api_key=api_key.strip(),
                model_name="llama-3.1-8b-instant",
            )
            ragas_scores, rag_results, ragas_df = run_ragas_eval(
                llm, cases, top_k=args.top, es=es,
            )
            output["ragas"] = ragas_scores

            for i, pq in enumerate(output["per_question"]):
                try:
                    pq["faithfulness"] = float(ragas_df.iloc[i].get("faithfulness", float("nan")))
                    pq["answer_relevancy"] = float(ragas_df.iloc[i].get("answer_relevancy", float("nan")))
                    if "factual_correctness" in ragas_df.columns:
                        pq["factual_correctness"] = float(ragas_df.iloc[i]["factual_correctness"])
                    pq["generated_answer"] = rag_results[i]["answer"]
                except Exception:
                    pass

    # --- Save results ---
    out_path = Path(__file__).parent / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    if retrieval_scores:
        print(f"  MRR:               {retrieval_scores['mrr']:.4f}")
        print(f"  Hit@1:             {retrieval_scores['hit_at_1']:.4f}")
        print(f"  Hit@3:             {retrieval_scores['hit_at_3']:.4f}")
        print(f"  Context Recall:    {retrieval_scores['context_recall']:.4f}")
    if output["ragas"]:
        print(f"  Faithfulness:      {output['ragas']['faithfulness']:.4f}")
        print(f"  Ans. Relevancy:    {output['ragas']['answer_relevancy']:.4f}")
        print(f"  Factual Correct.:  {output['ragas'].get('factual_correctness', float('nan')):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()