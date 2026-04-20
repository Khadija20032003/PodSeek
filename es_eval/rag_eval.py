"""
rag_eval.py — Evaluate PodSeek's RAG pipeline with RAGAS.
Evaluates faithfulness and answer relevancy — NO ground truth dataset needed.
Uses Groq (free) as the RAGAS judge LLM instead of OpenAI.

Metrics:
  - Faithfulness: Is the LLM answer grounded in the retrieved chunks? (no hallucination)
  - Answer Relevancy: Does the answer actually address the question?

Usage:
    python rag_eval.py
    python rag_eval.py --top 10
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
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "es_search"))
from config import ES_HOST, ES_INDEX
from search import search, format_time

load_dotenv()

# Test Questions 
EVAL_QUESTIONS = [
    "What are some common fitness myths?",
    "How does meditation help with stress?",
    "What is cryptocurrency and how does it work?",
    "What are the benefits of intermittent fasting?",
    "How does climate change affect the environment?",
    "What is artificial intelligence?",
    "How do you start a podcast?",
    "What is the importance of mental health?",
    "How does social media affect teenagers?",
    "What are the basics of investing in the stock market?",
]

# RAG Pipeline

def generate_answer(llm, question: str, hits: list) -> str:
    """Send retrieved contexts + question to the LLM and get an answer."""
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


def run_rag_pipeline(es, llm, questions, top_k=5):
    results = []
    for i, question in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] Processing: {question[:60]}...")
        try:
            result = search(es, question, top_k=top_k)
            hits = result["hits"]
            contexts = [
                hit["_source"].get("parent_text") or hit["_source"].get("text", "")
                for hit in hits
            ]
            answer = generate_answer(llm, question, hits)
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
            })
        except Exception as e:
            print(f"    ERROR on question {i+1}: {e}")
            results.append({
                "question": question,
                "answer": "Error generating answer.",
                "contexts": [],
            })
    return results


# Evaluation

def build_ragas_dataset(rag_results: list[dict]) -> EvaluationDataset:
    """Convert RAG pipeline output to RAGAS EvaluationDataset."""
    samples = []
    for item in rag_results:
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item["contexts"],
            )
        )
    return EvaluationDataset(samples=samples)


def run_evaluation(
    es: Elasticsearch,
    llm: ChatGroq,
    questions: list,
    top_k: int = 5,
) -> dict:
    """Run RAGAS evaluation — faithfulness and answer relevancy only."""
    print(f"\n{'='*60}")
    print(f"  Running RAG pipeline on {len(questions)} questions...")
    print(f"{'='*60}")

    # Run RAG pipeline
    rag_results = run_rag_pipeline(es, llm, questions, top_k=top_k)

    # Build RAGAS dataset 
    eval_dataset = build_ragas_dataset(rag_results)

    # Set up Groq as the RAGAS judge
    print("\n  Running RAGAS evaluation (using Groq as judge)...")

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Configure metrics
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    # strictness=1 prevents the 'n must be at most 1' error on Groq
    relevancy_metric.strictness = 1

    # RunConfig: serialize calls to avoid Groq rate limits + longer timeout
    run_config = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=3,
    )

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness_metric, relevancy_metric],
        run_config=run_config,
    )

    # Extract scores from EvaluationResult via pandas
    try:
        result_df = result.to_pandas()
        result_dict = result_df.mean(numeric_only=True).to_dict()
    except Exception:
        # Fallback: access columns individually
        result_dict = {}
        for col in ["faithfulness", "answer_relevancy"]:
            try:
                result_dict[col] = result.to_pandas()[col].mean()
            except Exception:
                result_dict[col] = float("nan")

    scores = {
        "faithfulness": round(result_dict.get("faithfulness", float("nan")), 4),
        "answer_relevancy": round(result_dict.get("answer_relevancy", float("nan")), 4),
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"  RAGAS EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Faithfulness:       {scores['faithfulness']}")
    print(f"    -> Is the answer grounded in the retrieved chunks?")
    print(f"    -> Score > 0.8 = good, the LLM is not hallucinating")
    print()
    print(f"  Answer Relevancy:   {scores['answer_relevancy']}")
    print(f"    -> Does the answer actually address the question?")
    print(f"    -> Score > 0.75 = good, answers are on-topic")
    print(f"{'='*60}\n")

    # Save summary results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  Results saved to: {output_path}")

    # Save per-question details (with individual scores)
    details_path = Path(__file__).parent / "eval_details.json"
    details = []
    for i, item in enumerate(rag_results):
        detail = {
            "question": item["question"],
            "answer": item["answer"],
            "num_contexts": len(item["contexts"]),
            "context_preview": item["contexts"][0][:300] if item["contexts"] else "",
        }
        # Add per-question scores from the dataframe
        try:
            detail["faithfulness"] = float(result_df.iloc[i].get("faithfulness", float("nan")))
            detail["answer_relevancy"] = float(result_df.iloc[i].get("answer_relevancy", float("nan")))
        except Exception:
            detail["faithfulness"] = None
            detail["answer_relevancy"] = None
        details.append(detail)

    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"  Per-question details saved to: {details_path}")

    return scores


# CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate PodSeek RAG pipeline with RAGAS")
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    args = parser.parse_args()

    if len(EVAL_QUESTIONS) < 3:
        print("WARNING: Add more questions to EVAL_QUESTIONS for meaningful results.\n")

    # Connect to Elasticsearch
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        sys.exit("Cannot connect to Elasticsearch. Is Docker running?")
    print(f"Connected to Elasticsearch at {ES_HOST}")

    # Connect to Groq LLM
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not found in environment. Check your .env file.")

    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_key.strip(),
        model_name="llama-3.1-8b-instant",
    )

    # Run evaluation
    run_evaluation(es, llm, EVAL_QUESTIONS, top_k=args.top)


if __name__ == "__main__":
    main()