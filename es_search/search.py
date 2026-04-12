"""
search.py — CLI to search podcast chunks in Elasticsearch.

Usage:
    python search.py "large language models"
    python search.py "carbonara recipe" --top 10
    python search.py "machine learning" --category Technology
    python search.py "climate change" --show "Science Weekly"
"""

import sys
import argparse
import json
import time
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ES_HOST, ES_INDEX, EMBEDDING_MODEL_NAME


_QUERY_EMBEDDER: SentenceTransformer | None = None


def _get_query_embedder() -> SentenceTransformer:
    global _QUERY_EMBEDDER
    if _QUERY_EMBEDDER is None:
        _QUERY_EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _QUERY_EMBEDDER


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def search(
    es: Elasticsearch,
    query: str,
    top_k: int = 5,
    category: str = None,
    show_name: str = None,
    embedder: SentenceTransformer | None = None,
) -> dict:
    """
    Search for chunks matching the query.
    
    Upgrades:
    - BM25F Field Boosting (prioritizes category and titles over raw text)
    - Fuzziness (auto-corrects minor typos)
    - Phrase Suggester (returns "did you mean?" spelling corrections)
    """
    must_clause = {
        "multi_match": {
            "query": query,
            # BM25F Boosting: ^5 means 5x importance. 
            "fields": [
                "category^5",       # Highest priority: Exact topic match
                "show_name^3",      # High priority: Show name match
                "episode_name^3",   # High priority: Episode title match
                "text^1",           # Standard priority: Child chunk text
                "parent_text^1"     # Standard priority: Surrounding context
            ],
            "type": "best_fields",
            "fuzziness": "AUTO"     # Typo tolerance
        }
    }

    t_total0 = time.perf_counter()

    embedder = embedder or _get_query_embedder()
    t_embed0 = time.perf_counter()
    query_vector = embedder.encode([query], show_progress_bar=False)[0].tolist()
    t_embed_ms = (time.perf_counter() - t_embed0) * 1000.0

    window_size = 100
    rank_constant = 60

    def _rrf_fuse(lex_hits: list, vec_hits: list) -> list:
        # Reciprocal Rank Fusion:
        # score(d) = sum_{systems} 1 / (rank_constant + rank(d, system))
        # Using 1-indexed ranks.
        fused: dict[str, dict] = {}

        def _accumulate(hits: list, source_name: str):
            for rank, hit in enumerate(hits[:window_size], 1):
                doc_id = hit.get("_id")
                if not doc_id:
                    continue

                entry = fused.get(doc_id)
                if entry is None:
                    entry = {
                        "_id": doc_id,
                        "_source": hit.get("_source", {}),
                        "highlight": hit.get("highlight"),
                        "_rrf": 0.0,
                        "_seen": set(),
                    }
                    fused[doc_id] = entry

                entry["_rrf"] += 1.0 / (rank_constant + rank)
                entry["_seen"].add(source_name)

                # Prefer lexical highlight/source when available
                if source_name == "lex":
                    if hit.get("highlight") is not None:
                        entry["highlight"] = hit.get("highlight")
                    if hit.get("_source") is not None:
                        entry["_source"] = hit.get("_source")

        _accumulate(lex_hits, "lex")
        _accumulate(vec_hits, "vec")

        fused_list = list(fused.values())
        fused_list.sort(key=lambda x: x["_rrf"], reverse=True)
        results = []
        for item in fused_list[:top_k]:
            out = {
                "_id": item["_id"],
                "_score": item["_rrf"],
                "_source": item.get("_source", {}),
            }
            if item.get("highlight") is not None:
                out["highlight"] = item["highlight"]
            results.append(out)
        return results

    lexical_body = {
        "size": window_size,
        "query": {
            "bool": {
                "must": [must_clause],
            }
        },
        # Spelling correction block
        "suggest": {
            "text": query,
            "spell_check": {
                "phrase": {
                    "field": "text",
                    "size": 1,
                    "direct_generator": [
                        {
                            "field": "text",
                            "suggest_mode": "popular",
                        }
                    ],
                }
            },
        },
        "highlight": {
            "fields": {
                "text": {
                    "fragment_size": 200,
                    "number_of_fragments": 2,
                    "pre_tags": ["**"],
                    "post_tags": ["**"],
                }
            }
        },
    }

    knn_body = {
        "size": window_size,
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": 50,
            "num_candidates": 100,
        },
    }

    filters = []
    if category:
        filters.append({"term": {"category.keyword": category}})
    if show_name:
        filters.append({"term": {"show_name.keyword": show_name}})
    if filters:
        lexical_body["query"]["bool"]["filter"] = filters
        knn_body["knn"]["filter"] = filters

    t_lex0 = time.perf_counter()
    lexical_resp = es.search(index=ES_INDEX, body=lexical_body)
    t_lex_ms = (time.perf_counter() - t_lex0) * 1000.0

    knn_resp = None
    try:
        t_knn0 = time.perf_counter()
        knn_resp = es.search(index=ES_INDEX, body=knn_body)
        t_knn_ms = (time.perf_counter() - t_knn0) * 1000.0
    except BadRequestError:
        knn_resp = {"hits": {"hits": []}}
        t_knn_ms = 0.0

    # Extract spelling suggestion if Elasticsearch found a better alternative
    suggestion = None
    suggest_options = lexical_resp.get("suggest", {}).get("spell_check", [])
    if suggest_options and suggest_options[0].get("options"):
        suggestion = suggest_options[0]["options"][0]["text"]

    t_rrf0 = time.perf_counter()
    fused_hits = _rrf_fuse(
        lexical_resp.get("hits", {}).get("hits", []),
        (knn_resp or {}).get("hits", {}).get("hits", []),
    )
    t_rrf_ms = (time.perf_counter() - t_rrf0) * 1000.0

    t_total_ms = (time.perf_counter() - t_total0) * 1000.0

    return {
        "hits": fused_hits,
        "suggestion": suggestion,
        "timings": {
            "embed_ms": t_embed_ms,
            "lexical_es_ms": t_lex_ms,
            "knn_es_ms": t_knn_ms,
            "rrf_ms": t_rrf_ms,
            "total_ms": t_total_ms,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Search podcast chunks")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--show", type=str, default=None, help="Filter by show name")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST)
    if not es.ping():
        sys.exit(f"Cannot connect to Elasticsearch at {ES_HOST}")

    # Run the upgraded search
    result = search(es, args.query, top_k=args.top, category=args.category, show_name=args.show)
    hits = result["hits"]
    suggestion = result["suggestion"]

    # Print JSON and exit early if requested
    if args.json:
        print(json.dumps([h["_source"] for h in hits], indent=2))
        return

    # Print Spelling Suggestion UI if one exists and isn't just the exact query
    if suggestion and suggestion.lower() != args.query.lower():
        print(f"\n💡 Did you mean: {suggestion}?")

    if not hits:
        print("No results found.")
        return

    print(f"\n{'='*70}")
    print(f"  Search results for: \"{args.query}\"  ({len(hits)} hits)")
    print(f"{'='*70}\n")

    for rank, hit in enumerate(hits, 1):
        src = hit["_source"]
        score = hit["_score"]
        highlights = hit.get("highlight", {}).get("text", [])

        start = format_time(src.get("start_time", 0))
        end = format_time(src.get("end_time", 0))

        print(f"  #{rank}  [score: {score:.2f}]")
        print(f"  Show:      {src.get('show_name', 'Unknown')}")
        print(f"  Episode:   {src.get('episode_name', 'Unknown')}")
        print(f"  Publisher: {src.get('publisher', 'Unknown')}  |  Category: {src.get('category', 'N/A')}")
        print(f"  Time:      {start} → {end}")
        print()
        if highlights:
            for h in highlights:
                print(f"    ... {h} ...")
        else:
            text_preview = src.get("text", "")[:300]
            print(f"    {text_preview}...")
        print(f"\n  {'-'*66}\n")


if __name__ == "__main__":
    main()