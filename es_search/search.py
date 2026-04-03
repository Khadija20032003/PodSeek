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
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ES_HOST, ES_INDEX


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
) -> list[dict]:
    """
    Search for chunks matching the query.

    Uses multi_match across text, show_name, and episode_name.
    Optionally filters by category or show_name.
    """
    must_clause = {
        "multi_match": {
            "query": query,
            "fields": ["text^3", "show_name", "episode_name"],
            "type": "best_fields",
        }
    }

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [must_clause],
            }
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

    filters = []
    if category:
        filters.append({"term": {"category": category}})
    if show_name:
        filters.append({"match": {"show_name": show_name}})
    if filters:
        body["query"]["bool"]["filter"] = filters

    response = es.search(index=ES_INDEX, body=body)
    return response["hits"]["hits"]


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

    hits = search(es, args.query, top_k=args.top, category=args.category, show_name=args.show)

    if not hits:
        print("No results found.")
        return

    if args.json:
        print(json.dumps([h["_source"] for h in hits], indent=2))
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