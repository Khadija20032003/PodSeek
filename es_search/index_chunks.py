"""
index_chunks.py — Bulk-indexes podcast chunks into Elasticsearch.

Usage:
    python index_chunks.py --recreate
"""

import sys
import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ELASTIC_READY_FILE, ES_HOST, ES_INDEX

DEFAULT_INPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "final_output"
    / "elastic_ready_with_embeddings.jsonl"
)

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "file_id":       {"type": "keyword"},
            "chunk_id":      {"type": "keyword"},
            "segment_id":    {"type": "integer"},
            "text":          {"type": "text", "analyzer": "standard"},
            "start_time":    {"type": "float"},
            "end_time":      {"type": "float"},

            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },

            "parent_id":         {"type": "keyword"},
            "parent_ids":        {"type": "keyword"},
            "parent_text":       {"type": "text", "analyzer": "standard"},
            "parent_texts":      {"type": "text", "analyzer": "standard"},
            "parent_start_time": {"type": "float"},
            "parent_end_time":   {"type": "float"},

            "show_name":     {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "episode_name":  {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "publisher":     {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "category":      {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "rss_link":      {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        }
    }
}


def create_index(es: Elasticsearch, recreate: bool = False):
    """Create the podcast_chunks index (optionally dropping it first)."""
    if es.indices.exists(index=ES_INDEX):
        if recreate:
            print(f"Deleting existing index '{ES_INDEX}' ...")
            es.indices.delete(index=ES_INDEX)
        else:
            print(f"Index '{ES_INDEX}' already exists. Use --recreate to reset.")
            return

    es.indices.create(index=ES_INDEX, body=INDEX_SETTINGS)
    print(f"Created index '{ES_INDEX}'")


def bulk_index(es: Elasticsearch, input_path: Path):
    """Read elastic_ready.jsonl line by line and bulk-index into Elasticsearch."""

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    def _generate_actions():
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                yield {
                    "_index": ES_INDEX,
                    "_id": chunk.get("elastic_id", ""),
                    "_source": {
                        "file_id":      chunk.get("file_id", ""),
                        "chunk_id":     chunk.get("chunk_id", ""),
                        "segment_id":   chunk.get("segment_id", 0),
                        "text":         chunk.get("text", ""),
                        "start_time":   chunk.get("start_time", 0.0),
                        "end_time":     chunk.get("end_time", 0.0),

                        "embedding":         chunk.get("embedding"),

                        "parent_id":         chunk.get("parent_id", ""),
                        "parent_ids":        chunk.get("parent_ids", []),
                        "parent_text":       chunk.get("parent_text", ""),
                        "parent_texts":      chunk.get("parent_texts", []),
                        "parent_start_time": chunk.get("parent_start_time", 0.0),
                        "parent_end_time":   chunk.get("parent_end_time", 0.0),

                        "show_name":    chunk.get("show_name", ""),
                        "episode_name": chunk.get("episode_name", ""),
                        "publisher":    chunk.get("publisher", ""),
                        "category":     chunk.get("category", ""),
                        "rss_link":     chunk.get("rss_link", ""),
                    }
                }

    print(f"Indexing from: {input_path}")
    success, errors = helpers.bulk(es, _generate_actions(), raise_on_error=False, chunk_size=1000)
    print(f"Indexed {success} chunks ({len(errors)} errors)")
    if errors:
        for err in errors[:5]:
            print(f"  {err}")


def main():
    parser = argparse.ArgumentParser(description="Index podcast chunks into Elasticsearch")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FILE),
                        help="Path to elastic_ready.jsonl")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the index")
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST, timeout=120, max_retries=3, retry_on_timeout=True)
    if not es.ping():
        sys.exit(f"Cannot connect to Elasticsearch at {ES_HOST}")
    print(f"Connected to Elasticsearch at {ES_HOST}")

    create_index(es, recreate=args.recreate)
    bulk_index(es, Path(args.input))


if __name__ == "__main__":
    main()