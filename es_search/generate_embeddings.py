"""
generate_embeddings.py
----------------------
Reads elastic_ready.jsonl and generates dense vector embeddings for each chunk
using sentence-transformers. Saves the result as elastic_ready_with_embeddings.jsonl.

The embedding model is all-MiniLM-L6-v2 (384 dimensions), which is fast and
produces good quality embeddings for semantic search.

Usage:
    python generate_embeddings.py
    python generate_embeddings.py --batch-size 256
    python generate_embeddings.py --model all-mpnet-base-v2  # 768 dims, slower but better quality
"""

import argparse
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "cleaned_output" / "elastic_ready.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "cleaned_output" / "elastic_ready_with_embeddings.jsonl"

DEFAULT_MODEL = "all-MiniLM-L6-v2"  


def count_lines(filepath: Path) -> int:
    """Count total lines for progress tracking."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def generate_embeddings(
    input_path: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

    print(f"Counting chunks in {input_path}...")
    total_lines = count_lines(input_path)
    print(f"Total chunks to embed: {total_lines}")

    # Check if a partial output already exists (for resuming)
    already_done = 0
    if output_path.exists():
        already_done = count_lines(output_path)
        if already_done >= total_lines:
            print(f"Already complete! {already_done} embeddings found.")
            return
        print(f"Resuming from chunk {already_done} (skipping already embedded)")

    start_time = time.time()
    processed = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "a" if already_done > 0 else "w", encoding="utf-8") as fout:

            # Skip already processed lines if resuming
            for _ in range(already_done):
                fin.readline()

            batch_texts = []
            batch_chunks = []

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                chunk = json.loads(line)
                batch_texts.append(chunk.get("text", ""))
                batch_chunks.append(chunk)

                if len(batch_texts) >= batch_size:
                    # Encode the batch
                    embeddings = model.encode(batch_texts, show_progress_bar=False)

                    for ch, emb in zip(batch_chunks, embeddings):
                        ch["embedding"] = emb.tolist()
                        fout.write(json.dumps(ch) + "\n")

                    processed += len(batch_texts)
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total_lines - already_done - processed) / rate / 60

                    print(
                        f"  Embedded {already_done + processed}/{total_lines} "
                        f"({rate:.0f} chunks/sec) — "
                        f"~{remaining:.1f} min remaining"
                    )

                    batch_texts.clear()
                    batch_chunks.clear()

            # Process remaining chunks in the last batch
            if batch_texts:
                embeddings = model.encode(batch_texts, show_progress_bar=False)
                for ch, emb in zip(batch_chunks, embeddings):
                    ch["embedding"] = emb.tolist()
                    fout.write(json.dumps(ch) + "\n")
                processed += len(batch_texts)

    elapsed = round((time.time() - start_time) / 60, 2)
    print(f"\nDONE! Embedded {processed} chunks in {elapsed} minutes.")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for podcast chunks")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to elastic_ready.jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Path to output file with embeddings")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Sentence transformer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for encoding (default: 128)")
    args = parser.parse_args()

    generate_embeddings(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()