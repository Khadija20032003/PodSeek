"""
EmbeddingGenerator.py — Generates dense vector embeddings for podcast chunks.

Reads elastic_ready.jsonl and uses sentence-transformers to create embeddings.
Saves the result as elastic_ready_with_embeddings.jsonl.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_BATCH_SIZE, 
    EMBEDDING_INPUT_FILE, 
    EMBEDDING_OUTPUT_FILE
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class EmbeddingGenerator:
    """
    Handles the loading of embedding models and the batch processing of 
    text chunks into dense vectors.
    """

    def __init__(
        self,
        input_file: Path = EMBEDDING_INPUT_FILE,
        output_file: Path = EMBEDDING_OUTPUT_FILE,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.batch_size = batch_size
        
        logging.info(f"Initializing model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Ensures the output directory exists."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def _count_lines(self, filepath: Path) -> int:
        """Count total lines for progress tracking."""
        if not filepath.exists():
            return 0
        with open(filepath, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def process(self) -> None:
        """
        Main execution loop. Handles resuming, batching, and writing to disk.
        """
        if not self.input_file.exists():
            logging.error(f"Input file not found: {self.input_file}")
            return

        total_chunks = self._count_lines(self.input_file)
        already_done = self._count_lines(self.output_file) if self.output_file.exists() else 0

        if already_done >= total_chunks and total_chunks > 0:
            logging.info(f"Already complete! {already_done} embeddings found.")
            return

        if already_done > 0:
            logging.info(f"Resuming from chunk {already_done}...")

        start_time = time.time()
        processed_this_session = 0

        with open(self.input_file, "r", encoding="utf-8") as fin, \
             open(self.output_file, "a" if already_done > 0 else "w", encoding="utf-8") as fout:

            # Skip lines already processed
            for _ in range(already_done):
                fin.readline()

            batch_texts = []
            batch_chunks = []

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                    # Embed only the child chunk's searchable text.
                    # Parent text is carried for retrieval/LLM context, not for embedding.
                    batch_texts.append(str(chunk.get("text", "")))
                    batch_chunks.append(chunk)
                except json.JSONDecodeError:
                    continue

                if len(batch_texts) >= self.batch_size:
                    self._process_batch(batch_texts, batch_chunks, fout)
                    
                    processed_this_session += len(batch_texts)
                    self._log_progress(processed_this_session, already_done, total_chunks, start_time)
                    
                    batch_texts.clear()
                    batch_chunks.clear()

            # Final partial batch
            if batch_texts:
                self._process_batch(batch_texts, batch_chunks, fout)
                processed_this_session += len(batch_texts)

        elapsed = round((time.time() - start_time) / 60, 2)
        logging.info(f"Finished! Processed {processed_this_session} chunks in {elapsed} mins.")
        logging.info(f"Output saved to: {self.output_file}")

    def _process_batch(self, texts: List[str], chunks: List[Dict], fout) -> None:
        """Encodes a batch of text and writes to the output file."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            fout.write(json.dumps(chunk) + "\n")

    def _log_progress(self, current: int, skipped: int, total: int, start_time: float) -> None:
        """Calculates and prints throughput and ETA."""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - skipped - current) / rate / 60 if rate > 0 else 0
        
        logging.info(
            f"Progress: {skipped + current}/{total} "
            f"({rate:.1f} chunks/sec) — ~{remaining:.1f} min left"
        )


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.process()