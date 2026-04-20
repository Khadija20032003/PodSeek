"""
embed_ground_truth.py — Add embeddings to ground truth dataset.

Loads dataset.json, embeds the "text" field in each true_chunk,
and saves the result as dataset_with_embedding.json.
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# File paths
GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ground_truth"
DATA_FILE = GROUND_TRUTH_DIR / "dataset.json"
OUTPUT_FILE = GROUND_TRUTH_DIR / "dataset_with_embedding.json"


class GroundTruthEmbedder:
    """
    Loads ground truth dataset and adds embeddings to true_chunks.
    """

    def __init__(
        self,
        input_file: Path = DATA_FILE,
        output_file: Path = OUTPUT_FILE,
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
        logging.info(f"Embedding dimension: {self.embedding_dim}")

    def process(self) -> None:
        """
        Load dataset, embed chunks, and save result.
        """
        if not self.input_file.exists():
            logging.error(f"Input file not found: {self.input_file}")
            return

        logging.info(f"Loading dataset from: {self.input_file}")
        with open(self.input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            logging.error("Dataset must be a list of items")
            return

        logging.info(f"Loaded {len(dataset)} items from dataset")

        # Count total chunks to embed
        total_chunks = sum(len(item.get("true_chunks", [])) for item in dataset)
        logging.info(f"Total chunks to embed: {total_chunks}")

        start_time = time.time()
        processed_count = 0

        # Process each item
        for item_idx, item in enumerate(dataset):
            true_chunks = item.get("true_chunks", [])

            if not true_chunks:
                continue

            # Extract texts to embed
            batch_texts = []
            batch_indices = []

            for chunk_idx, chunk in enumerate(true_chunks):
                text = chunk.get("text", "")
                if text:
                    batch_texts.append(text)
                    batch_indices.append(chunk_idx)

            if not batch_texts:
                continue

            # Process batch of texts
            logging.info(
                f"Item {item_idx + 1}/{len(dataset)}: "
                f"Embedding {len(batch_texts)} chunks"
            )
            embeddings = self.model.encode(batch_texts, show_progress_bar=False)

            # Add embeddings back to chunks
            for emb, chunk_idx in zip(embeddings, batch_indices):
                true_chunks[chunk_idx]["embedding"] = emb.tolist()

            processed_count += len(batch_texts)
            self._log_progress(processed_count, total_chunks, start_time)

        # Save result
        logging.info(f"Saving result to: {self.output_file}")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        elapsed = round((time.time() - start_time) / 60, 2)
        logging.info(f"Finished! Processed {processed_count} chunks in {elapsed} mins.")
        logging.info(f"Output saved to: {self.output_file}")

    def _log_progress(self, current: int, total: int, start_time: float) -> None:
        """Calculates and prints throughput and ETA."""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate / 60 if rate > 0 else 0

        logging.info(
            f"Progress: {current}/{total} "
            f"({rate:.1f} chunks/sec) — ~{remaining:.1f} min left"
        )


if __name__ == "__main__":
    embedder = GroundTruthEmbedder()
    embedder.process()
