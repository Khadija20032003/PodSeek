# """
# build_elastic_index.py — Merges chunks + TSV metadata + enriched RSS metadata
# into a single flat JSONL file ready for Elasticsearch indexing.
# """

# import sys
# import json
# import time
# from pathlib import Path

# import pandas as pd

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from config import CHUNKED_DIR, ENRICHED_META_DIR, ELASTIC_READY_FILE, TSV_FILE


# def build_elasticsearch_dataset():
#     print("Loading TSV metadata into memory...")
#     start_time = time.time()

#     try:
#         df = pd.read_csv(TSV_FILE, sep="\t")
#         df.fillna("", inplace=True)
#     except FileNotFoundError:
#         print(f"ERROR: Could not find TSV file at {TSV_FILE}")
#         return

#     tsv_lookup = {}
#     for _, row in df.iterrows():
#         prefix = str(row["episode_filename_prefix"])
#         tsv_lookup[prefix] = {
#             "show_name": str(row["show_name"]),
#             "episode_name": str(row["episode_name"]),
#             "publisher": str(row["publisher"]),
#             "rss_link": str(row["rss_link"]),
#         }

#     print("Denormalizing chunks with metadata...")
#     ELASTIC_READY_FILE.parent.mkdir(parents=True, exist_ok=True)

#     total_chunks = 0
#     missing_enriched_count = 0

#     with open(ELASTIC_READY_FILE, "w", encoding="utf-8") as outfile:

#         for chunk_file in CHUNKED_DIR.glob("*.json"):
#             with open(chunk_file, "r", encoding="utf-8") as cf:
#                 try:
#                     chunk_data = json.load(cf)
#                 except json.JSONDecodeError:
#                     continue

#             file_id = chunk_data.get("file_id", "")
#             segments = chunk_data.get("segments", [])

#             if not file_id or not segments:
#                 continue

#             base_meta = tsv_lookup.get(file_id, {})

#             category = "Uncategorized"
#             meta_file = ENRICHED_META_DIR / f"{file_id}.json"
#             if meta_file.exists():
#                 with open(meta_file, "r", encoding="utf-8") as mf:
#                     enriched_dict = json.load(mf)
#                     category = enriched_dict.get("show_category", "Uncategorized")
#                     if not category:
#                         category = "Uncategorized"
#             else:
#                 missing_enriched_count += 1

#             for segment in segments:
#                 elastic_doc = {
#                     "elastic_id": f"{file_id}_{segment.get('segment_id')}",
#                     "file_id": file_id,
#                     "segment_id": segment.get("segment_id"),
#                     "text": segment.get("text", ""),
#                     "start_time": segment.get("start"),
#                     "end_time": segment.get("end"),
#                     "show_name": base_meta.get("show_name", "Unknown Show"),
#                     "episode_name": base_meta.get("episode_name", "Unknown Episode"),
#                     "publisher": base_meta.get("publisher", "Unknown Publisher"),
#                     "category": category,
#                     "rss_link": base_meta.get("rss_link", ""),
#                 }

#                 outfile.write(json.dumps(elastic_doc) + "\n")
#                 total_chunks += 1

#     elapsed = round(time.time() - start_time, 2)
#     print("\n========================================")
#     print(f"DATASET BUILD COMPLETE in {elapsed} seconds.")
#     print(f"Total Elasticsearch-ready chunks generated: {total_chunks}")
#     print(f"Note: {missing_enriched_count} episodes didn't have scraped categories.")
#     print(f"Final index file ready at: {ELASTIC_READY_FILE}")
#     print("========================================")


# if __name__ == "__main__":
#     build_elasticsearch_dataset()

"""
build_elastic_index.py — Merges chunks + TSV metadata + enriched RSS metadata 
into a single flat JSONL file ready for Elasticsearch indexing.
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Import config from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKED_DIR, ENRICHED_META_DIR, ELASTIC_READY_FILE, TSV_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ElasticDatasetBuilder:
    """
    Handles the denormalization of podcast segments by merging transcript chunks 
    with base TSV metadata and enriched RSS category data.
    """

    def __init__(self, tsv_path: Path, chunks_dir: Path, enriched_dir: Path, output_file: Path):
        self.tsv_path = tsv_path
        self.chunks_dir = chunks_dir
        self.enriched_dir = enriched_dir
        self.output_file = output_file
        
        # Internal counters
        self.total_chunks = 0
        self.missing_enriched_count = 0

    def _load_tsv_lookup(self) -> Dict[str, Dict[str, str]]:
        """Loads the TSV metadata into a dictionary for O(1) lookups."""
        if not self.tsv_path.exists():
            logging.error(f"TSV file not found: {self.tsv_path}")
            return {}

        logging.info("Loading TSV metadata into memory...")
        try:
            df = pd.read_csv(self.tsv_path, sep="\t")
            df = df.fillna("")
            
            # Create a lookup dict keyed by the filename prefix
            return {
                str(row["episode_filename_prefix"]): {
                    "show_name": str(row["show_name"]),
                    "episode_name": str(row["episode_name"]),
                    "publisher": str(row["publisher"]),
                    "rss_link": str(row["rss_link"]),
                }
                for _, row in df.iterrows()
            }
        except Exception as e:
            logging.error(f"Failed to parse TSV file: {e}")
            return {}

    def _get_category(self, file_id: str) -> str:
        """Attempts to retrieve the show category from enriched metadata files."""
        meta_file = self.enriched_dir / f"{file_id}.json"
        
        if not meta_file.exists():
            self.missing_enriched_count += 1
            return "Uncategorized"

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                enriched_data = json.load(f)
                category = enriched_data.get("show_category")
                return category if category else "Uncategorized"
        except (json.JSONDecodeError, OSError):
            return "Uncategorized"

    def build(self) -> None:
        """Executes the merge process and writes the Elasticsearch-ready JSONL file."""
        tsv_lookup = self._load_tsv_lookup()
        if not tsv_lookup:
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Denormalizing chunks from {self.chunks_dir}...")
        
        start_time = time.time()

        with open(self.output_file, "w", encoding="utf-8") as outfile:
            for chunk_file in self.chunks_dir.glob("*.json"):
                try:
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    logging.warning(f"Skipping malformed chunk file: {chunk_file.name}")
                    continue

                file_id = data.get("file_id", "").replace(".json", "")
                segments = data.get("segments", [])

                if not file_id or not segments:
                    continue

                # Retrieve metadata
                base_meta = tsv_lookup.get(file_id, {})
                category = self._get_category(file_id)

                for segment in segments:
                    # Construct flat Elasticsearch document
                    elastic_doc = {
                        "elastic_id": f"{file_id}_{segment.get('segment_id')}",
                        "file_id": file_id,
                        "segment_id": segment.get("segment_id"),
                        "text": segment.get("text", "").strip(),
                        "start_time": segment.get("start"),
                        "end_time": segment.get("end"),
                        "show_name": base_meta.get("show_name", "Unknown Show"),
                        "episode_name": base_meta.get("episode_name", "Unknown Episode"),
                        "publisher": base_meta.get("publisher", "Unknown Publisher"),
                        "category": category,
                        "rss_link": base_meta.get("rss_link", ""),
                    }

                    outfile.write(json.dumps(elastic_doc) + "\n")
                    self.total_chunks += 1

        self._print_summary(start_time)

    def _print_summary(self, start_time: float) -> None:
        """Displays processing results."""
        elapsed = round(time.time() - start_time, 2)
        logging.info("========================================")
        logging.info(f"DATASET BUILD COMPLETE in {elapsed}s")
        logging.info(f"Total Chunks: {self.total_chunks}")
        logging.info(f"Episodes missing category: {self.missing_enriched_count}")
        logging.info(f"Output: {self.output_file}")
        logging.info("========================================")


if __name__ == "__main__":
    builder = ElasticDatasetBuilder(
        tsv_path=TSV_FILE,
        chunks_dir=CHUNKED_DIR,
        enriched_dir=ENRICHED_META_DIR,
        output_file=ELASTIC_READY_FILE
    )
    builder.build()