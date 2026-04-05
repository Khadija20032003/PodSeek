"""
preprocess_data.py (TranscriptExtractor) — Extracts clean transcript segments from raw podcast JSON files.
Uses multiprocessing to process files across CPU cores for maximum speed.
"""

import json
import logging
import time
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import config from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TRANSCRIPTS_JSON_DIR,
    CLEANED_OUTPUT_DIR,
    CLEANED_DATA_FILE,
    GROUPED_DATA_FILE,
    MAX_WORKERS_CPU,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TranscriptExtractor:
    """
    Handles parallel extraction of raw JSON podcast data into cleaned JSONL formats.
    """

    def __init__(self, input_dir: Path, output_file: Path, grouped_file: Path):
        self.input_dir = input_dir
        self.output_file = output_file
        self.grouped_file = grouped_file
        self.num_workers = min(cpu_count(), MAX_WORKERS_CPU)

        # Ensure output directories exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.grouped_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_segment_data(filepath: Path) -> List[Dict[str, Any]]:
        """
        Worker function: Processes a single JSON file.
        Static because it must be picklable for multiprocessing.
        """
        chunks = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for result in data.get("results", []):
                # Safely navigate the nested JSON structure
                alt = result.get("alternatives", [{}])[0]
                transcript = alt.get("transcript", "").strip()
                words = alt.get("words", [])

                if not transcript or not words:
                    continue

                # Parse timestamps (e.g., "1.5s" -> 1.5)
                start_sec = float(words[0]["startTime"].rstrip("s"))
                end_sec = float(words[-1]["endTime"].rstrip("s"))

                chunks.append(
                    {
                        "file_id": filepath.name,
                        "text": transcript,
                        "start_time": start_sec,
                        "end_time": end_sec,
                    }
                )
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            pass  # Skip malformed segments
        except Exception as e:
            logging.debug(f"Error processing {filepath.name}: {e}")

        return chunks

    def _get_all_json_files(self) -> List[Path]:
        """Recursively collects all .json files from the input directory."""
        return list(self.input_dir.rglob("*.json"))

    def run_extraction(self) -> None:
        """Phase 1 & 2: Collect files and process them in parallel."""
        logging.info(f"Scanning {self.input_dir}...")
        files = self._get_all_json_files()
        total_files = len(files)
        logging.info(
            f"Found {total_files} files. Starting extraction with {self.num_workers} workers..."
        )

        start_time = time.time()
        processed_count = 0
        total_chunks = 0
        write_buffer = []

        with open(self.output_file, "w", encoding="utf-8") as outfile:
            with Pool(processes=self.num_workers) as pool:
                # Use imap_unordered for better performance with large datasets
                for segments in pool.imap_unordered(
                    self._extract_segment_data, files, chunksize=500
                ):
                    processed_count += 1

                    if segments:
                        for seg in segments:
                            write_buffer.append(json.dumps(seg))
                        total_chunks += len(segments)

                    # Periodically flush buffer to disk
                    if len(write_buffer) >= 5000:
                        outfile.write("\n".join(write_buffer) + "\n")
                        write_buffer.clear()

                    # Progress logging
                    if processed_count % 5000 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        rem_min = ((total_files - processed_count) / rate) / 60
                        logging.info(
                            f"Progress: {processed_count}/{total_files} files | {total_chunks} chunks | ~{rem_min:.1f}m left"
                        )

            # Final flush
            if write_buffer:
                outfile.write("\n".join(write_buffer) + "\n")

        duration = (time.time() - start_time) / 60
        logging.info(
            f"Extraction complete. {total_chunks} chunks saved in {duration:.2f}m."
        )

    def group_by_episode(self) -> None:
        """Phase 3: Groups individual segments into full episodes."""
        logging.info("Grouping segments by episode...")
        start_time = time.time()
        episodes = {}

        if not self.output_file.exists():
            logging.error("Cleaned data file not found. Run extraction first.")
            return

        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                file_id = data["file_id"]

                if file_id not in episodes:
                    episodes[file_id] = []

                episodes[file_id].append(
                    {
                        "text": data["text"],
                        "start": data["start_time"],
                        "end": data["end_time"],
                    }
                )

        with open(self.grouped_file, "w", encoding="utf-8") as f:
            for file_id, segments in episodes.items():
                record = {"file_id": file_id, "segments": segments}
                f.write(json.dumps(record) + "\n")

        elapsed = time.time() - start_time
        logging.info(f"Grouped {len(episodes)} episodes in {elapsed:.2f}s.")
        logging.info(f"Final output: {self.grouped_file}")


if __name__ == "__main__":
    extractor = TranscriptExtractor(
        input_dir=TRANSCRIPTS_JSON_DIR,
        output_file=CLEANED_DATA_FILE,
        grouped_file=GROUPED_DATA_FILE,
    )

    extractor.run_extraction()
    extractor.group_by_episode()
