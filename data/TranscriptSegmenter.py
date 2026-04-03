"""
TranscriptSegmenter.py — Chunks podcast segments with a sliding window strategy.

Merges small speech-to-text segments into ~120-second chunks with 30-second overlap,
assigning global and local IDs to each chunk.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GROUPED_DATA_FILE, CHUNKED_DIR, TARGET_CHUNK_SECONDS, OVERLAP_SECONDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TranscriptSegmenter:
    """
    Reads raw podcast segments, applies a sliding window chunking strategy,
    and assigns global and local IDs to the new chunks.
    """

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        target_chunk_seconds: int = TARGET_CHUNK_SECONDS,
        overlap_seconds: int = OVERLAP_SECONDS,
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.target_chunk_seconds = target_chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.global_segment_id = 1

        self._setup_directories()

    def _setup_directories(self) -> None:
        """Checks if directories exist; if not, creates them."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory verified: {self.output_dir}")

    def process_file(self) -> None:
        """Reads the input JSONL file line by line and processes each podcast."""
        if not self.input_file.exists():
            logging.error(f"Input file not found: {self.input_file}")
            return

        with open(self.input_file, mode="r", encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    self._process_single_podcast(data)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON at line {line_number}. Skipping.")

    def _process_single_podcast(self, data: Dict[str, Any]) -> None:
        """Chunks the segments with a sliding window and saves the result."""
        file_id = str(data.get("file_id", ""))
        clean_file_id = file_id.removesuffix(".json")

        if not clean_file_id:
            logging.warning("Skipping entry: Missing 'file_id'.")
            return

        raw_segments = data.get("segments", [])
        if not raw_segments:
            logging.warning(f"Skipping '{clean_file_id}': No segments found.")
            return

        chunked_segments = self._apply_sliding_window(raw_segments)
        output_data = {"file_id": clean_file_id, "segments": chunked_segments}
        output_path = self.output_dir / f"{clean_file_id}_chunked.json"

        if output_path.exists():
            logging.warning(
                f"Duplicate chunked file detected for ID '{clean_file_id}'. Skipping."
            )
        else:
            with open(output_path, mode="w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)

    def _apply_sliding_window(
        self, raw_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merges small segments into larger chunks with overlap,
        assigning local chunk IDs and global segment IDs.
        """
        chunks = []
        current_chunk_text = []
        current_start_time = raw_segments[0]["start"]
        current_end_time = raw_segments[0]["end"]
        local_chunk_id = 1

        for i, segment in enumerate(raw_segments):
            current_chunk_text.append(segment["text"])
            current_end_time = segment["end"]

            current_duration = current_end_time - current_start_time

            if current_duration >= self.target_chunk_seconds:
                chunks.append(
                    {
                        "segment_id": self.global_segment_id,
                        "chunk_id": local_chunk_id,
                        "text": " ".join(current_chunk_text).strip(),
                        "start": current_start_time,
                        "end": current_end_time,
                    }
                )

                self.global_segment_id += 1
                local_chunk_id += 1

                overlap_start_target = current_end_time - self.overlap_seconds
                current_chunk_text = []

                for j in range(i, -1, -1):
                    if raw_segments[j]["start"] <= overlap_start_target:
                        current_start_time = raw_segments[j]["start"]
                        for k in range(j, i + 1):
                            current_chunk_text.append(raw_segments[k]["text"])
                        break

        if current_chunk_text:
            chunks.append(
                {
                    "segment_id": self.global_segment_id,
                    "chunk_id": local_chunk_id,
                    "text": " ".join(current_chunk_text).strip(),
                    "start": current_start_time,
                    "end": current_end_time,
                }
            )
            self.global_segment_id += 1

        return chunks


if __name__ == "__main__":
    chunker = TranscriptSegmenter(
        input_file=GROUPED_DATA_FILE,
        output_dir=CHUNKED_DIR,
    )
    chunker.process_file()
    print("Chunking complete! Check the output directory.")