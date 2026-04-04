"""
PodcastProcessor.py — Creates full transcripts and segment files per episode.

Reads the grouped JSONL (one line per episode with segments array) and produces:
  - One .txt transcript per episode
  - One .json segment file per episode
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GROUPED_DATA_FILE, TRANSCRIPT_DIR, SEGMENTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PodcastProcessor:
    """
    Handles the extraction and formatting of podcast transcripts and segments
    from a JSONL file.
    """

    def __init__(self, input_file: Path, transcript_dir: Path, segments_dir: Path):
        self.input_file = input_file
        self.transcript_dir = transcript_dir
        self.segments_dir = segments_dir
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Checks if directories exist; if not, creates them."""
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Output directories verified/created.")

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
        """Extracts text, checks for duplicates, and writes the output files."""
        file_id = str(data.get("file_id"))
        file_id = file_id.removesuffix(".json")
        if not file_id:
            logging.warning("Skipping entry: Missing 'file_id'.")
            return

        segments = data.get("segments", [])
        transcript = " ".join(s.get("text", "") for s in segments).strip()

        transcript_path = self.transcript_dir / f"{file_id}.txt"
        segment_path = self.segments_dir / f"{file_id}.json"

        if transcript_path.exists():
            logging.warning(
                f"Duplicate transcript detected for ID '{file_id}'. Skipping."
            )
        else:
            with open(transcript_path, mode="w", encoding="utf-8") as f:
                f.write(transcript)

        if segment_path.exists():
            logging.warning(
                f"Duplicate segments JSON detected for ID '{file_id}'. Skipping."
            )
        else:
            with open(segment_path, mode="w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    processor = PodcastProcessor(
        input_file=GROUPED_DATA_FILE,
        transcript_dir=TRANSCRIPT_DIR,
        segments_dir=SEGMENTS_DIR,
    )
    processor.process_file()