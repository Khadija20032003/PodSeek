"""
TranscriptSegmenter.py — Hierarchical chunking (Parent/Child) for RAG.

Creates fixed-size parent windows (default 120s), then subdivides each parent into
fixed-size child windows (default 30s).

The output is a denormalized *flat* list of child chunks that each carry the full
parent text so downstream steps can embed/search children while returning parents to
the LLM.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    GROUPED_DATA_FILE,
    CHUNKED_DIR,
    PARENT_CHUNK_SIZE_SECONDS,
    CHILD_CHUNK_SIZE_SECONDS,
    PARENT_OVERLAP_SECONDS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TranscriptSegmenter:
    """
    Reads raw podcast segments and emits denormalized child chunks that contain
    parent context fields.
    """

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        parent_chunk_seconds: int = PARENT_CHUNK_SIZE_SECONDS,
        child_chunk_seconds: int = CHILD_CHUNK_SIZE_SECONDS,
        parent_overlap_seconds: int = PARENT_OVERLAP_SECONDS,
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.parent_chunk_seconds = parent_chunk_seconds
        self.child_chunk_seconds = child_chunk_seconds
        self.parent_overlap_seconds = parent_overlap_seconds

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
        """Creates hierarchical chunks (parent/child) and saves the result.

        Output is a *flat list of child chunks*, each containing its parent's text.
        """
        file_id = str(data.get("file_id", ""))
        clean_file_id = file_id.removesuffix(".json")

        if not clean_file_id:
            logging.warning("Skipping entry: Missing 'file_id'.")
            return

        raw_segments = data.get("segments", [])
        if not raw_segments:
            logging.warning(f"Skipping '{clean_file_id}': No segments found.")
            return

        output_path = self.output_dir / f"{clean_file_id}_chunked.jsonl"

        if output_path.exists():
            logging.warning(
                f"Duplicate chunked file detected for ID '{clean_file_id}'. Skipping."
            )
        else:
            with open(output_path, mode="w", encoding="utf-8") as f:
                for chunk in self._iter_parent_child_chunks(
                    raw_segments=raw_segments,
                    file_id=clean_file_id,
                ):
                    f.write(json.dumps(chunk) + "\n")

    def _normalize_segment(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalizes a raw segment into a consistent schema.

        Expected input keys are produced by the upstream pipeline.
        If required fields are missing or invalid, returns None.
        """
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            return None

        if end <= start:
            return None

        text = str(segment.get("text", "")).strip()
        if not text:
            return None

        return {"start": start, "end": end, "text": text}

    def _collect_text_in_window(
        self,
        segments: List[Dict[str, Any]],
        window_start: float,
        window_end: float,
    ) -> str:
        """Collects transcript text from segments overlapping [window_start, window_end].

        We use overlap semantics so that child chunks align to time windows even if the
        upstream ASR segments don't align perfectly to those boundaries.
        """
        texts: List[str] = []
        for seg in segments:
            if seg["end"] <= window_start:
                continue
            if seg["start"] >= window_end:
                break
            texts.append(seg["text"])
        return " ".join(texts).strip()

    def _iter_parent_child_chunks(
        self,
        raw_segments: List[Dict[str, Any]],
        file_id: str,
    ):
        """Hierarchical chunking (Parent -> Child) with a denormalized flat output.

        - Parent windows are fixed-size (default 120s).
        - Each parent is subdivided into fixed-size child windows (default 30s).

        Output schema (per child chunk) matches the required downstream contract:
        {
          "file_id": ..., "chunk_id": ..., "text": ..., "start_time": ..., "end_time": ...,
          "parent_id": ..., "parent_text": ..., "parent_start_time": ..., "parent_end_time": ...
        }
        """

        normalized: List[Dict[str, Any]] = []
        for seg in raw_segments:
            norm = self._normalize_segment(seg)
            if norm is not None:
                normalized.append(norm)

        if not normalized:
            return

        # Ensure monotonic ordering so we can break early in window scans.
        normalized.sort(key=lambda s: (s["start"], s["end"]))

        first_start = normalized[0]["start"]
        last_end = max(s["end"] for s in normalized)

        parent_size = float(self.parent_chunk_seconds)
        child_size = float(self.child_chunk_seconds)
        parent_overlap = float(self.parent_overlap_seconds)
        parent_stride = max(parent_size - parent_overlap, child_size)

        parents: List[Dict[str, Any]] = []
        parent_start = first_start
        parent_idx = 1
        while parent_start < last_end:
            parent_end = min(parent_start + parent_size, last_end)
            parent_text = self._collect_text_in_window(normalized, parent_start, parent_end)
            if parent_text:
                parents.append(
                    {
                        "parent_id": f"{file_id}_parent{parent_idx}",
                        "parent_start_time": float(parent_start),
                        "parent_end_time": float(parent_end),
                        "parent_text": parent_text,
                    }
                )
                parent_idx += 1
            parent_start += parent_stride

        if not parents:
            return

        child_idx_global = 1
        child_start = first_start
        p0 = 0

        while child_start < last_end:
            child_end = min(child_start + child_size, last_end)
            child_text = self._collect_text_in_window(normalized, child_start, child_end)

            if child_text:
                while p0 < len(parents) and parents[p0]["parent_end_time"] <= child_start:
                    p0 += 1

                parent_ids: List[str] = []
                parent_texts: List[str] = []
                p = p0
                while p < len(parents) and parents[p]["parent_start_time"] < child_end:
                    if parents[p]["parent_end_time"] > child_start:
                        parent_ids.append(parents[p]["parent_id"])
                        parent_texts.append(parents[p]["parent_text"])
                    p += 1

                if parent_ids:
                    primary_parent = parents[p0] if p0 < len(parents) else parents[-1]
                    yield {
                        "file_id": file_id,
                        "chunk_id": f"{file_id}_child{child_idx_global}",
                        "text": child_text,
                        "start_time": float(child_start),
                        "end_time": float(child_end),
                        "parent_id": parent_ids[0],
                        "parent_text": parent_texts[0],
                        "parent_start_time": float(primary_parent["parent_start_time"]),
                        "parent_end_time": float(primary_parent["parent_end_time"]),
                        "parent_ids": parent_ids,
                        "parent_texts": parent_texts,
                    }
                    child_idx_global += 1

            child_start = child_end

    def _create_parent_child_chunks(
        self,
        raw_segments: List[Dict[str, Any]],
        file_id: str,
    ) -> List[Dict[str, Any]]:
        """Compatibility wrapper returning a list."""
        return list(self._iter_parent_child_chunks(raw_segments=raw_segments, file_id=file_id) or [])

if __name__ == "__main__":
    chunker = TranscriptSegmenter(
        input_file=GROUPED_DATA_FILE,
        output_dir=CHUNKED_DIR,
    )
    chunker.process_file()
    print("Chunking complete! Check the output directory.")