import logging
from pathlib import Path
from typing import List, Callable, Set
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATASET_DIR,
    CLEANED_DATA_FILE,
    GROUPED_DATA_FILE,
    TRANSCRIPT_DIR,
    SEGMENTS_DIR,
    CHUNKED_DIR,
    TARGET_CHUNK_SECONDS,
    OVERLAP_SECONDS,
    TSV_FILE,
    ENRICHED_META_DIR,
    MAX_WORKERS_NETWORK,
    ELASTIC_READY_FILE,
    EMBEDDING_INPUT_FILE,
    EMBEDDING_OUTPUT_FILE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    )
from preprocess_data import TranscriptExtractor
from podcast_processor import PodcastProcessor
from transcript_segmenter import TranscriptSegmenter
from rss_enrichment import RSSEnrichmentPipeline
from build_elastic_data import ElasticDatasetBuilder
from embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class DataPreprocessingPipeline:
    """The Product: A configured sequence of preprocessing steps."""
    
    def __init__(self):
        self.steps: List[Callable] = []
        self.step_names: List[str] = []

    def add_step(self, name: str, command: Callable) -> None:
        self.steps.append(command)
        self.step_names.append(name)

    def run(self) -> None:
        """Executes the pipeline steps sequentially with a progress bar."""
        print("\n" + "="*50)
        print("Starting Data Preprocessing Pipeline...")
        print("="*50 + "\n")
        
        with tqdm(total=len(self.steps), desc="Pipeline Progress", unit="step") as pbar:
            for name, step_func in zip(self.step_names, self.steps):
                tqdm.write(f"\n---> Executing: {name}")
                
                try:
                    step_func()
                except Exception as e:
                    tqdm.write(f"\n[ERROR] Pipeline failed at step '{name}': {e}")
                    raise e
                
                pbar.update(1)
                
        print("\n" + "="*50)
        print("Pipeline Execution Complete!")
        print("="*50 + "\n")


class PipelineBuilder:
    """The Builder: Configures steps using config.py and validates dependencies."""
    
    def __init__(self):
        self.pipeline = DataPreprocessingPipeline()
        self.requested_steps: Set[int] = set()
        self.include_processor = True  # Step 2 runs by default

    def set_skip_podcast_processor(self) -> 'PipelineBuilder':
        """Optional: Skip Step 2 (PodcastProcessor)."""
        self.include_processor = False
        return self

    def add_transcript_extractor(self) -> 'PipelineBuilder':
        """Step 1: Extract and Group Transcripts."""
        self.requested_steps.add(1)
        
        def run_step_1():
            extractor = TranscriptExtractor(
                input_dir=RAW_DATASET_DIR,
                output_file=CLEANED_DATA_FILE,
                grouped_file=GROUPED_DATA_FILE
            )
            extractor.run_extraction()
            extractor.group_by_episode()
            
        self.pipeline.add_step("Step 1: TranscriptExtractor", run_step_1)
        return self

    def add_podcast_processor(self) -> 'PipelineBuilder':
        """Step 2: Generate Transcript and Segment Files."""
        if not self.include_processor:
            return self
            
        self.requested_steps.add(2)
        
        def run_step_2():
            processor = PodcastProcessor(
                input_file=GROUPED_DATA_FILE,
                transcript_dir=TRANSCRIPT_DIR,
                segments_dir=SEGMENTS_DIR
            )
            processor.process_file()
            
        self.pipeline.add_step("Step 2: PodcastProcessor", run_step_2)
        return self

    def add_transcript_segmenter(self) -> 'PipelineBuilder':
        """Step 3: Chunk podcast segments."""
        self.requested_steps.add(3)
        
        def run_step_3():
            # Uses TARGET_CHUNK_SECONDS and OVERLAP_SECONDS implicitly based on your class defaults
            segmenter = TranscriptSegmenter(
                input_file=GROUPED_DATA_FILE,
                output_dir=CHUNKED_DIR,
                target_chunk_seconds=TARGET_CHUNK_SECONDS,
                overlap_seconds=OVERLAP_SECONDS
            )
            segmenter.process_file()
            
        self.pipeline.add_step("Step 3: TranscriptSegmenter", run_step_3)
        return self

    def add_rss_enrichment(self) -> 'PipelineBuilder':
        """Step 4: Enrich metadata via RSS."""
        self.requested_steps.add(4)
        
        def run_step_4():
            enricher = RSSEnrichmentPipeline(
                tsv_path=TSV_FILE,
                output_dir=ENRICHED_META_DIR,
                max_workers=MAX_WORKERS_NETWORK
            )
            enricher.process_dataset()
            
        self.pipeline.add_step("Step 4: RSSEnrichmentPipeline", run_step_4)
        return self

    def add_elastic_builder(self) -> 'PipelineBuilder':
        """Step 5: Merge into Elasticsearch-ready JSONL."""
        self.requested_steps.add(5)
        
        def run_step_5():
            builder = ElasticDatasetBuilder(
                tsv_path=TSV_FILE,
                chunks_dir=CHUNKED_DIR,
                enriched_dir=ENRICHED_META_DIR,
                output_file=ELASTIC_READY_FILE
            )
            builder.build()
            
        self.pipeline.add_step("Step 5: ElasticDatasetBuilder", run_step_5)
        return self

    def add_podcast_embedder(self) -> 'PipelineBuilder':
        """Step 6: Generate Embeddings."""
        self.requested_steps.add(6)
        
        def run_step_6():
            generator = EmbeddingGenerator(
                input_file=EMBEDDING_INPUT_FILE,
                output_file=EMBEDDING_OUTPUT_FILE,
                model_name=EMBEDDING_MODEL_NAME,
                batch_size=EMBEDDING_BATCH_SIZE
            )
            generator.process()
            
        self.pipeline.add_step("Step 6: PodcastEmbedder", run_step_6)
        return self

    def _verify_dependency(self, step_id: int, expected_files: List[Path], dependency_step: int, dependency_name: str) -> None:
        """Helper to ensure prior steps ran OR the required data exists on disk."""
        if dependency_step not in self.requested_steps:
            for file_path in expected_files:
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"[Validation Error] Step {step_id} requires data from {dependency_name}, "
                        f"but step {dependency_step} was skipped and path does not exist: {file_path}"
                    )

    def build(self) -> DataPreprocessingPipeline:
        """Validates all dependencies and returns the ready-to-run Pipeline."""
        logging.info("Validating pipeline dependencies against config.py...")

        # Base Validation: TSV file must always exist if Step 4 or 5 is requested
        if 4 in self.requested_steps or 5 in self.requested_steps:
            if not TSV_FILE.exists():
                raise FileNotFoundError(f"Base TSV metadata file missing: {TSV_FILE}")

        # Validate Step 2 inputs
        if 2 in self.requested_steps:
            self._verify_dependency(2, [GROUPED_DATA_FILE], 1, "TranscriptExtractor")

        # Validate Step 3 inputs
        if 3 in self.requested_steps:
            self._verify_dependency(3, [GROUPED_DATA_FILE], 1, "TranscriptExtractor")

        # Validate Step 5 inputs
        if 5 in self.requested_steps:
            self._verify_dependency(5, [CHUNKED_DIR], 3, "TranscriptSegmenter")
            self._verify_dependency(5, [ENRICHED_META_DIR], 4, "RSSEnrichmentPipeline")

        # Validate Step 6 inputs
        if 6 in self.requested_steps:
            self._verify_dependency(6, [ELASTIC_READY_FILE], 5, "ElasticDatasetBuilder")

        logging.info("Pipeline validation passed successfully.")
        return self.pipeline


if __name__ == "__main__":
    # Example 1: Build and Run the Full Pipeline
    try:
        pipeline = (
            PipelineBuilder()
            .add_transcript_extractor()
            .add_podcast_processor()      # Runs by default, explicitly requested here
            .add_transcript_segmenter()
            .add_rss_enrichment()
            .add_elastic_builder()
            .add_podcast_embedder()
            .build()
        )
        pipeline.run()
    except FileNotFoundError as e:
        print(e)

    # Example 2: Resume Pipeline halfway (e.g., skip 1 and 2, but requires grouped data to exist)
    """
    try:
        resume_pipeline = (
            PipelineBuilder()
            .set_skip_podcast_processor() # Ensure Step 2 is explicitly skipped
            .add_transcript_segmenter()   # Start at Step 3
            .add_elastic_builder()        # Skip Step 4 (requires enriched data to already exist)
            .add_podcast_embedder()
            .build()
        )
        resume_pipeline.run()
    except FileNotFoundError as e:
        print(f"Validation caught missing file correctly: {e}")
    """