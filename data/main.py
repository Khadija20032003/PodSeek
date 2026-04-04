from data.pipeline import PipelineBuilder


def main():
    try:
        pipeline = (
            PipelineBuilder()
            .add_transcript_extractor()
            .add_podcast_creator()
            .add_transcript_segmenter()
            .add_rss_enrichment()
            .add_elastic_builder()
            .add_podcast_embedder()
            .build()
        )
        pipeline.run()
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
