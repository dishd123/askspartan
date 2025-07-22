SCRAPER_CONFIG = {
    "base_url_substr": "sjsu.edu",
    "max_pages": 1000,
    "output_file": "data/scraped_data_v2.jsonl",
    "start_urls": [
        "https://www.sjsu.edu/",
        "https://catalog.sjsu.edu/content.php?catoid=13&navoid=4983",
        "https://catalog.sjsu.edu/preview_program.php?catoid=14&poid=8269",
    ],
}

TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1500,
    "input_data_file": SCRAPER_CONFIG["output_file"],
    "output_data_file": SCRAPER_CONFIG["output_file"].replace(
        ".jsonl", "_chunked.jsonl"
    ),
}

CHROMA_CONFIG = {
    "chroma_db_dir": "embeddings_db",
    "collection_name": "askspartan",
    "input_data_file": TEXT_SPLITTER_CONFIG["output_data_file"],
    "n_results": 5,
}

SENTENCE_TRANSFORMERS_CONFIG = {"model_name": "BAAI/bge-base-en-v1.5"}
