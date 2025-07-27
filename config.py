SCRAPER_CONFIG = {
    "base_url_substr": "sjsu.edu",
    "max_pages": 2000,
    "output_file": "data/scraped_data_v2.jsonl",
    "start_urls": [
        "https://www.sjsu.edu/",
        "https://catalog.sjsu.edu/content.php?catoid=14&navoid=5117",
        "https://catalog.sjsu.edu/content.php?catoid=14&navoid=5106",
        "https://catalog.sjsu.edu/content.php?catoid=14&navoid=5107",
        "https://catalog.sjsu.edu/preview_program.php?catoid=14&poid=8269",
        "https://www.sjsu.edu/tuition-and-fees/index.php",
        "https://www.sjsu.edu/global/",
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
    "chroma_db_dir": f"embeddings/{TEXT_SPLITTER_CONFIG["output_data_file"].replace('.jsonl', '')}",
    "collection_name": "askspartan",
    "input_data_file": TEXT_SPLITTER_CONFIG["output_data_file"],
    "n_results": 5,
}

SENTENCE_TRANSFORMERS_CONFIG = {"model_name": "BAAI/bge-base-en-v1.5"}

LLM_RESPONSE_GENERATOR_CONFIG = {
    "INIT_CONFIG": {
        "model_path": "models/Phi-3-mini-4k-instruct-q4.gguf",
        "n_ctx": 3500,
    },
    "CALL_CONFIG": {
        "max_tokens": 700,
        "temperature": 0.7,
        "top_p": 0.95,
        "stop": ["<|end|>", "<|user|>"],
        "repeat_penalty": 1.1,
    },
    "prompt_template": "<|user|>\n ### Context:\n{context}\n\n ### Question:\n{query} <|end|>\n <|assistant|>",
    "SUMMARY_GENERATOR_CONFIG": {
        "max_tokens": 250,
        "temperature": 0.5,
        "top_p": 0.95,
        "stop": ["<|end|>", "<|user|>"],
        "repeat_penalty": 1.3,
    },
    "summary_template": "<|user|>\n (1) Summarize the following conversation in a concise manner.\n(2) Only keep the most crucial and important information.\n(3) Keep it brief and to the point.:\n\n{raw_dialogue} <|end|>\n <|assistant|>",
    "use_chat_history": True,
    "chat_history_lookback": 3,
}
