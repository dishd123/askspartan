SCRAPER_CONFIG = {
    "base_url_substr": "sjsu.edu",
    "max_pages": 2000,
    "output_file": "data/scraped_data_v3.jsonl",
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

LLM_CONFIG = {
    "RESPONSE_GENERATOR_CONFIG": {
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
        "prompt_template": """
<|user|> You are a helpful AI assistant. Your task is to answer the user's Question based on the provided Context.

### Context:
{context}

### Question:
{query} <|end|>
<|assistant|>""",
    },
    "SUMMARY_GENERATOR_CONFIG": {
        "CALL_CONFIG": {
            "max_tokens": 250,
            "temperature": 0.5,
            "top_p": 0.95,
            "stop": ["<|end|>", "<|user|>"],
            "repeat_penalty": 1.3,
        },
        "prompt_template": """
<|user|> You are a helpful AI assistant. Your task is to summarize the following conversation between a USER and an AI CHATBOT. Follow these instructions:
(1) Summary should definitely include the most crucial and important information.
(2) Summary should be brief and succinct.

### Conversation:
{raw_dialogue} <|end|>
<|assistant|>""",
        "use_coversation_history": True,
        "coversation_history_lookback": 2,
    },
    "QUERY_REFORMULATION_CONFIG": {
        "CALL_CONFIG": {
            "max_tokens": 150,
            "temperature": 0.5,
            "top_p": 0.95,
            "stop": ["<|end|>", "<|user|>"],
            "repeat_penalty": 1.3,
        },
        "prompt_template": """
<|user>| You are a helpful AI assistant. Your task is to reformulate the User's Query to include relevant context from the previous conversation. Conversation summary is provided below. If there is no relevant context, just return the original User's Query.

### Conversation Summary:
{summary}

### User's Query:
{query}

### Reformulated Query: <|end|>
<|assistant|>
 """,
    },
}
