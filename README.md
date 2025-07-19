# 🏛️ AskSpartan: AI-Based SJSU FAQ Chatbot

AskSpartan is an experimental AI chatbot designed to answer frequently asked questions about **San Jose State University (SJSU)**. It uses a **Retrieval-Augmented Generation (RAG)** architecture, leveraging local large language models (LLMs) and embeddings to provide answers sourced exclusively from publicly available data on the SJSU website.

This project was created for educational purposes to explore the implementation of end-to-end RAG pipelines with open-source tools.

---

## 🚀 Key Features

-   **Completely Offline & Free:** Runs locally without any paid APIs.
-   **SJSU-Focused:** Trained on scraped content from the SJSU website.
-   **RAG Architecture:** Ensures answers are grounded in factual, retrieved documents.
-   **Simple Web UI:** An intuitive and easy-to-use interface powered by **Streamlit**.
-   **Lightweight & Efficient:** Designed to run on consumer hardware (like a MacBook) using quantized models like **Phi-3 Mini**, **Mistral**, or **Llama**.

---

## 🛠️ Tech Stack

-   **Language:** Python 3
-   **Web Scraping:** `requests`, `beautifulsoup4`
-   **Embeddings:** `sentence-transformers`
-   **Vector Store:** `chromadb`
-   **LLM Inference:** `llama-cpp-python` or Hugging Face `transformers`
-   **Web UI:** `streamlit`

---

## ⚙️ Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

-   Python 3.8 or newer
-   Git

### 2. Installation

First, clone the repository to your local machine:

```bash
git clone [https://github.com/your-username/askspartan.git](https://github.com/your-username/askspartan.git)
cd askspartan
```
Next, install the required Python dependencies:
```
pip install -r requirements.txt
```

### 3. Download a Model

This project uses quantized GGUF models for efficient local inference. Download a model of your choice (e.g., Phi-3-mini) and place it in the models/ directory.

You can find suitable models on Hugging Face, like the [Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf).

### 4. Scrape Website Data

Run the scraper to collect publicly available information from the SJSU website. This data will be used to build the knowledge base.
```
python scraper.py
```

> **Note**: The scraper is configured to be respectful of the SJSU servers. Please be mindful when modifying scraping parameters. The user is solely responsible for ensuring compliance with SJSU's terms of service and web policies.

### 5. Generate Embeddings & Run the Chatbot

*(This section is currently a work in progress.)*

Once the setup is complete, you can launch the Streamlit web application. The first run will generate embeddings and store them in the ChromaDB vector store.

```
streamlit run ui.py
```

---

## 📂 Project Structure

```
askspartan/
├── data/             # Scraped and cleaned SJSU content
├── embeddings/       # Vector database files (ChromaDB)
├── models/           # Quantized LLM files (e.g., .gguf)
├── scraper.py        # Python script to scrape SJSU website
├── main.py           # Core RAG pipeline logic
├── ui.py             # Streamlit web application
├── requirements.txt  # Python package dependencies
└── README.md         # This file
```

---


## ⚖️ Disclaimer

This is an unofficial, community-driven project and is not affiliated with San Jose State University. All data is sourced from publicly accessible websites, and the accuracy of the responses depends entirely on the information available at the time of scraping.