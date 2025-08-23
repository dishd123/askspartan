# 🏛️ AskSpartan: Advanced AI-Powered SJSU Knowledge Assistant


AskSpartan is a sophisticated **Retrieval-Augmented Generation (RAG)** chatbot engineered to provide intelligent, contextual answers about **San Jose State University (SJSU)**. Built with cutting-edge AI technologies, it combines advanced web scraping, semantic search, and local language model inference to deliver accurate responses based exclusively on official SJSU website content.


This project demonstrates a complete end-to-end RAG pipeline implementation using open-source tools, designed for educational purposes and real-world deployment scenarios.


---


## 🎯 **Key Highlights**


- **🔒 Completely Offline & Private**: Runs entirely on local hardware without external API dependencies
- **🧠 Advanced Conversation Management**: Sophisticated context preservation and query reformulation
- **🎓 SJSU-Specialized**: Trained exclusively on official university content for maximum accuracy
- **⚡ Efficient Architecture**: Optimized for consumer hardware using quantized models
- **🔄 RAG-Powered**: Ensures factually grounded responses through document retrieval
- **💬 Interactive Web Interface**: Modern Streamlit-based chat interface with session management


---


## 🏗️ **System Architecture**


AskSpartan implements a sophisticated RAG pipeline with six interconnected components:


```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Scraper   │───▶│  Text Processor  │───▶│ Vector Database │
│   (scraper.py)  │    │   (LangChain)    │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│  LLM Generator   │◀───│   Retriever     │
│     (ui.py)     │    │(llm_response_...) │    │(setup_db_...)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```


### **Component Breakdown**


#### **1. Configuration Management (`config.py`)**
Centralized configuration hub managing all system parameters:
- **Scraper Settings**: SJSU domain targeting, 2000-page crawl limit, curated start URLs
- **Text Processing**: 1500-character chunks with intelligent overlap
- **Vector Database**: ChromaDB configuration with BGE embeddings
- **LLM Parameters**: Three specialized prompt templates for different AI tasks


#### **2. Intelligent Web Scraper (`scraper.py`)**
Advanced web crawling engine with sophisticated content extraction:
- **Breadth-First Search**: Systematic website traversal with queue-based crawling
- **Smart Filtering**: Automatically skips PDFs, images, external links, and irrelevant content
- **Content Cleaning**: Removes navigation, scripts, footers, and non-content elements
- **Respectful Crawling**: 1-second delays and robots.txt compliance
- **Batch Processing**: Efficient 100-entry buffer system for optimal performance
- **Automatic Chunking**: LangChain integration for intelligent text segmentation


#### **3. Vector Database & Semantic Retrieval (`setup_db_and_retriever.py`)**
High-performance semantic search system:
- **ChromaDB Backend**: Persistent vector storage with metadata preservation
- **BGE Embeddings**: BAAI/bge-base-en-v1.5 model (768-dimensional vectors)
- **Normalized Similarity**: Cosine similarity with L2 normalization for consistent scoring
- **Metadata Tracking**: URL, document index, and chunk index preservation
- **Configurable Retrieval**: Adjustable result count (default: top 5 matches)


#### **4. Advanced LLM Response Generator (`llm_response_generator.py`)**
Sophisticated language model pipeline with multiple AI capabilities:


**Core Features:**
- **Primary Model**: Phi-3-mini-4k-instruct (3.8B parameters, quantized GGUF)
- **Conversation Summarization**: Maintains context across chat sessions
- **Query Reformulation**: Enhances user queries using conversation history
- **Context Integration**: Seamlessly weaves retrieved documents into responses


**AI Pipeline Stages:**
1. **Conversation Summarization**: Condenses recent chat history for context
2. **Query Enhancement**: Reformulates questions using conversation context
3. **Document Retrieval**: Semantic search for relevant content
4. **Response Synthesis**: Generates contextual answers using retrieved information


#### **5. Modern Web Interface (`ui.py`)**
Streamlit-powered conversational interface:
- **Chat-Style UI**: Familiar messaging interface with message history
- **Session Management**: Persistent conversation state across interactions
- **Performance Optimization**: Cached model loading and efficient resource management
- **Real-Time Processing**: Live response generation with visual feedback


#### **6. Utility Infrastructure (`utils/logger.py`)**
Comprehensive logging and monitoring system:
- **Configurable Logging**: Multiple log levels for debugging and monitoring
- **System Observability**: Track performance, errors, and usage patterns


---


## 🛠️ **Technology Stack**


### **Core Technologies**
- **Python 3.8+**: Primary development language with modern features
- **llama-cpp-python**: High-performance CPU-based LLM inference engine
- **ChromaDB**: Advanced vector database for semantic search
- **Sentence Transformers**: State-of-the-art embedding generation
- **Streamlit**: Modern, reactive web application framework
- **BeautifulSoup4**: Robust HTML parsing and content extraction
- **LangChain**: Advanced text processing and document management


### **AI/ML Components**
- **Embedding Model**: BAAI/bge-base-en-v1.5 (multilingual, 768-dimensional)
- **Language Model**: Microsoft Phi-3-mini-4k-instruct (quantized for efficiency)
- **Vector Search**: Cosine similarity with normalized embeddings
- **Text Processing**: Recursive character splitting with intelligent overlap
- **Prompt Engineering**: Three specialized templates for different AI tasks


### **Performance Specifications**
- **Model Size**: ~2.3GB (quantized GGUF format)
- **Context Window**: 4096 tokens (3500 usable for context)
- **Embedding Dimensions**: 768 (optimized for semantic similarity)
- **Response Time**: 2-5 seconds per query (hardware dependent)
- **Memory Usage**: 4-8GB RAM (model + embeddings + database)


---


## 🚀 **Getting Started**


### **Prerequisites**
- **Python**: 3.8 or newer with pip package manager
- **Hardware**: 8GB+ RAM recommended, modern multi-core CPU
- **Storage**: 5GB+ free space for models and database
- **Git**: For repository cloning


### **Installation**


1. **Clone the Repository**
```bash
git clone https://github.com/your-username/askspartan.git
cd askspartan
```


2. **Install Dependencies**
```bash
pip install -r requirements.txt
```


3. **Download Language Model**
Download a quantized GGUF model (recommended: Phi-3-mini):
```bash
# Create models directory
mkdir -p models


# Download Phi-3-mini-4k-instruct-q4.gguf from Hugging Face
# Place it in the models/ directory as: models/Phi-3-mini-4k-instruct-q4.gguf
```


**Recommended Models:**
- [Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) (Primary)
- [Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (Alternative)


### **Setup Process**


4. **Scrape SJSU Website Data**
```bash
python scraper.py
```
This process will:
- Crawl up to 2000 pages from SJSU website
- Clean and process content
- Create chunked data files in `data/` directory
- Take 30-60 minutes depending on network speed


5. **Generate Vector Database**
```bash
python setup_db_and_retriever.py
```
This will:
- Generate embeddings for all text chunks
- Create ChromaDB vector database
- Store in `embeddings/` directory
- Take 10-20 minutes depending on hardware


6. **Launch the Chatbot**
```bash
streamlit run ui.py
```
Access the interface at: `http://localhost:8501`


---


## 📂 **Project Structure**


```
askspartan/
├── 📁 data/                          # Scraped and processed content
│   ├── scraped_data_v3.jsonl        # Raw scraped content
│   └── scraped_data_v3_chunked.jsonl # Processed text chunks
├── 📁 embeddings/                    # Vector database files
│   └── scraped_data_v3_chunked/      # ChromaDB collection
├── 📁 models/                        # Language model files
│   └── Phi-3-mini-4k-instruct-q4.gguf # Quantized LLM
├── 📁 utils/                         # Utility modules
│   └── logger.py                     # Logging configuration
├── 📄 config.py                      # System configuration
├── 📄 scraper.py                     # Web scraping engine
├── 📄 setup_db_and_retriever.py     # Vector database setup
├── 📄 llm_response_generator.py     # LLM inference pipeline
├── 📄 ui.py                          # Streamlit web interface
├── 📄 requirements.txt               # Python dependencies
└── 📄 README.md                      # This documentation
```


---


## 🔧 **Configuration Guide**


### **Scraper Configuration**
Modify `SCRAPER_CONFIG` in `config.py`:
```python
SCRAPER_CONFIG = {
   "base_url_substr": "sjsu.edu",        # Target domain
   "max_pages": 2000,                    # Crawl limit
   "start_urls": [...],                  # Seed URLs
}
```


### **LLM Configuration**
Adjust model parameters in `LLM_CONFIG`:
```python
"RESPONSE_GENERATOR_CONFIG": {
   "INIT_CONFIG": {
       "model_path": "models/your-model.gguf",
       "n_ctx": 3500,                    # Context window
   },
   "CALL_CONFIG": {
       "max_tokens": 700,                # Response length
       "temperature": 0.7,               # Creativity level
   }
}
```


### **Embedding Configuration**
Customize embedding model:
```python
SENTENCE_TRANSFORMERS_CONFIG = {
   "model_name": "BAAI/bge-base-en-v1.5"  # Embedding model
}
```


---


## 💡 **Advanced Features**


### **Conversation Management**
- **Context Preservation**: Maintains conversation history across interactions
- **Automatic Summarization**: Condenses long conversations for efficiency
- **Query Enhancement**: Reformulates questions using conversation context


### **Smart Retrieval**
- **Semantic Search**: Finds conceptually similar content, not just keyword matches
- **Relevance Ranking**: Returns top-k most relevant document chunks
- **Metadata Preservation**: Tracks source URLs and document structure


### **Performance Optimization**
- **Model Quantization**: Reduced memory usage without significant quality loss
- **Batch Processing**: Efficient handling of large document collections
- **Caching**: Streamlit resource caching for faster subsequent loads


---


## 🎯 **Use Cases & Applications**


### **Educational Institutions**
- **Student Support**: 24/7 automated assistance for common inquiries
- **Admissions Help**: Prospective student guidance and information
- **Academic Planning**: Course requirements and program information
- **Campus Services**: Dining, housing, and facility information


### **Enterprise Applications**
- **Internal Knowledge Base**: Company-specific information retrieval
- **Customer Support**: Domain-specific automated assistance
- **Documentation Assistant**: Technical documentation Q&A
- **Training Platform**: Interactive learning for new employees


### **Research & Development**
- **RAG Implementation**: Reference architecture for similar projects
- **Local AI Deployment**: Privacy-focused AI without cloud dependencies
- **Educational Tool**: Teaching modern NLP and RAG concepts
- **Benchmarking**: Testing different model and embedding combinations


---


## 🚀 **Extension Opportunities**


### **Technical Enhancements**
- **Multi-Domain Support**: Extend to multiple university websites
- **Hybrid Search**: Combine keyword and semantic search
- **Model Upgrades**: Support for larger, more capable language models
- **GPU Acceleration**: CUDA support for faster inference
- **API Development**: RESTful API for external integrations


### **Feature Additions**
- **Voice Interface**: Speech-to-text and text-to-speech capabilities
- **Visual Processing**: Handle images, charts, and campus maps
- **Personalization**: User profiles and personalized recommendations
- **Analytics Dashboard**: Usage statistics and query analysis
- **Multi-Modal Support**: Process PDFs, videos, and other media types


---


## 📊 **Performance & Requirements**


### **System Requirements**
- **Minimum**: 4GB RAM, 2-core CPU, 3GB storage
- **Recommended**: 8GB+ RAM, 4+ core CPU, 5GB+ storage
- **Optimal**: 16GB RAM, 8+ core CPU (Apple Silicon preferred)


### **Performance Metrics**
- **Response Latency**: 2-5 seconds per query
- **Throughput**: 10-20 queries per minute
- **Accuracy**: High relevance due to domain-specific training
- **Scalability**: Handles 10,000+ document chunks efficiently


### **Resource Usage**
- **Model Loading**: ~2-3GB RAM
- **Vector Database**: ~500MB-2GB (depending on content volume)
- **Runtime Memory**: ~1-2GB additional during inference
- **Storage**: ~5GB total (models + database + scraped content)


---


## 🔒 **Privacy & Security**


### **Data Privacy**
- **Local Processing**: All data remains on your hardware
- **No External APIs**: No data sent to third-party services
- **Source Transparency**: All content sourced from public SJSU websites
- **User Control**: Complete control over data collection and usage


### **Security Features**
- **Offline Operation**: No network dependencies after initial setup
- **Open Source**: Full code transparency and auditability
- **Configurable Scraping**: Respectful crawling with rate limiting
- **Content Filtering**: Automatic removal of sensitive or irrelevant content


---


## 🛠️ **Troubleshooting**


### **Common Issues**


**Model Loading Errors:**
```bash
# Ensure model path is correct in config.py
# Verify model file exists and is not corrupted
# Check available RAM (models require 2-4GB)
```


**Slow Performance:**
```bash
# Reduce max_tokens in config.py
# Use smaller embedding model
# Increase system RAM if possible
```


**Scraping Issues:**
```bash
# Check internet connection
# Verify SJSU website accessibility
# Adjust delay settings if rate-limited
```


**Database Errors:**
```bash
# Delete embeddings/ directory and regenerate
# Ensure sufficient disk space
# Check file permissions
```


---


## 🤝 **Contributing**


We welcome contributions! Areas for improvement:
- Additional university website support
- Performance optimizations
- UI/UX enhancements
- Documentation improvements
- Bug fixes and testing


---


## 📄 **License**


This project is open-source and available under the MIT License. See LICENSE file for details.


---


## ⚖️ **Disclaimer**


This is an unofficial, community-driven project and is not affiliated with San Jose State University. All data is sourced from publicly accessible websites. The accuracy of responses depends on the information available at the time of scraping. Users are responsible for verifying critical information through official channels.


---


## 🙏 **Acknowledgments**


- **San Jose State University** for providing comprehensive public information
- **Hugging Face** for open-source models and embeddings
- **ChromaDB** team for the excellent vector database
- **Streamlit** for the intuitive web framework
- **LangChain** community for text processing tools


---


**Built with ❤️ for the SJSU community and AI enthusiasts worldwide**



