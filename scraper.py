import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import json
import config
from langchain.text_splitter import RecursiveCharacterTextSplitter


class WebScraper:
    """Web scraper for crawling and extracting text from web pages."""

    def __init__(
        self, base_url_substr, max_pages=100, output_file="scraped_data.jsonl"
    ):
        """Initialize the scraper with base URL substring, max pages, and output file."""
        self.base_url_substr = base_url_substr
        self.visited = set()
        self.max_pages = max_pages
        self.pages_scraped = 0
        self.buffer = []
        self.batch_size = 100
        self.output_file = output_file
        # URLs containing any of these substrings will be skipped
        self.skip_substrs = [
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".css",
            ".js",
            "mailto:",
            "tel:",
            "#",
            "javascript:",
            "blogs",
            "newsroom",
            "http://",
            "sjsuone",
            "events",
        ]
        self.tags_to_remove = [
            "script",
            "style",
            "header",
            "footer",
            "nav",
            "noscript",
            "form",
            "aside",
            "iframe",
            "svg",
        ]

    def flush_buffer(self):
        """Write buffered scraped data to output file and clear buffer."""
        if not self.buffer:
            return  # Nothing to flush
        with open(self.output_file, "a", encoding="utf-8") as f:
            for record in self.buffer:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(f"Flushed {len(self.buffer)} entries to {self.output_file}")
        self.buffer.clear()

    def add_to_buffer(self, url, text):
        """Add scraped text to buffer; flush if buffer is full."""
        if len(text) < 100:  # Skip very short texts
            return

        self.buffer.append({"url": url, "text": text})
        if len(self.buffer) >= self.batch_size:
            self.flush_buffer()

    def is_skippable_url(self, url):
        """Check if the URL should be skipped based on substrings."""
        return any(substr in url.lower() for substr in self.skip_substrs)

    def is_internal_url(self, url):
        """Check if the URL is internal to the base domain."""
        return self.base_url_substr in url

    def clean_text(self, soup):
        """Remove unwanted tags and elements, then extract clean text from HTML soup."""
        # Remove scripts, styles, headers, footers, navs
        for tag in soup.find_all(self.tags_to_remove):
            tag.decompose()

        # Remove specific unwanted elements such as navigation bars, return-to-top links, FB/Twitter links, etc.
        for tag in soup.select(
            "a.navbar, span.sr-only, td.block_footer_rb, img.return-to"
        ):
            tag.decompose()

        for tag in soup.find_all("a", href="javascript:void(0);"):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def scrape_page(self, url):
        """Scrape a single page, returning cleaned text and internal links."""
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return "", []

        soup = BeautifulSoup(response.text, "html.parser")
        text = self.clean_text(soup)

        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        internal_links = [link for link in links if self.is_internal_url(link)]
        return text, internal_links

    def scrape_via_queue(self, start_urls):
        """Scrape pages using a queue (BFS) starting from the given URL."""
        queue = deque(start_urls)

        while queue and self.pages_scraped < self.max_pages:
            url = queue.popleft()
            if url in self.visited or self.is_skippable_url(url):
                continue

            self.visited.add(url)
            self.pages_scraped += 1

            time.sleep(1)  # Be polite to the server
            text, links = self.scrape_page(url)
            print(f"---Page {self.pages_scraped}: {url} ---{text}")
            self.add_to_buffer(url, text)
            queue.extend(links)

        self.flush_buffer()  # Flush any remaining data in buffer

    def create_chunks():
        """
        Splits input text into smaller chunks and writes them to an output file.
        Each chunk includes metadata for later retrieval.
        """

        CHUNK_SIZE = config.TEXT_SPLITTER_CONFIG[
            "chunk_size"
        ]  # characters based on token limit of 512
        CHUNK_OVERLAP = CHUNK_SIZE // 10  # characters

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        input_path = config.TEXT_SPLITTER_CONFIG["input_data_file"]
        output_path = config.TEXT_SPLITTER_CONFIG["output_data_file"]

        with open(input_path, "r", encoding="utf-8") as infile, open(
            output_path, "w", encoding="utf-8"
        ) as outfile:
            doc_index = 0
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["text"]
                url = data["url"]

                chunks = text_splitter.split_text(text)

                chunk_index = 0
                for chunk in chunks:
                    chunked_data = {
                        "metadata": {
                            "url": url,
                            "doc_index": doc_index,
                            "chunk_index": chunk_index,
                            "id": f"{doc_index}_{chunk_index}",
                        },
                        "text": chunk,
                    }
                    outfile.write(json.dumps(chunked_data, ensure_ascii=False) + "\n")

                    chunk_index += 1

                doc_index += 1


if __name__ == "__main__":
    """Run the web scraper on sjsu.edu domain."""
    scraper = WebScraper(
        base_url_substr=config.SCRAPER_CONFIG["base_url_substr"],
        max_pages=config.SCRAPER_CONFIG["max_pages"],
        output_file=config.SCRAPER_CONFIG["output_file"],
    )
    scraper.scrape_via_queue(
        start_urls=config.SCRAPER_CONFIG["start_urls"],
    )
    print("Creating chunks from scraped data...")
    scraper.create_chunks()
