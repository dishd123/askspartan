import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import json


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


if __name__ == "__main__":
    """Run the web scraper on sjsu.edu domain."""
    scraper = WebScraper(
        base_url_substr="sjsu.edu", max_pages=200, output_file="scraped_data_1.jsonl"
    )
    scraper.scrape_via_queue(
        start_urls=[
            "https://www.sjsu.edu/",
            "https://catalog.sjsu.edu/content.php?catoid=13&navoid=4983",
        ]
    )
