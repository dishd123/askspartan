import chromadb
from sentence_transformers import SentenceTransformer
import json
import config


class ChromaDbRetriever:
    """
    Handles creation, loading, and querying of a ChromaDB collection using sentence embeddings.
    """

    def __init__(self):
        """
        Initializes the ChromaDB retriever with configuration and sets up the embedding model and ChromaDB client.
        """
        self.CHROMA_DB_PATH = config.CHROMA_CONFIG["chroma_db_dir"]
        self.COLLECTION_NAME = config.CHROMA_CONFIG["collection_name"]
        self.EMB_MODEL_NAME = config.SENTENCE_TRANSFORMERS_CONFIG["model_name"]
        self.client = chromadb.PersistentClient(path=self.CHROMA_DB_PATH)
        self.emb_model = SentenceTransformer(self.EMB_MODEL_NAME)

    def _embed_text(self, text):
        """
        Generates embeddings for the input text using the configured model.
        """
        return self.emb_model.encode(text, normalize_embeddings=True)

    def create_collection(self):
        """
        Creates a new ChromaDB collection. It throws an error if the collection already exists.
        This method is idempotent; it will not create a new collection if one already exists.
        """
        self.collection = self.client.create_collection(name=self.COLLECTION_NAME)

        data_path = config.CHROMA_CONFIG["input_data_file"]
        counter = 0

        with open(data_path, "r", encoding="utf-8") as infile:

            for line in infile:
                counter += 1
                if counter % 100 == 0:
                    print(f"Processed {counter} chunks...")
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["text"]
                metadata = data["metadata"]
                id_ = metadata["id"]

                embedding = self._embed_text(text)

                self.collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[id_],
                    embeddings=[embedding],
                )

    def load_an_existing_collection(self):
        """
        Loads an existing ChromaDB collection.
        """
        self.collection = self.client.get_collection(name=self.COLLECTION_NAME)

    def query(self, query_text, n_results=5):
        """
        Queries the ChromaDB collection for the most relevant documents based on the input text.
        Returns a list of document IDs and their corresponding metadata.
        """
        query_embedding = self._embed_text(query_text)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        return results


if __name__ == "__main__":
    """Main entry point to create a ChromaDB collection."""
    print(
        f"Creating ChromaDB collection with the following configuration:\n{json.dumps(config.CHROMA_CONFIG, indent=4)}"
    )
    retriever = ChromaDbRetriever()
    retriever.create_collection()
    print(
        "ChromaDB collection created successfully.\n"
        "You can now load the collection and query it using the retriever instance.\n"
        "Example usage:\n"
        "retriever.load_an_existing_collection()\n"
        "retriever.query('Your query here')\n"
        "Make sure to replace 'Your query here' with your actual query text.\n"
        "You can also use the retriever instance in your scripts to interact with the ChromaDB collection."
    )
