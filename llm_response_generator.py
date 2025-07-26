import os
from typing import List
import config
from llama_cpp import Llama
from setup_db_and_retriever import ChromaDbRetriever


class LLMResponseGenerator:
    """
    LLMResponseGenerator handles local inference using a quantized LLM model
    via llama-cpp-python. It integrates with a retriever (e.g., ChromaDbRetriever)
    to generate context-aware answers from retrieved documents.

    Example usage:
        generator = LLMResponseGenerator()
        response = generator.generate_response("What is the capital of France?")
        print(response)
        print(response["choices"][0]["text"])

    """

    def __init__(self):
        """
        Initialize the LLM pipeline and retriever.
        """
        self.init_config = config.LLM_RESPONSE_GENERATOR_CONFIG["INIT_CONFIG"]
        self.call_config = config.LLM_RESPONSE_GENERATOR_CONFIG["CALL_CONFIG"]
        self.prompt_template = config.LLM_RESPONSE_GENERATOR_CONFIG["prompt_template"]

        self.retriever = ChromaDbRetriever()
        self.retriever.load_an_existing_collection()
        self.llm = self._load_llm_model()

    def _load_llm_model(self):
        """
        Loads the quantized LLM model using llama-cpp-python.

        Returns:
            An instance of Llama from llama_cpp.
        """
        model_path = self.init_config["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        return Llama(**self.init_config)

    def _build_prompt(self, query: str, documents: List[str]) -> str:
        """
        Builds a prompt for the LLM using the retrieved documents and the user's query.

        Args:
            query (str): The user's question.
            documents (List[str]): The context documents retrieved by the retriever.

        Returns:
            str: The formatted prompt for the LLM.
        """
        context = "\n\n".join(documents)
        return self.prompt_template.format(context=context, query=query)

    def generate_response(self, query: str) -> str:
        """
        Generates a response to the user's query using the LLM and retrieved documents.

        Args:
            query (str): The user's question.

        Returns:
            str: The generated response from the LLM.
        """

        retriever_results = self.retriever.query(query)
        documents = retriever_results["documents"][0]
        prompt = self._build_prompt(query, documents)

        response = self.llm(prompt, **self.call_config)
        return response
