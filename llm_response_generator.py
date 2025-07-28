import os
from typing import List
import config
from llama_cpp import Llama
from setup_db_and_retriever import ChromaDbRetriever
from utils.logger import get_logger

logger = get_logger(level="INFO")  # or "WARNING" or "ERROR" or "INFO" or "DEBUG"


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
        self.summary_generator_config = config.LLM_CONFIG["SUMMARY_GENERATOR_CONFIG"]
        self.query_reformulation_config = config.LLM_CONFIG[
            "QUERY_REFORMULATION_CONFIG"
        ]
        self.response_generator_config = config.LLM_CONFIG["RESPONSE_GENERATOR_CONFIG"]

        self.retriever = ChromaDbRetriever()
        self.retriever.load_an_existing_collection()
        self.llm = self._load_llm_model()

    def _load_llm_model(self):
        """
        Loads the quantized LLM model using llama-cpp-python.

        Returns:
            An instance of Llama from llama_cpp.
        """
        model_path = self.response_generator_config["INIT_CONFIG"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        return Llama(**self.response_generator_config["INIT_CONFIG"])

    def _summarize_coversation_history(self, coversation_history) -> str:
        """
        Summarizes the last k rounds of chat history using the same LLM.

        Args:
            coversation_history (List[Tuple[str, str]]): List of (user, assistant) messages.
            k (int): Number of recent turns to summarize.

        Returns:
            str: A concise summary of recent conversation.
        """
        if (
            not self.summary_generator_config["use_coversation_history"]
            or not coversation_history
            or len(coversation_history) == 0
        ):
            # No chat history to summarize
            return ""

        recent_history = coversation_history[
            -self.summary_generator_config["coversation_history_lookback"] :
        ]
        raw_dialogue = "\n".join(
            [f"USER: {u}\nCHATBOT: {a}" for u, a in recent_history]
        )

        summarization_prompt = self.summary_generator_config["prompt_template"].format(
            raw_dialogue=raw_dialogue
        )
        # logger.info(f"Summary prompt:\n{summarization_prompt}\n")
        summary_response = self.llm(
            summarization_prompt, **self.summary_generator_config["CALL_CONFIG"]
        )
        summarized_conversation_history = summary_response["choices"][0]["text"].strip()
        logger.info(
            f"Conversation history summary:\n{summarized_conversation_history}\n"
        )
        return summarized_conversation_history

    def _reformulate_query(
        self, query: str, summarized_conversation_history: str
    ) -> str:
        """
        Reformulates the user's query to include relevant context from the conversation summary.

        Args:
            query (str): The user's original question.
            summarized_conversation_history (str): Summary of recent k conversations.

        Returns:
            str: The reformulated query.
        """
        if not summarized_conversation_history:
            return query

        reformulation_prompt = self.query_reformulation_config[
            "prompt_template"
        ].format(summary=summarized_conversation_history, query=query)

        reformulation_response = self.llm(
            reformulation_prompt, **self.query_reformulation_config["CALL_CONFIG"]
        )
        reformulated_query = reformulation_response["choices"][0]["text"].strip()
        logger.info(f"Reformulated query:\n{reformulated_query}\n")
        return reformulated_query

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
        prompt = self.response_generator_config["prompt_template"].format(
            context=context, query=query
        )

        logger.info(f"Final prompt to LLM:\n{prompt}\n")
        return prompt

    def generate_response(self, query: str, coversation_history=None) -> str:
        """
        Generates a response to the user's query using the LLM and retrieved documents.

        Args:
            query (str): The user's question.
            coversation_history (List[Tuple[str, str]], optional): List of (user, assistant) messages for context.

        Returns:
            str: The generated response from the LLM.
        """

        summarized_conversation_history = self._summarize_coversation_history(
            coversation_history
        )

        reformulated_query = self._reformulate_query(
            query, summarized_conversation_history
        )

        retriever_results = self.retriever.query(reformulated_query)
        documents = retriever_results["documents"][0]
        prompt = self._build_prompt(query, documents)

        response = self.llm(prompt, **self.response_generator_config["CALL_CONFIG"])
        # logger.info(f"LLM raw response:\n{response["choices"][0]["text"]}\n")
        return response
