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
        self.summary_generator_config = config.LLM_RESPONSE_GENERATOR_CONFIG[
            "SUMMARY_GENERATOR_CONFIG"
        ]
        self.use_chat_history = config.LLM_RESPONSE_GENERATOR_CONFIG.get(
            "use_chat_history", True
        )
        self.chat_history_lookback = config.LLM_RESPONSE_GENERATOR_CONFIG.get(
            "chat_history_lookback", 3
        )
        self.prompt_template = config.LLM_RESPONSE_GENERATOR_CONFIG["prompt_template"]
        self.summary_template = config.LLM_RESPONSE_GENERATOR_CONFIG["summary_template"]

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

    def _summarize_chat_history(self, chat_history) -> str:
        """
        Summarizes the last k rounds of chat history using the same LLM.

        Args:
            chat_history (List[Tuple[str, str]]): List of (user, assistant) messages.
            k (int): Number of recent turns to summarize.

        Returns:
            str: A concise summary of recent conversation.
        """

        recent_history = chat_history[-self.chat_history_lookback :]
        raw_dialogue = "\n".join(
            [f"USER_INPUT: {u}\nCHATBOT_RESPONSE: {a}" for u, a in recent_history]
        )

        summary_prompt = self.summary_template.format(raw_dialogue=raw_dialogue)
        # print(f"Summary prompt:\n{summary_prompt}\n")
        summary_response = self.llm(summary_prompt, **self.summary_generator_config)
        summary_text = summary_response["choices"][0]["text"].strip()
        # print(f"Chat history summary:\n{summary_text}\n")
        return summary_text

    def _build_prompt(self, query: str, documents: List[str], chat_history) -> str:
        """
        Builds a prompt for the LLM using the retrieved documents and the user's query.

        Args:
            query (str): The user's question.
            documents (List[str]): The context documents retrieved by the retriever.

        Returns:
            str: The formatted prompt for the LLM.
        """

        context = "\n\n".join(documents)
        prompt = self.prompt_template.format(context=context, query=query)

        if self.use_chat_history and chat_history:
            chat_history_summary = self._summarize_chat_history(chat_history)
            prompt = (
                f"### Summary of previous chats:\n{chat_history_summary}\n\n" + prompt
            )
        # print(f"Final prompt to LLM:\n{prompt}\n")
        return prompt

    def generate_response(self, query: str, chat_history=None) -> str:
        """
        Generates a response to the user's query using the LLM and retrieved documents.

        Args:
            query (str): The user's question.

        Returns:
            str: The generated response from the LLM.
        """

        retriever_results = self.retriever.query(query)
        documents = retriever_results["documents"][0]
        prompt = self._build_prompt(query, documents, chat_history)

        response = self.llm(prompt, **self.call_config)
        # print(f"LLM raw response:\n{response["choices"][0]["text"]}\n")
        return response
