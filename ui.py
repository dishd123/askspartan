import streamlit as st
from llm_response_generator import LLMResponseGenerator


# Example usage:
#     generator = LLMResponseGenerator()
#     response = generator.generate_response("What is the capital of France?")
#     print(response)
#     print(response["choices"][0]["text"])


# Initialize LLM generator (cached to avoid reloading on every rerun)
@st.cache_resource
def setup_components():
    generator = LLMResponseGenerator()
    return generator


generator = setup_components()

st.title("🧠 AskSpartan - Local SJSU AI Chatbot")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about SJSU...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant message
    with st.chat_message("assistant"):
        response = generator.generate_response(
            query=user_input,
            coversation_history=st.session_state.conversation_history,
        )

        response_simplified = response["choices"][0]["text"]
        st.markdown(f"**AskSpartan:** {response_simplified}")

    # Update history
    st.session_state.conversation_history.append((user_input, response_simplified))
    st.session_state.messages.append(
        {"role": "assistant", "content": response_simplified}
    )


# streamlit run ask_spartan_app.py
