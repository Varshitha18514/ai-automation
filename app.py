
import uuid
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat/stream"


st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 AI Knowledge Assistant")
st.caption("Ask questions about your documents using RAG.")


# Sidebar settings
with st.sidebar:
    st.header("Settings")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        disabled=True,
    )

    st.write("Powered by Gemini + Qdrant + Hybrid Search")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Ask something about your documents...")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        try:
            # Get API key from environment
            import os
            from dotenv import load_dotenv

            load_dotenv()

            api_key = os.getenv("API_SECRET_KEY")

            if not api_key:
                raise ValueError("API_SECRET_KEY is missing from .env")

            # Connect to streaming API
            with requests.post(
                API_URL,
                json={
                    "question": question,
                    "session_id": st.session_state.session_id,
                },
                headers={
                    "Authorization": f"Bearer {api_key}"
                },
                stream=True,
                timeout=(10, 180),
            ) as response:

                response.raise_for_status()

                # Display chunks as they arrive
                for chunk in response.iter_content(
                    chunk_size=None,
                    decode_unicode=True,
                ):

                    if chunk:
                        if isinstance(chunk, bytes):
                            chunk = chunk.decode("utf-8")

                        full_response += chunk

                        response_placeholder.markdown(
                            full_response + "▌"
                        )

            response_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })

        except requests.exceptions.RequestException as error:
            st.error(f"API connection error: {error}")

        except Exception as error:
            st.error(f"Something went wrong: {error}")
