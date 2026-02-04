"""
IMDB Movie Agent - Streamlit Application

A conversational AI agent that answers questions about the IMDB Top 1000 movies.
"""
import streamlit as st
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OPENAI_API_KEY, DATASET_PATH, CHROMA_PERSIST_DIR, LLM_MODEL
from data.preprocessor import load_and_preprocess_data
from vectorstore.store import initialize_vector_store, get_or_create_collection, populate_collection
from agents.movie_agent import create_movie_agent


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def initialize_agent():
    """Initialize the movie agent with data and vector store."""
    with st.spinner("Loading movie database..."):
        # Load and preprocess data
        df = load_and_preprocess_data(DATASET_PATH)
        st.session_state.df = df

    with st.spinner("Setting up vector store (this may take a moment on first run)..."):
        # Initialize vector store
        client = initialize_vector_store(CHROMA_PERSIST_DIR)
        collection = get_or_create_collection(client)

        # Populate if needed
        if collection.count() < len(df):
            populate_collection(collection, df)

        st.session_state.collection = collection

    with st.spinner("Initializing AI agent..."):
        # Create agent
        agent = create_movie_agent(df, collection)
        st.session_state.agent = agent

    st.session_state.initialized = True
    return agent


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="IMDB Movie Agent",
        page_icon="🎬",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("🎬 IMDB Movie Expert")
    st.markdown("*Ask me anything about the top 1000 IMDB movies!*")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI-powered assistant can answer questions about the IMDB Top 1000 movies dataset.

        **Capabilities:**
        - Look up movie details
        - Find top movies by rating, score, or earnings
        - Search movies by plot themes
        - Summarize director filmographies
        - Get movie recommendations
        """)

        st.divider()

        st.subheader("Example Questions")
        example_questions = [
            "When did The Matrix release?",
            "Top 5 movies of 2019 by meta score",
            "Top 7 comedy movies between 2010-2020 by IMDB rating",
            "Movies before 1990 with police involvement in the plot",
            "Summarize Steven Spielberg's top sci-fi movie plots",
            "Al Pacino movies with IMDB rating above 8",
            "Recommend movies similar to Inception"
        ]

        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
                st.session_state.example_query = q

        st.divider()

        st.subheader("Settings")
        st.info(f"Model: {LLM_MODEL}")

        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_memory()
            st.rerun()

        st.divider()

        st.caption("Powered by OpenAI + LangChain + ChromaDB")

    # Check for API key
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.code("API_KEY=your_openai_api_key_here")
        return

    # Initialize agent if needed
    if not st.session_state.initialized:
        try:
            initialize_agent()
            st.success("✅ Movie database loaded successfully!")
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            st.exception(e)
            return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle example query from sidebar
    if "example_query" in st.session_state:
        prompt = st.session_state.example_query
        del st.session_state.example_query

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask me about movies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error processing your request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
