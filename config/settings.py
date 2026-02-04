"""
Configuration settings for the IMDB Movie Agent.
Modify this file to change LLM models and other settings.

Setup:
1. Copy .env.example to .env
2. Add your OpenAI API key to .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Key
# Set via .env file (copy .env.example to .env and add your key)
OPENAI_API_KEY = os.getenv("API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: API_KEY not found in .env file.")
    print("Please copy .env.example to .env and add your OpenAI API key.")

# LLM Model Configuration
# Options: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
LLM_MODEL = "gpt-4o-mini"

# Embedding Model
EMBEDDING_MODEL = "text-embedding-3-small"

# Vector Store Settings
CHROMA_PERSIST_DIR = "vectorstore/chroma_db"
COLLECTION_NAME = "movie_overviews"

# Agent Settings
MAX_ITERATIONS = 10
MEMORY_WINDOW_SIZE = 10

# Data Settings
DATASET_PATH = "imdb_dataset/imdb_top_1000.csv"
