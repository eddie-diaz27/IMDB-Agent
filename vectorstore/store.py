"""
ChromaDB vector store for semantic search on movie overviews.
"""
import os
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from typing import List, Dict, Any, Optional

from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR, COLLECTION_NAME


def get_embedding_function():
    """Get the OpenAI embedding function for ChromaDB."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )


def initialize_vector_store(persist_directory: str = CHROMA_PERSIST_DIR) -> chromadb.PersistentClient:
    """Initialize ChromaDB with persistent storage.

    Args:
        persist_directory: Directory to store the vector database

    Returns:
        ChromaDB PersistentClient instance
    """
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the movie overviews collection.

    Args:
        client: ChromaDB client

    Returns:
        ChromaDB Collection for movie overviews
    """
    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def populate_collection(collection: chromadb.Collection, df: pd.DataFrame) -> None:
    """Populate the vector store with movie overviews.

    Args:
        collection: ChromaDB collection
        df: Preprocessed DataFrame with movie data
    """
    # Check if already populated
    if collection.count() >= len(df):
        print(f"Collection already has {collection.count()} documents. Skipping population.")
        return

    print(f"Populating collection with {len(df)} movie overviews...")

    # Prepare documents and metadata
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Skip rows with empty overview
        overview = row.get("Overview", "")
        if pd.isna(overview) or overview == "":
            continue

        documents.append(str(overview))

        metadata = {
            "Series_Title": str(row.get("Series_Title", "")),
            "Released_Year": int(row["Released_Year_cleaned"]) if pd.notna(row.get("Released_Year_cleaned")) else 0,
            "Genre": str(row.get("Genre", "")),
            "Director": str(row.get("Director", "")),
            "IMDB_Rating": float(row["IMDB_Rating"]) if pd.notna(row.get("IMDB_Rating")) else 0.0,
            "Meta_score": float(row["Meta_score"]) if pd.notna(row.get("Meta_score")) else 0.0,
            "Star1": str(row.get("Star1", "")),
            "Gross": int(row["Gross_cleaned"]) if pd.notna(row.get("Gross_cleaned")) else 0
        }
        metadatas.append(metadata)
        ids.append(f"movie_{idx}")

    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"Added documents {i} to {end_idx}...")

    print(f"Collection populated with {collection.count()} documents.")


def semantic_search(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 10,
    where_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Perform semantic search on movie overviews.

    Args:
        collection: ChromaDB collection
        query: Search query describing plot themes or elements
        n_results: Number of results to return
        where_filter: Optional metadata filter (e.g., {"Released_Year": {"$lt": 1990}})

    Returns:
        Search results with documents, metadatas, and distances
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_search_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format ChromaDB search results into a more usable format.

    Args:
        results: Raw ChromaDB query results

    Returns:
        List of formatted movie results
    """
    formatted = []

    if not results or not results.get("documents") or not results["documents"][0]:
        return formatted

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        distance = distances[i] if i < len(distances) else None
        formatted.append({
            "title": metadata.get("Series_Title", "Unknown"),
            "year": metadata.get("Released_Year", "Unknown"),
            "genre": metadata.get("Genre", "Unknown"),
            "director": metadata.get("Director", "Unknown"),
            "imdb_rating": metadata.get("IMDB_Rating", 0),
            "meta_score": metadata.get("Meta_score", 0),
            "overview": doc,
            "similarity_score": 1 - distance if distance else None
        })

    return formatted
