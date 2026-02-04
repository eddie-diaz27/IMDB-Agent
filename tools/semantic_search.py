"""
Semantic search tool for vector similarity search on movie overviews.
"""
from typing import Any, Optional, Type, Dict, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import chromadb

from vectorstore.store import semantic_search, format_search_results


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(description="Description of plot themes or elements to search for")
    year_before: Optional[int] = Field(default=None, description="Optional: only include movies released before this year")
    year_after: Optional[int] = Field(default=None, description="Optional: only include movies released after this year")
    min_rating: Optional[float] = Field(default=None, description="Optional: minimum IMDB rating")


class SemanticSearchTool(BaseTool):
    """Tool for semantic search on movie plot overviews."""

    name: str = "semantic_movie_search"
    description: str = """Use this tool to find movies based on plot themes, story elements, or content.
Good for:
- Finding movies "about" something (e.g., "movies about time travel")
- Thematic searches (e.g., "films with themes of redemption")
- Plot-based queries (e.g., "movies involving a heist")
- Finding movies with specific story elements (e.g., "movies with police involvement")
- Similarity searches based on plot descriptions

You can optionally filter by:
- year_before: Movies released before a specific year
- year_after: Movies released after a specific year
- min_rating: Minimum IMDB rating

Input should describe the plot elements or themes you're looking for."""

    args_schema: Type[BaseModel] = SemanticSearchInput
    collection: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, collection: chromadb.Collection, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'collection', collection)

    def _run(
        self,
        query: str,
        year_before: Optional[int] = None,
        year_after: Optional[int] = None,
        min_rating: Optional[float] = None
    ) -> str:
        """Execute semantic search."""
        collection = object.__getattribute__(self, 'collection')
        return execute_semantic_search(
            collection, query,
            year_before=year_before,
            year_after=year_after,
            min_rating=min_rating
        )


def execute_semantic_search(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 10,
    year_before: Optional[int] = None,
    year_after: Optional[int] = None,
    min_rating: Optional[float] = None
) -> str:
    """Execute a semantic search on movie overviews.

    Args:
        collection: ChromaDB collection
        query: Search query describing plot themes or elements
        n_results: Number of results to return
        year_before: Only include movies released before this year
        year_after: Only include movies released after this year
        min_rating: Minimum IMDB rating

    Returns:
        Formatted search results
    """
    # Build where filter
    where_filter = None
    conditions = []

    if year_before is not None:
        conditions.append({"Released_Year": {"$lt": year_before}})
    if year_after is not None:
        conditions.append({"Released_Year": {"$gt": year_after}})
    if min_rating is not None:
        conditions.append({"IMDB_Rating": {"$gte": min_rating}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # Execute search
    results = semantic_search(
        collection,
        query,
        n_results=n_results,
        where_filter=where_filter
    )

    # Format results
    formatted = format_search_results(results)

    if not formatted:
        return f"No movies found matching plot themes related to: {query}"

    # Build output string
    output_lines = [f"Found {len(formatted)} movies related to '{query}':\n"]

    for i, movie in enumerate(formatted, 1):
        line = f"{i}. **{movie['title']}** ({movie['year']})"
        line += f" - IMDB: {movie['imdb_rating']}/10"
        if movie['meta_score']:
            line += f", Meta: {int(movie['meta_score'])}"
        line += f"\n   Genre: {movie['genre']} | Director: {movie['director']}"
        line += f"\n   Plot: {movie['overview'][:200]}..."
        if movie['similarity_score']:
            line += f"\n   Relevance: {movie['similarity_score']:.2%}"
        output_lines.append(line)

    return "\n\n".join(output_lines)


def hybrid_search(
    collection: chromadb.Collection,
    df: Any,
    query: str,
    genre_filter: Optional[str] = None,
    year_range: Optional[tuple] = None
) -> List[Dict[str, Any]]:
    """Perform hybrid search combining semantic and structured filtering.

    First performs semantic search, then filters results by structured criteria.

    Args:
        collection: ChromaDB collection
        df: Pandas DataFrame with full movie data
        query: Semantic search query
        genre_filter: Optional genre to filter by
        year_range: Optional (start_year, end_year) tuple

    Returns:
        List of matching movies
    """
    # Build year filter for semantic search
    where_filter = None
    if year_range:
        where_filter = {
            "$and": [
                {"Released_Year": {"$gte": year_range[0]}},
                {"Released_Year": {"$lte": year_range[1]}}
            ]
        }

    # Get semantic search results
    results = semantic_search(collection, query, n_results=50, where_filter=where_filter)
    formatted = format_search_results(results)

    # Apply genre filter if specified
    if genre_filter and formatted:
        genre_lower = genre_filter.lower()
        formatted = [
            m for m in formatted
            if genre_lower in m.get("genre", "").lower()
        ]

    return formatted
