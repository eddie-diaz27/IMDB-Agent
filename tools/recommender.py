"""
Movie recommendation tool for suggesting similar films.
"""
from typing import Any, Type, List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
import pandas as pd
import chromadb

from config.settings import OPENAI_API_KEY, LLM_MODEL
from vectorstore.store import semantic_search, format_search_results


class RecommenderInput(BaseModel):
    """Input schema for recommender tool."""
    movie_title: Optional[str] = Field(default=None, description="Title of a movie to find similar movies to")
    preferences: Optional[str] = Field(default=None, description="Description of what kind of movies the user likes")


class RecommenderTool(BaseTool):
    """Tool for recommending similar movies."""

    name: str = "movie_recommender"
    description: str = """Use this tool to recommend movies similar to a given movie or based on user preferences.
Good for:
- Finding movies similar to a specific film
- Recommending movies based on user preferences (genre, themes, etc.)
- Suggesting movies with similar ratings or themes

Input should be either:
- A movie title to find similar movies to
- A description of preferences (e.g., "movies with high ratings and sci-fi themes")"""

    args_schema: Type[BaseModel] = RecommenderInput
    df: pd.DataFrame = Field(default=None, exclude=True)
    collection: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, df: pd.DataFrame, collection: chromadb.Collection, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'df', df)
        object.__setattr__(self, 'collection', collection)

    def _run(
        self,
        movie_title: Optional[str] = None,
        preferences: Optional[str] = None
    ) -> str:
        """Generate movie recommendations."""
        df = object.__getattribute__(self, 'df')
        collection = object.__getattribute__(self, 'collection')
        return recommend_movies(df, collection, movie_title, preferences)


def recommend_movies(
    df: pd.DataFrame,
    collection: chromadb.Collection,
    movie_title: Optional[str] = None,
    preferences: Optional[str] = None
) -> str:
    """Generate movie recommendations.

    Args:
        df: Movie DataFrame
        collection: ChromaDB collection
        movie_title: Optional movie title to find similar films to
        preferences: Optional user preferences description

    Returns:
        Formatted recommendations
    """
    recommendations = []

    if movie_title:
        # Find the source movie
        mask = df["Series_Title"].str.lower().str.contains(movie_title.lower(), na=False)
        source_movies = df[mask]

        if len(source_movies) == 0:
            return f"Could not find a movie matching '{movie_title}'. Please check the spelling."

        source = source_movies.iloc[0]

        # Get similar movies using semantic search on the overview
        results = semantic_search(collection, source["Overview"], n_results=11)
        similar = format_search_results(results)

        # Remove the source movie from results
        similar = [m for m in similar if m["title"].lower() != source["Series_Title"].lower()][:5]

        # Also find movies with similar ratings and genre
        source_rating = source["IMDB_Rating"]
        source_genre = source["Genre_list"][0] if source["Genre_list"] else ""

        rating_similar = df[
            (df["IMDB_Rating"] >= source_rating - 0.5) &
            (df["IMDB_Rating"] <= source_rating + 0.5) &
            (df["Series_Title"] != source["Series_Title"])
        ]

        if source_genre:
            rating_similar = rating_similar[
                rating_similar["Genre"].str.contains(source_genre, case=False, na=False)
            ]

        rating_similar = rating_similar.nlargest(3, "No_of_Votes")

        # Build response
        output = f"**Recommendations based on '{source['Series_Title']}' ({source['Released_Year_cleaned']})**\n"
        output += f"Original: {source['Genre']} | IMDB: {source['IMDB_Rating']}/10\n\n"

        output += "**Similar by plot/themes:**\n"
        for i, movie in enumerate(similar[:5], 1):
            output += f"{i}. **{movie['title']}** ({movie['year']}) - {movie['imdb_rating']}/10\n"
            output += f"   {movie['genre']}\n"

        output += "\n**Similar by genre and rating:**\n"
        for i, (_, row) in enumerate(rating_similar.head(3).iterrows(), 1):
            output += f"{i}. **{row['Series_Title']}** ({row['Released_Year_cleaned']}) - {row['IMDB_Rating']}/10\n"
            output += f"   {row['Genre']}\n"

    elif preferences:
        # Use LLM to understand preferences and search
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        # Parse preferences
        parse_prompt = f"""Analyze these movie preferences and extract key criteria:
Preferences: {preferences}

Return a JSON with:
- search_query: a query to find similar plot themes
- min_rating: minimum IMDB rating (default 7.0)
- genres: list of preferred genres (if any)
- era: preferred era like "80s", "modern", "classic" (if any)

Only return JSON, no other text."""

        try:
            import json
            response = llm.invoke(parse_prompt)
            criteria = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        except Exception:
            criteria = {"search_query": preferences, "min_rating": 7.0}

        # Search by plot
        where_filter = None
        if criteria.get("min_rating"):
            where_filter = {"IMDB_Rating": {"$gte": criteria["min_rating"]}}

        results = semantic_search(
            collection,
            criteria.get("search_query", preferences),
            n_results=10,
            where_filter=where_filter
        )
        similar = format_search_results(results)

        # Filter by genre if specified
        if criteria.get("genres"):
            genres = [g.lower() for g in criteria["genres"]]
            similar = [
                m for m in similar
                if any(g in m.get("genre", "").lower() for g in genres)
            ]

        output = f"**Recommendations based on your preferences:**\n"
        output += f"'{preferences}'\n\n"

        for i, movie in enumerate(similar[:7], 1):
            output += f"{i}. **{movie['title']}** ({movie['year']}) - {movie['imdb_rating']}/10\n"
            output += f"   {movie['genre']} | Director: {movie['director']}\n"
            output += f"   {movie['overview'][:150]}...\n\n"

    else:
        # Default: recommend top-rated diverse movies
        output = "**Top Recommended Movies (diverse selection):**\n\n"

        # Get top movies from different genres
        genres = ["Drama", "Action", "Comedy", "Sci-Fi", "Crime"]
        for genre in genres:
            genre_movies = df[df["Genre"].str.contains(genre, case=False, na=False)]
            if len(genre_movies) > 0:
                top = genre_movies.nlargest(1, "IMDB_Rating").iloc[0]
                output += f"**{genre}:** {top['Series_Title']} ({top['Released_Year_cleaned']}) - {top['IMDB_Rating']}/10\n"

    # Add reasoning
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.5, api_key=OPENAI_API_KEY)
    reasoning_prompt = f"""Based on these movie recommendations, add a brief (2-3 sentences) explanation of why these movies were recommended and what common elements they share:

{output}

Keep it conversational and insightful."""

    reasoning = llm.invoke(reasoning_prompt)
    output += f"\n---\n**Why these recommendations:** {reasoning.content}"

    return output
