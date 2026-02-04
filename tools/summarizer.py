"""
Summarization tool for synthesizing information from multiple movies.
"""
from typing import Any, Type, List, Dict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
import pandas as pd

from config.settings import OPENAI_API_KEY, LLM_MODEL


class SummarizerInput(BaseModel):
    """Input schema for summarizer tool."""
    request: str = Field(description="Description of what to summarize (e.g., 'Summarize Steven Spielberg's sci-fi movie plots')")


class SummarizerTool(BaseTool):
    """Tool for summarizing movie plots and synthesizing information."""

    name: str = "movie_summarizer"
    description: str = """Use this tool to summarize or synthesize information from multiple movies.
Good for:
- Summarizing a director's movie plots (e.g., "Summarize Spielberg's sci-fi movies")
- Comparing themes across movies
- Synthesizing information about an actor's filmography
- Creating overviews of movie collections by genre, era, or theme

This tool first retrieves relevant movies based on your request, then synthesizes a coherent summary.

Input should describe what you want summarized."""

    args_schema: Type[BaseModel] = SummarizerInput
    df: pd.DataFrame = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'df', df)

    def _run(self, request: str) -> str:
        """Execute the summarization."""
        df = object.__getattribute__(self, 'df')
        return summarize_movies(df, request)


def summarize_movies(df: pd.DataFrame, request: str) -> str:
    """Summarize movie plots based on the request.

    Args:
        df: Movie DataFrame
        request: Description of what to summarize

    Returns:
        Synthesized summary
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=OPENAI_API_KEY
    )

    # First, identify which movies to summarize
    filter_prompt = f"""Analyze this request and extract the filter criteria:
Request: {request}

Return a JSON object with these fields (use null if not specified):
- director: director name if mentioned
- genre: genre if mentioned
- actor: actor name if mentioned
- year_start: start year if a range is mentioned
- year_end: end year if a range is mentioned
- limit: max number of movies (default 5)
- sort_by: "rating" or "metascore" or "year" (default "rating")

Only return the JSON, no other text."""

    try:
        filter_response = llm.invoke(filter_prompt)
        import json

        # Clean up JSON response
        filter_text = filter_response.content.strip()
        if filter_text.startswith("```json"):
            filter_text = filter_text[7:]
        if filter_text.startswith("```"):
            filter_text = filter_text[3:]
        if filter_text.endswith("```"):
            filter_text = filter_text[:-3]

        filters = json.loads(filter_text)
    except Exception:
        # Default filters
        filters = {"limit": 5, "sort_by": "rating"}

    # Apply filters to get relevant movies
    filtered_df = df.copy()

    if filters.get("director"):
        director = filters["director"]
        filtered_df = filtered_df[
            filtered_df["Director"].str.contains(director, case=False, na=False)
        ]

    if filters.get("genre"):
        genre = filters["genre"]
        filtered_df = filtered_df[
            filtered_df["Genre"].str.contains(genre, case=False, na=False)
        ]

    if filters.get("actor"):
        actor = filters["actor"]
        filtered_df = filtered_df[
            filtered_df["Stars_str"].str.contains(actor, case=False, na=False)
        ]

    if filters.get("year_start"):
        filtered_df = filtered_df[
            filtered_df["Released_Year_cleaned"] >= filters["year_start"]
        ]

    if filters.get("year_end"):
        filtered_df = filtered_df[
            filtered_df["Released_Year_cleaned"] <= filters["year_end"]
        ]

    # Sort
    sort_by = filters.get("sort_by", "rating")
    if sort_by == "metascore":
        filtered_df = filtered_df.dropna(subset=["Meta_score"]).sort_values(
            "Meta_score", ascending=False
        )
    elif sort_by == "year":
        filtered_df = filtered_df.sort_values("Released_Year_cleaned", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("IMDB_Rating", ascending=False)

    # Limit results
    limit = filters.get("limit", 5)
    filtered_df = filtered_df.head(limit)

    if len(filtered_df) == 0:
        return f"No movies found matching the criteria in: {request}"

    # Prepare movie data for summarization
    movies_info = []
    for _, row in filtered_df.iterrows():
        movie_info = {
            "title": row["Series_Title"],
            "year": row.get("Released_Year_cleaned", row.get("Released_Year")),
            "genre": row["Genre"],
            "rating": row["IMDB_Rating"],
            "director": row["Director"],
            "overview": row["Overview"]
        }
        movies_info.append(movie_info)

    # Create summary
    movies_text = "\n\n".join([
        f"**{m['title']}** ({m['year']}) - {m['rating']}/10\nDirector: {m['director']}\nGenre: {m['genre']}\nPlot: {m['overview']}"
        for m in movies_info
    ])

    summary_prompt = f"""Based on the following movies, create a comprehensive summary that synthesizes their plot themes and stories.

Request: {request}

Movies to summarize:
{movies_text}

Provide:
1. A brief overview of the common themes or elements across these movies
2. Individual highlights of each movie's plot
3. Any interesting patterns or observations

Format your response clearly with sections."""

    summary_response = llm.invoke(summary_prompt)

    # Add the movie list header
    movie_list = "\n".join([
        f"- **{m['title']}** ({m['year']}) - IMDB: {m['rating']}/10"
        for m in movies_info
    ])

    return f"**Movies analyzed:**\n{movie_list}\n\n---\n\n{summary_response.content}"
