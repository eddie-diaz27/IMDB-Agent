"""
Structured query tool for Pandas-based filtering and aggregation.
"""
import json
import pandas as pd
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from config.settings import OPENAI_API_KEY, LLM_MODEL


class StructuredQueryInput(BaseModel):
    """Input schema for structured query tool."""
    query: str = Field(description="Natural language query about movies")


class StructuredQueryTool(BaseTool):
    """Tool for executing structured queries on the movie dataset."""

    name: str = "structured_movie_query"
    description: str = """Use this tool for questions about specific movies or filtering movies by attributes.
Good for:
- Looking up specific movie details (release year, rating, director, cast)
- Finding top N movies by rating, score, or gross earnings
- Filtering movies by year, genre, rating thresholds
- Aggregating data (e.g., directors with multiple high-grossing films)
- Comparing movies by specific attributes

Do NOT use this for:
- Questions about plot content or themes (use semantic_movie_search instead)
- Questions requiring understanding of movie descriptions

Input should be a natural language question about movie data."""

    args_schema: Type[BaseModel] = StructuredQueryInput
    df: pd.DataFrame = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'df', df)

    def _run(self, query: str) -> str:
        """Execute the structured query."""
        df = object.__getattribute__(self, 'df')
        return execute_structured_query(df, query)


def execute_structured_query(df: pd.DataFrame, query: str) -> str:
    """Execute a structured query using LLM to generate pandas code.

    Args:
        df: The preprocessed movie DataFrame
        query: Natural language query

    Returns:
        Formatted query results
    """
    # Create LLM for query parsing
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    # Get column info for the prompt
    columns_info = """
Available columns:
- Series_Title: Movie name (string)
- Released_Year_cleaned: Year of release (int, use this for year comparisons)
- Certificate: Age rating (string: U, UA, A, PG, R, etc.)
- Runtime_mins: Duration in minutes (int)
- Genre: Genre(s) as string (e.g., "Action, Crime, Drama")
- Genre_list: Genre(s) as list (e.g., ["Action", "Crime", "Drama"])
- IMDB_Rating: User rating 1-10 (float)
- Meta_score: Critic score 1-100 (float, may have NaN values)
- Director: Director name (string)
- Star1, Star2, Star3, Star4: Lead actors (strings)
- Stars_str: All stars as comma-separated string
- No_of_Votes: Number of IMDB votes (int)
- Gross_cleaned: Box office earnings in USD (int, may have None values)
- Overview: Plot summary (string)
"""

    # Prompt for generating pandas code
    prompt = f"""You are a data analyst. Given a user question about movies, generate Python pandas code to answer it.

{columns_info}

Important notes:
- The DataFrame is named 'df'
- For genre filtering, use: df[df['Genre'].str.contains('Comedy', case=False, na=False)]
- For actor searches, use: df[df['Stars_str'].str.contains('Actor Name', case=False, na=False)]
- For lead actor only: df[df['Star1'].str.contains('Actor Name', case=False, na=False)]
- Handle NaN values appropriately with .dropna() or fillna() when needed
- Always sort results in a meaningful way
- Limit results to a reasonable number (default 10 for top-N queries)
- Return result as a DataFrame or Series

User question: {query}

Respond with ONLY the Python code, no explanations. The code should produce a result that answers the question.
The last line should be the expression that produces the result (no need to print or return statement).
"""

    try:
        response = llm.invoke(prompt)
        code = response.content.strip()

        # Clean up code - remove markdown formatting if present
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # Execute the generated code safely
        local_vars = {"df": df.copy(), "pd": pd, "result": None}

        # Wrap code to capture result
        code_lines = code.strip().split('\n')
        # Assign last expression to result variable
        if code_lines:
            code_lines[-1] = f"result = {code_lines[-1]}"
        wrapped_code = '\n'.join(code_lines)

        exec(wrapped_code, {"pd": pd}, local_vars)
        result = local_vars.get("result")

        if result is None:
            return handle_fallback_query(df, query)

        return format_query_result(result, query)

    except Exception as e:
        # Fallback: try direct pattern matching for common queries
        fallback_result = handle_fallback_query(df, query)
        if "couldn't process" not in fallback_result:
            return fallback_result
        return f"Error processing query: {str(e)}. Please try rephrasing."


def format_query_result(result: Any, query: str) -> str:
    """Format query results for display.

    Args:
        result: Query result (DataFrame, Series, or scalar)
        query: Original query for context

    Returns:
        Formatted string result
    """
    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return "No movies found matching your criteria."

        # Format as a nice list
        output_lines = []
        for idx, row in result.head(20).iterrows():
            title = row.get("Series_Title", "Unknown")
            year = row.get("Released_Year_cleaned", row.get("Released_Year", "?"))
            rating = row.get("IMDB_Rating", "?")
            meta = row.get("Meta_score", "N/A")
            genre = row.get("Genre", "?")
            director = row.get("Director", "?")
            gross = row.get("Gross_cleaned", None)

            line = f"**{title}** ({year})"
            line += f" - IMDB: {rating}/10"
            if pd.notna(meta) and meta != "N/A":
                line += f", Meta: {int(meta)}"
            line += f"\n  Genre: {genre} | Director: {director}"
            if gross and pd.notna(gross):
                line += f" | Gross: ${gross:,}"
            output_lines.append(line)

        return "\n\n".join(output_lines)

    elif isinstance(result, pd.Series):
        if result.name == "Series_Title" or len(result) == 1:
            return str(result.values[0]) if len(result) == 1 else "\n".join(str(v) for v in result.values)
        return result.to_string()

    else:
        return str(result)


def handle_fallback_query(df: pd.DataFrame, query: str) -> str:
    """Handle common query patterns with direct logic.

    Args:
        df: Movie DataFrame
        query: User query

    Returns:
        Query result or error message
    """
    import re
    query_lower = query.lower()
    filtered_df = df.copy()

    # Pattern: "When did [movie] release?"
    if "release" in query_lower or "when" in query_lower:
        for _, row in df.iterrows():
            title = str(row["Series_Title"]).lower()
            if title in query_lower:
                year = row.get("Released_Year_cleaned", row.get("Released_Year"))
                return f"**{row['Series_Title']}** was released in {year}."

    # Extract number for "top N" queries
    n_match = re.search(r"top\s+(\d+)", query_lower)
    n = int(n_match.group(1)) if n_match else 10

    # Genre filter
    genres = ["comedy", "drama", "action", "horror", "thriller", "sci-fi",
              "romance", "adventure", "crime", "fantasy", "animation", "mystery"]
    for genre in genres:
        if genre in query_lower:
            filtered_df = filtered_df[filtered_df["Genre"].str.contains(genre, case=False, na=False)]
            break

    # Year range filter (e.g., "between 2010-2020" or "2010 to 2020")
    year_range_match = re.search(r"(\d{4})\s*[-to]+\s*(\d{4})", query_lower)
    if year_range_match:
        start_year = int(year_range_match.group(1))
        end_year = int(year_range_match.group(2))
        filtered_df = filtered_df[
            (filtered_df["Released_Year_cleaned"] >= start_year) &
            (filtered_df["Released_Year_cleaned"] <= end_year)
        ]

    # Single year filter (e.g., "of 2019")
    single_year_match = re.search(r"of\s+(\d{4})", query_lower)
    if single_year_match and not year_range_match:
        year = int(single_year_match.group(1))
        filtered_df = filtered_df[filtered_df["Released_Year_cleaned"] == year]

    # Year before filter (e.g., "before 1990")
    before_year_match = re.search(r"before\s+(\d{4})", query_lower)
    if before_year_match:
        year = int(before_year_match.group(1))
        filtered_df = filtered_df[filtered_df["Released_Year_cleaned"] < year]

    # Sorting - determine sort column
    if "meta" in query_lower or "metascore" in query_lower:
        filtered_df = filtered_df.dropna(subset=["Meta_score"])
        result = filtered_df.nlargest(n, "Meta_score")
    elif "gross" in query_lower or "earning" in query_lower or "box office" in query_lower:
        filtered_df = filtered_df.dropna(subset=["Gross_cleaned"])
        result = filtered_df.nlargest(n, "Gross_cleaned")
    elif "vote" in query_lower:
        result = filtered_df.nlargest(n, "No_of_Votes")
    else:
        # Default to IMDB rating
        result = filtered_df.nlargest(n, "IMDB_Rating")

    if len(result) > 0:
        return format_query_result(result, query)

    return "I couldn't process that query. Please try rephrasing your question with more specific details about what movie information you're looking for."
