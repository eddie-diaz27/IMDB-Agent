"""
Aggregation tool for complex multi-step queries involving groupby, count, and threshold filtering.
"""
from typing import Type
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class AggregationInput(BaseModel):
    """Input schema for aggregation queries."""
    group_by: str = Field(description="Column to group by (e.g., 'Director', 'Star1')")
    filter_column: str = Field(description="Column to apply threshold filter (e.g., 'Gross_cleaned', 'IMDB_Rating')")
    filter_threshold: float = Field(description="Minimum value for the filter column")
    min_count: int = Field(default=2, description="Minimum number of items per group to include")
    sort_by: str = Field(default="count", description="Sort results by: 'count' or 'total'")
    limit: int = Field(default=10, description="Maximum number of groups to return")


class AggregationTool(BaseTool):
    """Tool for complex aggregation queries on the movie dataset."""

    name: str = "movie_aggregation"
    description: str = """REQUIRED for queries with "at least twice", "at least N times", or "multiple movies".

USE THIS TOOL when the query asks for:
- Directors/actors with MULTIPLE movies meeting a criteria
- "at least twice" or "at least N times" - ALWAYS use this tool
- "Top directors and their highest grossing movies with gross > X at least twice"
- Counting how many movies a person has that meet a threshold

Example query: "Top directors with highest grossing movies with gross earnings greater than 500M at least twice"
Example call: group_by="Director", filter_column="Gross_cleaned", filter_threshold=500000000, min_count=2

Parameters:
- group_by: Column to group by (Director, Star1, Star2, Star3, Star4)
- filter_column: Column to filter on (Gross_cleaned, IMDB_Rating, Meta_score, No_of_Votes)
- filter_threshold: Minimum value (e.g., 500000000 for $500M, 8.0 for rating)
- min_count: How many movies must meet the criteria (default: 2)
- sort_by: 'count' or 'total' (default: 'count')
- limit: Max results (default: 10)
"""

    args_schema: Type[BaseModel] = AggregationInput
    df: pd.DataFrame = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'df', df)

    def _run(
        self,
        group_by: str,
        filter_column: str,
        filter_threshold: float,
        min_count: int = 2,
        sort_by: str = "count",
        limit: int = 10
    ) -> str:
        """Execute the aggregation query."""
        df = object.__getattribute__(self, 'df')

        # Map common column name variations to actual column names
        column_mapping = {
            'Gross': 'Gross_cleaned',
            'gross': 'Gross_cleaned',
            'gross_earnings': 'Gross_cleaned',
            'earnings': 'Gross_cleaned',
            'box_office': 'Gross_cleaned',
            'Runtime': 'Runtime_mins',
            'runtime': 'Runtime_mins',
            'Released_Year': 'Released_Year_cleaned',
            'Year': 'Released_Year_cleaned',
            'year': 'Released_Year_cleaned',
        }

        # Apply column mapping
        if filter_column in column_mapping:
            filter_column = column_mapping[filter_column]

        # Validate columns
        valid_group_columns = ['Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Genre', 'Certificate']
        valid_filter_columns = ['Gross_cleaned', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime_mins', 'Released_Year_cleaned']

        if group_by not in df.columns:
            return f"Error: '{group_by}' is not a valid column. Use one of: {', '.join(valid_group_columns)}"
        if filter_column not in df.columns:
            return f"Error: '{filter_column}' is not a valid column. Use one of: {', '.join(valid_filter_columns)}"

        # Step 1: Drop NaN values in filter column and filter movies meeting the threshold
        working_df = df.dropna(subset=[filter_column]).copy()
        filtered = working_df[working_df[filter_column] >= filter_threshold].copy()

        if len(filtered) == 0:
            return f"No movies found with {filter_column} >= {filter_threshold:,.0f} in the IMDB Top 1000 dataset."

        # Step 2: Group and count
        grouped = filtered.groupby(group_by).agg({
            'Series_Title': 'count',
            filter_column: 'sum'
        }).rename(columns={'Series_Title': 'movie_count', filter_column: 'total'})

        # Step 3: Filter by minimum count
        qualifying = grouped[grouped['movie_count'] >= min_count]

        if len(qualifying) == 0:
            return f"No {group_by}s found with at least {min_count} movies where {filter_column} >= {filter_threshold:,.0f} in the IMDB Top 1000 dataset."

        # Step 4: Sort
        if sort_by == "count":
            qualifying = qualifying.sort_values('movie_count', ascending=False)
        else:
            qualifying = qualifying.sort_values('total', ascending=False)

        # Step 5: Format results with details
        output_lines = [f"Found {len(qualifying)} {group_by}(s) with at least {min_count} movies where {filter_column} >= {filter_threshold:,.0f}:\n"]

        for name, row in qualifying.head(limit).iterrows():
            count = int(row['movie_count'])
            total = row['total']

            # Get the specific movies for this person/category
            person_movies = filtered[filtered[group_by] == name].nlargest(5, filter_column)

            output_lines.append(f"**{name}** ({count} qualifying movies)")
            if filter_column == 'Gross_cleaned':
                output_lines.append(f"  Total gross earnings: ${total:,.0f}")

            for _, movie in person_movies.iterrows():
                title = movie['Series_Title']
                year = movie.get('Released_Year_cleaned', movie.get('Released_Year', '?'))
                value = movie[filter_column]
                rating = movie.get('IMDB_Rating', '?')

                if filter_column == 'Gross_cleaned':
                    output_lines.append(f"  - {title} ({year}) - Gross: ${value:,.0f}, IMDB: {rating}")
                else:
                    output_lines.append(f"  - {title} ({year}) - {filter_column}: {value}, IMDB: {rating}")

            output_lines.append("")  # Add blank line between groups

        return "\n".join(output_lines)
