"""
Data preprocessing module for IMDB dataset.
Handles cleaning and normalization of the CSV data.
"""
import pandas as pd
import re
from typing import Optional


def parse_gross(value: str) -> Optional[int]:
    """Convert gross earnings string to integer.

    Examples:
        "28,341,469" -> 28341469
        None/NaN -> None
    """
    if pd.isna(value) or value == "":
        return None
    # Remove commas and quotes, then convert to int
    cleaned = str(value).replace(",", "").replace('"', "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_runtime(value: str) -> Optional[int]:
    """Extract runtime in minutes from string.

    Examples:
        "142 min" -> 142
        None/NaN -> None
    """
    if pd.isna(value) or value == "":
        return None
    match = re.search(r"(\d+)", str(value))
    if match:
        return int(match.group(1))
    return None


def parse_year(value) -> Optional[int]:
    """Parse release year to integer.

    Handles edge cases like 'PG' (data quality issue in some rows).
    """
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the IMDB dataset.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Cleaned pandas DataFrame with additional computed columns
    """
    df = pd.read_csv(csv_path)

    # Clean Gross column
    df["Gross_cleaned"] = df["Gross"].apply(parse_gross)

    # Clean Runtime column
    df["Runtime_mins"] = df["Runtime"].apply(parse_runtime)

    # Clean Released_Year column
    df["Released_Year_cleaned"] = df["Released_Year"].apply(parse_year)

    # Clean No_of_Votes - convert to numeric
    df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"], errors="coerce")

    # Clean Meta_score - convert to numeric
    df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")

    # Clean IMDB_Rating - ensure numeric
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")

    # Create list of all stars for actor searches
    df["All_Stars"] = df.apply(
        lambda row: [
            str(row["Star1"]) if pd.notna(row["Star1"]) else "",
            str(row["Star2"]) if pd.notna(row["Star2"]) else "",
            str(row["Star3"]) if pd.notna(row["Star3"]) else "",
            str(row["Star4"]) if pd.notna(row["Star4"]) else "",
        ],
        axis=1
    )

    # Create searchable stars string for easier searching
    df["Stars_str"] = df["All_Stars"].apply(lambda x: ", ".join([s for s in x if s]))

    # Parse genres into list
    df["Genre_list"] = df["Genre"].apply(
        lambda x: [g.strip() for g in str(x).split(",")] if pd.notna(x) else []
    )

    return df


def get_movie_by_title(df: pd.DataFrame, title: str) -> Optional[pd.Series]:
    """Find a movie by its title (case-insensitive).

    Args:
        df: The preprocessed DataFrame
        title: Movie title to search for

    Returns:
        Movie row as Series if found, None otherwise
    """
    mask = df["Series_Title"].str.lower() == title.lower()
    matches = df[mask]
    if len(matches) > 0:
        return matches.iloc[0]

    # Try partial match
    mask = df["Series_Title"].str.lower().str.contains(title.lower(), na=False)
    matches = df[mask]
    if len(matches) > 0:
        return matches.iloc[0]

    return None


def filter_by_genre(df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """Filter movies by genre (case-insensitive).

    Args:
        df: The preprocessed DataFrame
        genre: Genre to filter by

    Returns:
        Filtered DataFrame
    """
    genre_lower = genre.lower()
    mask = df["Genre_list"].apply(
        lambda genres: any(g.lower() == genre_lower for g in genres)
    )
    return df[mask]


def filter_by_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter movies by year range (inclusive).

    Args:
        df: The preprocessed DataFrame
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Filtered DataFrame
    """
    mask = (df["Released_Year_cleaned"] >= start_year) & (df["Released_Year_cleaned"] <= end_year)
    return df[mask]


def filter_by_actor(df: pd.DataFrame, actor_name: str, lead_only: bool = False) -> pd.DataFrame:
    """Filter movies by actor name.

    Args:
        df: The preprocessed DataFrame
        actor_name: Actor name to search for
        lead_only: If True, only search in Star1 column

    Returns:
        Filtered DataFrame
    """
    actor_lower = actor_name.lower()

    if lead_only:
        mask = df["Star1"].str.lower().str.contains(actor_lower, na=False)
    else:
        mask = df["Stars_str"].str.lower().str.contains(actor_lower, na=False)

    return df[mask]
