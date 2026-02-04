# IMDB Movie Agent

A Gen AI-powered conversational agent that answers questions about the IMDB Top 1000 movies dataset using OpenAI and LangChain.

## Features

- **Movie Lookups**: Get details about specific movies (release year, cast, ratings, etc.)
- **Filtering & Ranking**: Find top movies by rating, meta score, or box office earnings
- **Semantic Search**: Search movies by plot themes and content
- **Summarization**: Synthesize information about director filmographies or compare movie plots
- **Recommendations**: Get personalized movie recommendations based on preferences
- **Clarification**: The bot asks follow-up questions when queries are ambiguous

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini (configurable)
- **Framework**: LangChain
- **Vector Store**: ChromaDB (local persistence)
- **Embeddings**: OpenAI text-embedding-3-small
- **UI**: Streamlit
- **Data Processing**: Pandas

## Setup Instructions

### 1. Clone/Extract the Project

```bash
cd realpage
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env` and replace `your_openai_api_key_here` with your actual API key:

```
API_KEY=sk-proj-your-actual-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Configuration

To change the LLM model, edit `config/settings.py`:

```python
# Options: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
LLM_MODEL = "gpt-4o-mini"
```

## Example Questions

1. **Simple Lookup**: "When did The Matrix release?"
2. **Top N by Score**: "What are the top 5 movies of 2019 by meta score?"
3. **Filtered Ranking**: "Top 7 comedy movies between 2010-2020 by IMDB rating?"
4. **Multi-Filter**: "Top horror movies with a meta score above 85 and IMDB rating above 8"
5. **Aggregation**: "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice"
6. **Complex Filter**: "Top 10 movies with over 1M votes but lower gross earnings"
7. **Semantic + Filter**: "List of movies from the comedy genre where there is death or dead people involved"
8. **Summarization**: "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies"
9. **Semantic Search**: "List of movies before 1990 that have involvement of police in the plot"
10. **Recommendations**: "Recommend movies similar to Inception"

## Project Structure

```
realpage/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── .env.example                # API key template (copy to .env)
├── .env                        # Your API key (not in git)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── config/
│   └── settings.py             # Model and app configuration
├── imdb_dataset/
│   └── imdb_top_1000.csv       # IMDB dataset
├── data/
│   └── preprocessor.py         # Data cleaning utilities
├── vectorstore/
│   ├── store.py                # ChromaDB setup and operations
│   └── chroma_db/              # Persisted vector store (auto-generated)
├── agents/
│   ├── movie_agent.py          # LangChain agent orchestration
│   └── prompts.py              # System prompts
└── tools/
    ├── structured_query.py     # Pandas-based filtering
    ├── semantic_search.py      # Vector similarity search
    ├── summarizer.py           # Plot summarization
    └── recommender.py          # Movie recommendations
```

## How It Works

1. **Data Preprocessing**: The CSV is loaded and cleaned (parsing gross earnings, runtime, etc.)
2. **Vector Store**: Movie plot overviews are embedded and stored in ChromaDB for semantic search
3. **Tool Routing**: The LangChain agent selects the appropriate tool based on the query type:
   - Structured queries → Pandas filtering
   - Plot-based queries → ChromaDB semantic search
   - Summarization → Multi-movie synthesis
   - Recommendations → Hybrid semantic + structured matching
4. **Response Generation**: Results are formatted and returned with context

## First Run

On the first run, the application will:
1. Load and preprocess the IMDB dataset
2. Generate embeddings for all 1000 movie overviews (takes ~1-2 minutes)
3. Persist the vector store locally

Subsequent runs will load the persisted vector store and start quickly.

## Notes

- The vector store is persisted in `vectorstore/chroma_db/` - delete this folder to regenerate embeddings
- For best results with complex queries, the agent may make multiple tool calls
- The bot will ask clarifying questions for ambiguous queries (e.g., actor role specification)
