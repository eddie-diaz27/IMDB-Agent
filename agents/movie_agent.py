"""
Movie agent orchestration using LangChain.
"""
import json
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import pandas as pd
import chromadb

from config.settings import OPENAI_API_KEY, LLM_MODEL, MAX_ITERATIONS, MEMORY_WINDOW_SIZE
from agents.prompts import SYSTEM_PROMPT, CLARIFICATION_PROMPT
from tools.structured_query import StructuredQueryTool
from tools.semantic_search import SemanticSearchTool
from tools.summarizer import SummarizerTool
from tools.recommender import RecommenderTool


class MovieAgent:
    """Main movie agent that orchestrates tools and conversation."""

    def __init__(self, df: pd.DataFrame, collection: chromadb.Collection):
        """Initialize the movie agent.

        Args:
            df: Preprocessed movie DataFrame
            collection: ChromaDB collection with movie overviews
        """
        self.df = df
        self.collection = collection

        # Initialize tools
        self.tools = {
            "structured_movie_query": StructuredQueryTool(df=df),
            "semantic_movie_search": SemanticSearchTool(collection=collection),
            "movie_summarizer": SummarizerTool(df=df),
            "movie_recommender": RecommenderTool(df=df, collection=collection)
        }

        # Create LLM with tool binding
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        ).bind_tools(list(self.tools.values()))

        # Conversation history
        self.chat_history: List = []

        # Track clarification state
        self.pending_clarification = None

    def _build_messages(self, user_input: str) -> List:
        """Build the message list for the LLM."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add recent chat history (keep last N exchanges)
        history_limit = MEMORY_WINDOW_SIZE * 2  # pairs of messages
        for msg in self.chat_history[-history_limit:]:
            messages.append(msg)

        messages.append(HumanMessage(content=user_input))
        return messages

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return its result."""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                return tool._run(**tool_args)
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        return f"Unknown tool: {tool_name}"

    def check_for_clarification(self, query: str) -> Optional[dict]:
        """Check if the query needs clarification.

        Args:
            query: User query

        Returns:
            Clarification dict if needed, None otherwise
        """
        query_lower = query.lower()

        # Common words to exclude from actor name detection
        exclude_words = {
            "top", "best", "worst", "movies", "films", "movie", "film",
            "the", "and", "with", "from", "about", "list", "show", "find",
            "what", "when", "where", "who", "how", "which", "rating",
            "score", "meta", "imdb", "gross", "comedy", "drama", "action",
            "horror", "sci-fi", "thriller", "romance", "genre", "year",
            "before", "after", "between", "above", "below", "over", "under"
        }

        # Check for actor queries without role specification
        # Only trigger for queries that look like "[Actor Name] movies" patterns
        actor_query_patterns = ["movies", "films", "filmography"]

        if any(pattern in query_lower for pattern in actor_query_patterns):
            # Check if query mentions an actor but not role
            if not any(role in query_lower for role in ["lead", "star1", "main role", "supporting", "any role"]):
                # Look for potential actor names (consecutive capitalized words)
                words = query.split()
                potential_names = []

                # Find sequences of capitalized words that could be names
                i = 0
                while i < len(words):
                    if words[i] and words[i][0].isupper() and words[i].lower() not in exclude_words:
                        # Start building a potential name
                        name_parts = [words[i]]
                        j = i + 1
                        while j < len(words) and words[j] and words[j][0].isupper() and words[j].lower() not in exclude_words:
                            name_parts.append(words[j])
                            j += 1
                        # Only consider multi-word names (first + last) or single names > 4 chars
                        if len(name_parts) >= 2 or (len(name_parts) == 1 and len(name_parts[0]) > 4):
                            potential_names.append(" ".join(name_parts))
                        i = j
                    else:
                        i += 1

                # Check if any potential name is in our dataset
                for name in potential_names:
                    # Check for exact name match in Stars columns
                    name_found = False
                    for star_col in ["Star1", "Star2", "Star3", "Star4"]:
                        if self.df[star_col].str.contains(name, case=False, na=False).any():
                            name_found = True
                            break

                    if name_found:
                        return {
                            "needs_clarification": True,
                            "clarification_type": "actor_role",
                            "question": f"Are you looking for movies where {name} is the lead actor (main role) or any movies featuring them in any role?",
                            "options": [
                                f"{name} as lead actor only",
                                f"Any movie featuring {name}"
                            ]
                        }

        return None

    def invoke(self, query: str) -> str:
        """Process a user query.

        Args:
            query: User's question or request

        Returns:
            Agent's response
        """
        # Check for clarification if no pending clarification
        if not self.pending_clarification:
            clarification = self.check_for_clarification(query)
            if clarification and clarification.get("needs_clarification"):
                self.pending_clarification = {
                    "original_query": query,
                    "clarification": clarification
                }
                options_text = "\n".join([f"- {opt}" for opt in clarification["options"]])
                return f"{clarification['question']}\n\n{options_text}"

        # Handle clarification response
        if self.pending_clarification:
            original_query = self.pending_clarification["original_query"]
            clarification = self.pending_clarification["clarification"]

            # Modify query based on clarification response
            if clarification["clarification_type"] == "actor_role":
                if "lead" in query.lower() or "main" in query.lower() or "1" in query:
                    query = f"{original_query} (lead role only, Star1)"
                else:
                    query = f"{original_query} (any role)"

            self.pending_clarification = None

        # Execute agent with tool calling loop
        try:
            messages = self._build_messages(query)

            # Agentic loop - allow multiple tool calls
            for _ in range(MAX_ITERATIONS):
                response = self.llm.invoke(messages)

                # Check if the model wants to call tools
                if response.tool_calls:
                    # Execute each tool call
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        result = self._execute_tool(tool_name, tool_args)
                        tool_results.append(f"Tool '{tool_name}' result:\n{result}")

                    # Add tool results to messages and continue
                    messages.append(response)
                    for i, tool_call in enumerate(response.tool_calls):
                        messages.append(ToolMessage(
                            content=tool_results[i],
                            tool_call_id=tool_call["id"]
                        ))
                else:
                    # No more tool calls, return the final response
                    final_response = response.content
                    # Update chat history
                    self.chat_history.append(HumanMessage(content=query))
                    self.chat_history.append(AIMessage(content=final_response))
                    return final_response

            return "I reached the maximum number of steps. Please try a simpler query."

        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}\nPlease try rephrasing your question."

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        self.pending_clarification = None


def create_movie_agent(df: pd.DataFrame, collection: chromadb.Collection) -> MovieAgent:
    """Factory function to create a movie agent.

    Args:
        df: Preprocessed movie DataFrame
        collection: ChromaDB collection with movie overviews

    Returns:
        Configured MovieAgent instance
    """
    return MovieAgent(df, collection)
