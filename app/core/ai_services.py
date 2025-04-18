# app/core/ai_services.py

import os
import asyncio
from typing import Dict, List, Optional, Any
import vertexai
from vertexai.language_models import TextEmbeddingModel
from pinecone import Pinecone, ServerlessSpec
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import Tool, create_tool_calling_agent
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# No need for dotenv here if main.py loads it first

# --- Global Dictionary to Hold Shared AI/Langchain Resources ---
# This dictionary will be populated by initialize_ai_services
shared_services: Dict[str, Any] = {}

# --- Helper Functions for Pinecone Retrieval ---
# These functions are used internally during initialization and potentially by the Tool

async def _get_embedding_vector_shared(text: str, embeddings_model: TextEmbeddingModel) -> list[float]:
    """Shared helper to get embedding vector."""
    if not text: print("Warning: Attempting to embed empty text."); return []
    try:
        embeddings_response = embeddings_model.get_embeddings([text])
        if embeddings_response and embeddings_response[0].values: return embeddings_response[0].values
        else: print(f"Warning: Could not get embedding values for text: {text[:50]}..."); return []
    except Exception as e: print(f"Error getting embedding for text '{text[:50]}...': {e}"); return []

async def _query_pinecone_index_shared(query_embeddings: list[float], pinecone_index, top_k: int = 5, filter: dict | None = None) -> list[dict]:
    """Shared helper to query Pinecone index."""
    if not query_embeddings: print("Skipping Pinecone query due to empty embedding vector."); return []
    try:
        result = pinecone_index.query(
            vector=query_embeddings,
            top_k=top_k,
            filter=filter,
            include_values=False,
            include_metadata=True,
            # rerank={ ... } # Optional: Add rerank config here if used
            )
        return result.matches if result and result.matches else []
    except Exception as e: print(f"Error querying Pinecone index: {e}"); return []

async def _execute_pinecone_retrieval_shared(query: str, embeddings_model: TextEmbeddingModel, pinecone_index) -> str:
    """Shared retrieval logic for the Langchain Tool's coroutine."""
    print(f"--- Tool: Retrieving context for query: {query[:100]}... ---")
    try:
        query_embedding = await _get_embedding_vector_shared(query, embeddings_model)
        if not query_embedding: return "Could not process the query to search the knowledge base."

        relevant_matches = await _query_pinecone_index_shared(query_embedding, pinecone_index, top_k=5)
        if not relevant_matches: print("--- Tool: No relevant chunks found in Pinecone. ---"); return "No relevant information found in the knowledge base for this specific query."

        context_parts = []; seen_content = set()
        for match in relevant_matches:
            if match.metadata and "chunk_text" in match.metadata:
                content = match.metadata["chunk_text"]
                if content not in seen_content: context_parts.append(content); seen_content.add(content)
            else: print(f"Warning: Match {match.id} missing 'chunk_text' in metadata.")
        if not context_parts: print("--- Tool: Matches found but no 'chunk_text' extracted. ---"); return "Relevant documents were found, but their text content could not be extracted."

        formatted_context = "\n\n---\n\n".join(context_parts)
        print(f"--- Tool: Formatted context generated (length: {len(formatted_context)}). ---")
        return f"Context from Knowledge Base:\n{formatted_context}"
    except Exception as e: print(f"Error during Pinecone retrieval execution: {e}"); return "An error occurred while trying to retrieve information from the knowledge base."


# --- Initialization Function ---
async def initialize_ai_services():
    """
    Initializes all shared AI clients and Langchain components.
    Populates the shared_services dictionary.
    """
    print("Initializing shared AI services...")
    global shared_services

    # --- Load Configuration ---
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "co-ordinate-geometry-index")
    gcp_project = os.environ.get("GCP_PROJECT_ID")
    gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
    google_creds_path_from_env = os.environ.get("GOOGLE_CREDENTIALS_PATH")

    # --- Validate Configuration & Set Google Creds ---
    if not pinecone_api_key: raise ValueError("PINECONE_API_KEY not set.")
    if not gcp_project: raise ValueError("GCP_PROJECT_ID not set.")
    if google_creds_path_from_env:
        if os.path.exists(google_creds_path_from_env):
            # Set env var ONLY if path is provided via GOOGLE_CREDENTIALS_PATH
            # This allows external setting of GOOGLE_APPLICATION_CREDENTIALS to still work
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path_from_env
            print(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {google_creds_path_from_env}")
        else:
            raise FileNotFoundError(f"Google credentials file not found at: {google_creds_path_from_env}")
    elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_CREDENTIALS_PATH/GOOGLE_APPLICATION_CREDENTIALS not set.")

    # --- Initialize Shared Clients ---
    try:
        # Check if already initialized (useful in some environments)
        try:
            vertexai.get_initialized_project()
            print("Vertex AI SDK already initialized.")
        except ValueError:
            vertexai.init(project=gcp_project, location=gcp_location)
            print("Vertex AI SDK Initialized.")

        # Store initialized clients in the shared dictionary
        shared_services["embeddings_model"] = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        print("Shared Vertex AI TextEmbeddingModel Initialized.")
        shared_services["llm"] = ChatVertexAI(model_name="gemini-1.5-flash-001", temperature=0, max_retries=2, project=gcp_project, location=gcp_location)
        print("Shared Langchain ChatVertexAI Initialized.")
    except Exception as e: print(f"Fatal Error initializing Vertex AI components: {e}"); raise

    try:
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        shared_services["pinecone_index"] = pinecone_client.Index(pinecone_index_name)
        print(f"Shared Pinecone client initialized for index '{pinecone_index_name}'.")
    except Exception as e: print(f"Fatal Error initializing Pinecone: {e}"); raise

    # --- Initialize Shared Langchain Components ---
    retrieval_coroutine = lambda query: _execute_pinecone_retrieval_shared(
        query=query,
        embeddings_model=shared_services["embeddings_model"],
        pinecone_index=shared_services["pinecone_index"]
    )
    shared_services["vector_db_tool"] = Tool(
        name="KnowledgeBaseRetriever",
        coroutine=retrieval_coroutine,
        description="Use this ONLY to retrieve information about 12th standard curriculum subjects (Physics, Chemistry, Maths, Biology, History, Geography, Economics, etc.). Input should be a specific question or topic.",
    )
    print("Shared Langchain Tool 'KnowledgeBaseRetriever' created.")

    planner_system_template = """You are an expert query analyzer... [Rest of prompt unchanged] ... Chat Summary: {chat_summary}""" # Truncated
    planner_prompt = ChatPromptTemplate.from_messages([ SystemMessagePromptTemplate.from_template(planner_system_template), HumanMessagePromptTemplate.from_template("User Query: {user_query}") ])
    shared_services["planner_chain"] = LLMChain( llm=shared_services["llm"], prompt=planner_prompt, verbose=True )
    print("Shared Planner LLMChain created.")

    executor_system_message = """You are a helpful and knowledgeable AI assistant... [Rest of prompt unchanged] ... Begin!""" # Truncated
    executor_prompt = ChatPromptTemplate.from_messages([ ("system", executor_system_message), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad"), ])
    shared_services["executor_agent_logic"] = create_tool_calling_agent(
        llm=shared_services["llm"],
        tools=[shared_services["vector_db_tool"]],
        prompt=executor_prompt
    )
    print("Shared Executor Agent core logic created.")

    print("--- Shared AI services initialized successfully ---")


# --- Dependency Getter Functions ---
# These allow FastAPI/other modules to safely access the initialized services

def get_llm() -> ChatVertexAI:
    """Dependency injector for the shared LLM instance."""
    service = shared_services.get("llm")
    if not service:
        raise RuntimeError("LLM service not initialized.")
    return service

def get_planner_chain() -> LLMChain:
    """Dependency injector for the shared Planner Chain instance."""
    service = shared_services.get("planner_chain")
    if not service:
        raise RuntimeError("Planner Chain service not initialized.")
    return service

def get_vector_db_tool() -> Tool:
    """Dependency injector for the shared Vector DB Tool instance."""
    service = shared_services.get("vector_db_tool")
    if not service:
        raise RuntimeError("Vector DB Tool service not initialized.")
    return service

def get_executor_agent_logic(): # -> AgentType (replace with actual type if known)
    """Dependency injector for the shared Executor Agent logic."""
    service = shared_services.get("executor_agent_logic")
    if not service:
        raise RuntimeError("Executor Agent Logic not initialized.")
    return service

# Add getters for embeddings_model and pinecone_index if needed directly elsewhere
def get_embeddings_model() -> TextEmbeddingModel:
    """Dependency injector for the shared Embeddings Model instance."""
    service = shared_services.get("embeddings_model")
    if not service:
        raise RuntimeError("Embeddings Model service not initialized.")
    return service

def get_pinecone_index(): # -> Pinecone Index Type
    """Dependency injector for the shared Pinecone Index instance."""
    service = shared_services.get("pinecone_index")
    if not service:
        raise RuntimeError("Pinecone Index service not initialized.")
    return service