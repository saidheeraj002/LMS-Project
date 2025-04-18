from pinecone import Pinecone, ServerlessSpec
from langchain_google_vertexai import ChatVertexAI
from typing import Dict, Union, List, Optional, Any
from fastapi.params import Depends
from app.services.ai_conversation_service import AIChatService # Assuming this path is correct

from langchain.memory import ConversationSummaryMemory # Keep for later steps
from langchain.agents import AgentExecutor # The runner for the agent
from langchain_core.messages import SystemMessage, BaseMessage # For memory handling

import os, asyncio
import vertexai
from vertexai.language_models import TextEmbeddingModel

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_tool_calling_agent

from langchain.agents import Tool
from dotenv import load_dotenv

load_dotenv()


class AIUserQuerySupport:
    def __init__(self, chat_service: AIChatService = Depends()):
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")  # Consider moving to env variables
        self.pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "co-ordinate-geometry-index")
        self.gcp_project = os.environ.get("GCP_PROJECT_ID")
        self.gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        # self.google_creds_path = "D:\projects\LMS-Project\\application_default_credentials.json"

        self.google_creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_creds_path

        # Initialize Vertex AI SDK (needed for TextEmbeddingModel)
        try:
            vertexai.init(project=self.gcp_project, location=self.gcp_location)
            print("Vertex AI SDK Initialized.")
        except Exception as e:
            print(f"Error initializing Vertex AI SDK: {e}")
            # Handle error appropriately, maybe raise it

        # Initialize Pinecone
        try:
            self.pinecone_initialise = Pinecone(api_key=self.pinecone_api_key)
            self.pinecone_index = self.pinecone_initialise.Index(self.pinecone_index_name)
            print(f"Pinecone client initialized for index '{self.pinecone_index_name}'.")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            # Handle error appropriately

        self.model_langchain = ChatVertexAI(
            model_name="gemini-2.0-flash-001",  # Updated model name based on your original code
            temperature=0,
            max_retries=2,
            project=self.gcp_project,
            location=self.gcp_location
        )

        # Initialize Embeddings Model (Using TextEmbeddingModel as per your usage)
        try:
            self.embeddings_model = TextEmbeddingModel.from_pretrained(
                "text-embedding-004")  # Using gecko as 004 might still be preview/specific access
            print("Vertex AI TextEmbeddingModel Initialized.")
        except Exception as e:
            print(f"Error initializing TextEmbeddingModel: {e}")
            # Handle error appropriately

        self.chat_service = chat_service

        self.vector_db_tool = Tool(
            name="KnowledgeBaseRetriever",
            func=self._execute_pinecone_retrieval_sync,  # Sync wrapper
            coroutine=self._execute_pinecone_retrieval_async,  # Async function
            description="Use this ONLY to retrieve information about 12th standard curriculum subjects (Physics, Chemistry, Maths, Biology, History, Geography, Economics, etc.). Input should be a specific question or topic.",
        )
        print("Langchain Tool 'KnowledgeBaseRetriever' created.")

        # planner_system_template = """You are an expert query analyzer for a chatbot specializing in the 12th standard curriculum (Physics, Chemistry, Maths, Biology, History, etc.).
        # Your goal is to determine the next action based on the user's latest query and the ongoing conversation summary.
        #
        # Analyze the 'User Query' in the context of the 'Chat Summary'.
        #
        # Possible Actions:
        # 1.  **Refine for Retrieval:** If the query asks for specific information found within the 12th standard curriculum knowledge base, refine the query into the most effective search term or question for retrieving relevant documents. Output *only* the refined search query string.
        # 2.  **Mark as Conversational:** If the query is a greeting, closing, thanks, or a general conversational remark that doesn't require knowledge base lookup, output the exact keyword: CONVERSATIONAL
        # 3.  **Mark as Out of Scope:** If the query is clearly unrelated to 12th standard subjects (e.g., current events, general knowledge, personal advice), output the exact keyword: OUT_OF_SCOPE
        #
        # **Important:**
        # - Focus ONLY on the LATEST 'User Query'. Use the 'Chat Summary' for context but don't act on older queries in the summary.
        # - Do NOT answer the question yourself.
        # - Output *only* the refined query string OR one of the keywords ('CONVERSATIONAL', 'OUT_OF_SCOPE'). No extra text or explanation.
        #
        # Chat Summary:
        # {chat_summary}"""

        planner_system_template = """You are an intelligent query planner for an educational chatbot that assists students across various academic levels and subjects (such as Physics, Chemistry, Math, Biology, History, etc.). 
            Your role is to analyze the latest 'User Query' in the context of the ongoing 'Chat Summary' and determine the appropriate next step for handling the query.
            
            Possible Actions:
            1. **Refine for Retrieval:**  
            If the query requests specific subject-related information that is likely present in the academic knowledge base, rewrite the query into a more precise and effective form for document retrieval. Output only the refined search query string.
            
            2. **Mark as Conversational:**  
            If the query is a greeting, farewell, expression of thanks, or general conversational remark that doesn’t require knowledge lookup, output the keyword: CONVERSATIONAL
            
            3. **Mark as Out of Scope:**  
            If the query is unrelated to academic subjects (e.g., questions about current events, personal advice, or unrelated topics), output the keyword: OUT_OF_SCOPE
            
            Important:
            - Focus only on the latest 'User Query'. Use the 'Chat Summary' only for context.
            - Do NOT answer the question.
            - Output only the refined query string OR one of the keywords: CONVERSATIONAL, OUT_OF_SCOPE. No extra text or explanation.
            
            Chat Summary:  
            {chat_summary}"""

        self.planner_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(planner_system_template),
            HumanMessagePromptTemplate.from_template("User Query: {user_query}")
        ])

        self.planner_chain = LLMChain(
            llm=self.model_langchain,
            prompt=self.planner_prompt,
            verbose=True  # Set to True for debugging planner's thoughts (optional)
        )
        print("Planner LLMChain created.")

        # executor_system_message = """You are a helpful and knowledgeable AI assistant specializing in the 12th standard curriculum (Physics, Chemistry, Mathematics, Biology, History, Geography, Economics, etc.).
        # Your primary goal is to answer the user's questions accurately based on the provided context and conversation history.
        #
        # **Instructions:**
        # 1.  **Tool Use:** You MUST use the 'KnowledgeBaseRetriever' tool to find information related to 12th standard subjects if the answer is not obvious from the chat history. Do not answer curriculum questions from memory; always retrieve fresh context if needed.
        # 2.  **Context Synthesis:** If the 'KnowledgeBaseRetriever' tool provides context, base your answer *strictly* on that retrieved context and the relevant parts of the chat history.
        # 3.  **Handling No Context:** If the tool returns "No relevant information found..." or similar, inform the user politely that the specific information is not available in the knowledge base. Do not invent answers.
        # 4.  **Conversational Flow:** Respond naturally to greetings, closings, and conversational remarks based on the chat history.
        # 5.  **Clarity and Formatting:** Provide clear, concise answers. Use LaTeX formatting for mathematical/scientific notation where appropriate. Use bullet points or numbered lists for structured information if helpful. # <-- DOUBLE CHECK THIS LINE
        # 6.  **Scope:** Do not answer questions outside the 12th standard curriculum domain. If asked, politely state your specialization.
        #
        # Begin!"""

        executor_system_message = """You are a helpful and knowledgeable AI assistant specializing in answering academic questions across various subjects such as Physics, Chemistry, Mathematics, Biology, History, Geography, Economics, and more.

            Your primary goal is to respond accurately to the user's questions based on the provided tool context and ongoing conversation history.
            
            **Instructions:**
            1. **Tool Use:**  
               You MUST use the 'KnowledgeBaseRetriever' tool to retrieve information related to academic subjects when the answer is not directly evident from the chat history. Do not rely solely on memory for academic questions; always attempt to retrieve up-to-date and relevant information.
            
            2. **Context Synthesis:**  
               If the 'KnowledgeBaseRetriever' tool returns context, base your answer strictly on that context and relevant parts of the conversation history.
            
            3. **Handling No Context:**  
               If the tool returns "No relevant information found..." or similar, inform the user politely that the specific information isn't available in the knowledge base. Do not make up or assume answers.
            
            4. **Conversational Flow:**  
               Respond naturally to greetings, farewells, or conversational remarks, based on the tone and flow of the chat history.
            
            5. **Clarity and Formatting:**  
               Provide clear and concise answers. Use LaTeX formatting for any mathematical or scientific expressions when appropriate. For structured or multi-step answers, use bullet points or numbered lists to improve readability.
            
            6. **Scope:**  
               Do not answer questions that fall outside the domain of academic learning. If asked, politely inform the user that your assistance is focused on academic subjects.
            
            Begin!
            """

        self.executor_prompt = ChatPromptTemplate.from_messages([
            ("system", executor_system_message),
            MessagesPlaceholder(variable_name="chat_history"),  # Provides conversation context
            ("human", "{input}"),  # The user's query (potentially refined by the planner)
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # Place for agent's intermediate thoughts/tool calls
        ])

        print("Executor Agent Prompt Template created.")
        print(f"Executor Prompt Input Variables: {self.executor_prompt.input_variables}")

        self.executor_agent_logic = create_tool_calling_agent(
            llm=self.model_langchain,
            tools=[self.vector_db_tool],  # Provide the tool list here for structure definition
            prompt=self.executor_prompt
        )

    async def _get_embedding_vector(self, text: str) -> List[float]:
        """Internal method to get the raw embedding vector for a text."""
        if not text:
            print("Warning: Attempting to embed empty text.")
            return []
        try:
            embeddings_response = self.embeddings_model.get_embeddings([text])

            if embeddings_response and embeddings_response[0].values:
                return embeddings_response[0].values
            else:
                print(f"Warning: Could not get embedding values for text: {text[:50]}...")
                return []
        except Exception as e:
            print(f"Error getting embedding for text '{text[:50]}...': {e}")
            return []  # Returning empty for now, agent needs to handle "not found"

    async def _query_pinecone_index(self, query_embeddings: List[float], top_k: int = 5,
                                    filter: Optional[Dict] = None) -> List[Dict]:
        """Internal method to query Pinecone index, including reranking."""
        if not query_embeddings:
            print("Skipping Pinecone query due to empty embedding vector.")
            return []
        try:
            # Using top_k=5 as default. Reranking happens *after* this initial retrieval.
            # The reranker model specified here needs to be supported by your Pinecone setup.
            # Ensure 'chunk_text' is the correct metadata field containing the text.
            result = self.pinecone_index.query(
                vector=query_embeddings,
                top_k=top_k,  # Initial retrieval count
                filter=filter,
                include_values=False,
                include_metadata=True,
                # Reranking configuration - KEPT AS REQUESTED
                rerank={
                    "model": "bge-reranker-v2-m3", # Ensure this model is available/configured
                    "top_n": top_k, # Rerank the initially retrieved top_k results
                    "rank_fields": ["chunk_text"] # Field containing text for reranking
                }
            )
            # Return the list of matches (potentially reranked by Pinecone if configured)
            return result.matches if result and result.matches else []
        except Exception as e:
            print(f"Error querying Pinecone index '{self.pinecone_index_name}': {e}")
            # Return empty list to indicate failure to the calling function
            return []

    async def _execute_pinecone_retrieval_async(self, query: str) -> str:
        """
        Takes a query string, gets embedding, queries Pinecone (with reranking),
        and returns formatted context string for the Langchain Tool.
        """
        print(f"--- Tool: Retrieving context for query: {query[:100]}... ---")
        try:
            # 1. Get embedding vector
            query_embedding = await self._get_embedding_vector(query)

            if not query_embedding:
                # If embedding failed, inform the agent
                return "Could not process the query to search the knowledge base."

            # 2. Retrieve relevant chunks from Pinecone (includes reranking if configured)
            relevant_matches = await self._query_pinecone_index(query_embedding, top_k=5)  # Using default top_k=5

            if not relevant_matches:
                print("--- Tool: No relevant chunks found in Pinecone. ---")
                # Clear message for the agent
                return "No relevant information found in the knowledge base for this specific query."

            # 3. Format the retrieved chunks into a context string
            context_parts = []
            seen_content = set()  # Optional: Prevent duplicate chunks if metadata is identical
            for match in relevant_matches:
                # Ensure metadata and the specific text field exist
                if match.metadata and "chunk_text" in match.metadata:
                    content = match.metadata["chunk_text"]
                    if content not in seen_content:
                        context_parts.append(content)
                        seen_content.add(content)
                else:
                    # Log if expected metadata is missing
                    print(f"Warning: Match {match.id} missing 'chunk_text' in metadata.")

            if not context_parts:
                print("--- Tool: Matches found but no 'chunk_text' extracted. ---")
                # Message indicating retrieval happened but content extraction failed
                return "Relevant documents were found, but their text content could not be extracted."

            formatted_context = "\n\n---\n\n".join(context_parts)
            print(f"--- Tool: Formatted context generated (length: {len(formatted_context)}). ---")
            return f"Context from Knowledge Base:\n{formatted_context}"

        except Exception as e:
            print(f"Error during Pinecone retrieval execution: {e}")
            return "An error occurred while trying to retrieve information from the knowledge base."

    def _execute_pinecone_retrieval_sync(self, query: str) -> str:
        """Synchronous wrapper for the async retrieval method."""
        try:
            return asyncio.run(self._execute_pinecone_retrieval_async(query))
        except RuntimeError as e:
            print(f"RuntimeError in sync wrapper (possibly nested event loops): {e}. Trying another approach.")
            return "Error: Could not execute retrieval due to async loop issue."
        except Exception as e:
            print(f"Error in sync wrapper for Pinecone retrieval: {e}")
            return "An error occurred while trying to retrieve information (sync wrapper)."

    def _extract_summary_string_from_memory(self, memory_output: Dict[str, Any]) -> str:
        """
        Extracts the summary string from ConversationSummaryMemory output.
        NOTE: This might need adjustment based on the exact structure returned
        by memory.aload_memory_variables() when return_messages=True.
        """
        chat_history = memory_output.get("chat_history", [])
        if isinstance(chat_history, str):  # Should not happen with return_messages=True but check
            return chat_history
        elif isinstance(chat_history, list):
            for message in reversed(chat_history):
                if isinstance(message, SystemMessage) and "current conversation summary:" in message.content.lower():
                    parts = message.content.split(":", 1)
                    if len(parts) > 1:
                        print("Extracted summary from SystemMessage.")
                        return parts[1].strip()
            print("Warning: Could not find explicit summary message in chat history.")
            return ""  # Or potentially stringify the last few messages as a crude summary
        else:
            print(f"Warning: Unexpected chat_history format in memory: {type(chat_history)}")
            return ""

    async def handle_agentic_query(self, request_body: Dict[str, Any]):
        """
        Handles user queries using the Planner-Executor agent architecture.
        Replaces the old rag_flow method.
        """
        user_query = request_body.get("user_query")
        # Assuming 'window_id' is the key for chat session identification
        chat_id = request_body.get("window_id")

        if not user_query or not chat_id:
            return {"error": "Missing 'user_query' or 'window_id' in request body."}  # Basic validation

        print(f"\n--- Handling Agentic Query for chat_id: {chat_id} ---")
        print(f"User Query: {user_query}")

        ai_response = "Sorry, I encountered an issue. Please try again."  # Default error response
        new_summary = ""  # Default empty summary

        try:
            summary_data = await self.chat_service.get_chat_summery(request_body)  # Or pass chat_id directly
            current_summary = summary_data.chat_summery if summary_data and hasattr(summary_data,
                                                                                    'chat_summery') else ""
            print(f"Fetched summary (length {len(current_summary)}): '{current_summary[:100]}...'")

            # 2. Initialize Memory for this request
            memory = ConversationSummaryMemory(
                llm=self.model_langchain,
                memory_key="chat_history",  # Must match placeholder in executor prompt
                input_key="input",  # Must match placeholder in executor prompt
                buffer=current_summary,  # Load existing summary
                return_messages=True  # Crucial for agents using chat prompts
            )

            print("ConversationSummaryMemory initialized.")

            # 3. Invoke Planner
            print("\n--- Invoking Planner ---")
            planner_input = {"user_query": user_query, "chat_summary": current_summary}
            planner_output = await self.planner_chain.ainvoke(planner_input)
            planner_decision = planner_output['text'].strip()
            print(f"Planner Decision: {planner_decision}")

            interaction_input = user_query
            interaction_output = ""

            # 4. Decision Branching & Execution
            if planner_decision == 'CONVERSATIONAL':
                print("\n--- Handling as Conversational ---")
                simple_prompt = ChatPromptTemplate.from_messages([
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                simple_chain = LLMChain(llm=self.model_langchain, prompt=simple_prompt, memory=memory)
                response_data = await simple_chain.ainvoke({"input": user_query})
                ai_response = response_data['text']
                interaction_output = ai_response

            elif planner_decision == 'OUT_OF_SCOPE':
                print("\n--- Handling as Out of Scope ---")
                ai_response = "My expertise is limited to the 12th standard curriculum. I cannot assist with that request."
                interaction_output = ai_response

            else:
                print(f"\n--- Invoking Executor Agent with refined query: {planner_decision} ---")
                agent_executor = AgentExecutor(
                    agent=self.executor_agent_logic,  # The logic defined in Step 3
                    tools=[self.vector_db_tool],  # The tool from Step 1
                    memory=memory,  # The memory instance for this request
                    verbose=True,  # Essential for debugging agent steps
                    handle_parsing_errors=True,  # Attempt to recover from LLM format errors
                    max_iterations=5,  # Safety limit to prevent loops
                    # return_intermediate_steps=True # Optional: Get intermediate steps for debugging
                )
                try:
                    executor_response = await agent_executor.ainvoke({"input": planner_decision})
                    ai_response = executor_response['output']
                    interaction_output = ai_response
                except Exception as agent_exc:
                    print(f"Error during Agent Executor execution: {agent_exc}")
                    ai_response = "I encountered an error while processing that request with my tools. Please try rephrasing."
                    interaction_output = ai_response

            print(f"\n--- Final AI Response: {ai_response} ---")

            if memory and interaction_output:  # Only save if there was an output
                print(
                    f"DEBUG: Explicitly calling asave_context. Input: '{interaction_input[:50]}...', Output: '{interaction_output[:50]}...'")
                try:
                    memory.save_context({"input": interaction_input}, {"output": interaction_output})
                except Exception as e:
                    print("save_context:", e)

                new_summary = memory.buffer

                print("summery", memory.buffer)
                print(f"DEBUG: Memory buffer after explicit save_context: '{memory.buffer[:100]}...'")

            elif not memory:
                print("ERROR: Memory object was not initialized.")
            else:
                print("DEBUG: Skipping context saving as interaction_output is empty.")

            # final_memory_state = await memory.aload_memory_variables({})
            # new_summary = self._extract_summary_string_from_memory(final_memory_state)

            # print(f"Extracted new summary (length {len(new_summary)}): '{new_summary[:100]}...'")

        except Exception as e:
            print(f"Error in handle_agentic_query: {e}")

        try:
            conversation_details = {
                "window_id": chat_id,
                "user_query": user_query,
                "llm_response": ai_response,
                "chat_summery": new_summary  # Save the latest summary
            }
            # Assuming insert_chat_conversation handles the DB interaction
            await self.chat_service.insert_chat_conversation(conversation_details)
            print(f"Conversation turn saved for chat_id: {chat_id}")
        except Exception as db_exc:
            print(f"Error saving conversation to DB for chat_id {chat_id}: {db_exc}")
        return {"response": ai_response}





