from pinecone import Pinecone, ServerlessSpec
from langchain_google_vertexai import ChatVertexAI
# from langchain_google_vertexai import VertexAIEmbeddings
from typing import Dict, Union
from fastapi.params import Depends
from app.services.ai_conversation_service import AIChatService
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel


class UserQuerySupport:
    def __init__(self, chat_service: AIChatService = Depends()):
        self.pinecone_initialise = Pinecone(api_key="pcsk_4gQsLN_PDNbGoZbjStT8sqdPSCyrTJMvz6rnULgJyJzbmQrMhCnVtHj34apJK4B7xPSgpC")
        self.pinecone_index = self.pinecone_initialise.Index("co-ordinate-geometry-index")
        self.vertex_init = vertexai.init(project="genai-432214", location="us-central1")
        self.gcred = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\projects\LMS-Project\\application_default_credentials.json"
        self.model_langchain = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0, max_retries=2)
        # self.embeddings_model = VertexAIEmbeddings(model="text-embedding-004")
        self.embeddings_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.chat_service = chat_service

    async def get_text_embedding_from_text_embedding_model(self, text: str):
        embeddings = self.embeddings_model.get_embeddings([text])

        text_embedding = [embedding.values for embedding in embeddings][0]

        return text_embedding

    async def get_page_text_embeddings(self, text_data: Union[dict, str]):
        embeddings_dict = {}

        if not text_data:
            return embeddings_dict

        text_embed = await self.get_text_embedding_from_text_embedding_model(text=text_data)
        embeddings_dict["text_embedding"] = text_embed

        return embeddings_dict

    async def query_vector_db(self, query_embeddings, top_k=10, filter=None):
        try:
            result = self.pinecone_index.query(
                vector=query_embeddings,
                top_k=top_k,
                filter=filter,
                include_values=False,
                include_metadata=True,
                rerank={  # This is the rerank parameter
                    "model": "bge-reranker-v2-m3",  # Choose your reranker model
                    "top_n": top_k,  # Number of results to rerank (can be less than top_k)
                    "rank_fields": ["chunk_text"]  # Metadata field containing the text to rerank
                }
            )

            return result.matches
        except Exception as e:
            print(e)

    async def get_gemini_response(self, model_input, stream: bool = True) -> str:
        """
        This function generates text in response to a list of model inputs.

        Args:
            model_input: A list of strings representing the inputs to the model.
            stream: Whether to generate the response in a streaming fashion (returning chunks of text at a time) or all at once. Defaults to False.

        Returns:
            The generated text as a string.
            :param stream:
            :param model_input:
            :param generation_config:
        """
        response = await self.model_langchain.invoke(
            model_input,
            stream=stream,
        )
        response_list = []

        for chunk in response:
            try:
                response_list.append(chunk.text)
            except Exception as e:
                print(
                    "Exception occurred while calling gemini. Something is wrong. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                    e,
                )
                response_list.append("Exception occurred")
                continue
        response = "".join(response_list)

        return response

    async def rag_flow(self, request_body):
        try:
            existing_summery = await self.chat_service.get_chat_summery(request_body)
            summary_text = existing_summery.chat_summery if existing_summery else {}

            # 1. Generate the embedding for the query
            query_embedding = await self.get_page_text_embeddings(request_body['user_query'])

            # 2. Retrieve relevant chunks from Pinecone
            relevant_chunks = await self.query_vector_db(query_embedding['text_embedding'])

            memory = ConversationSummaryMemory(llm=self.model_langchain)
            memory.buffer = summary_text

            conversation_sum = ConversationChain(
                llm=self.model_langchain,
                memory=memory
            )

            # 3. Format the retrieved chunks as context for the LLM
            context = "\n".join([match.metadata["chunk_text"] for match in relevant_chunks])

            # instruction = f"""Answer the question with the given context.
            #     If the information is not available in the context, you can respond with i am not aware of this query.
            #     Greet the user based on the messages the user asks.
            #     Question: {request_body['user_query']}
            #     Context: {context}
            #     Answer:
            #     """

            # instruction = f"""
            # You are a highly specialized AI assistant designed to answer questions related to the 12th standard curriculum (e.g., Physics, Chemistry, Mathematics, Biology, History, Geography, Economics, etc.). Your expertise is strictly limited to this domain.
            #
            # **Instructions:**
            #
            # 1.  **Context-Based Answers:**
            #     * Carefully analyze the provided context.
            #     * Extract the relevant information from the context to answer the user's question accurately.
            #     * If the answer is explicitly stated in the context, provide a direct and concise response.
            #     * If the context contains information that can be used to infer the answer, use that information to construct the response.
            # 2.  **Handling Unanswerable Questions:**
            #     * If the provided context does not contain the information necessary to answer the user's question, respond with: "I am not aware of this query."
            #     * Do not attempt to fabricate answers or provide information outside the given context.
            # 3.  **Out-of-Scope Questions:**
            #     * If the user's question is clearly outside the scope of 12th standard subjects, respond with: "I am capable of answering the query."
            # 4.  **Greeting and Conversational Tone:**
            #     * If the user initiates the conversation with a greeting (e.g., "Hello," "Hi," "Good morning"), respond with an appropriate greeting.
            #     * Maintain a polite and professional tone throughout the conversation.
            #     * If the user does not greet, then just answer the question.
            # 5.  **Mathematical and Scientific Notation:**
            #     * When appropriate, use LaTeX formatting for mathematical and scientific notations (e.g., $E=mc^2$, $\frac{1}{'x'}$, $\sin(\theta)$).
            # 6.  **Conciseness and Clarity:**
            #     * Provide answers that are clear, concise, and easy to understand.
            #     * Avoid unnecessary jargon or overly complex explanations.
            #
            # **Input:**
            #
            # * **Question:** {request_body['user_query']}
            # * **Context:** {context}
            #
            # **Output:**
            #
            # Answer:
            # """

            instruction = f"""
            You are a highly specialized AI assistant designed to answer questions related to the 12th standard curriculum (e.g., Physics, Chemistry, Mathematics, Biology, History, Geography, Economics, etc.). Your expertise is strictly limited to this domain.

            **Instructions:**

            1.  **Context-Based Answers:**
                * Carefully analyze the provided context.
                * Extract the relevant information from the context to answer the user's question accurately.
                * If the answer is explicitly stated in the context, provide a direct and concise response.
                * If the context contains information that can be used to infer the answer, use that information to construct the response.
            2.  **Ordered and Structured Answers:**
                * Present answers in a logical and sequential order.
                * When appropriate, use bullet points to organize information and enhance readability.
                * If the answer involves a series of steps or a list of items, use numbered lists.
            3.  **Handling Unanswerable Questions:**
                * If the provided context does not contain the information necessary to answer the user's question, respond with: "I am not aware of this query."
                * Do not attempt to fabricate answers or provide information outside the given context.
            4.  **Out-of-Scope Questions:**
                * If the user's question is clearly outside the scope of 12th standard subjects, respond with: "I am capable of answering the query."
            5.  **Greeting and Conversational Tone:**
                * If the user initiates the conversation with a greeting (e.g., "Hello," "Hi," "Good morning"), respond with an appropriate greeting.
                * Maintain a polite and professional tone throughout the conversation.
                * If the user does not greet, then just answer the question.
            6.  **Mathematical and Scientific Notation:**
                * When appropriate, use LaTeX formatting for mathematical and scientific notations (e.g., $E=mc^2$, $\frac{1}{'x'}$, $\sin(\theta)$).
            7.  **Conciseness and Clarity:**
                * Provide answers that are clear, concise, and easy to understand.
                * Avoid unnecessary jargon or overly complex explanations.

            **Input:**

            * **Question:** {request_body['user_query']}
            * **Context:** {context}

            **Output:**

            Answer:
            """

            contents = [instruction]

            response = conversation_sum(contents)

            conversation_details = {"window_id": request_body['window_id'], "user_query": request_body['user_query'],
                                    "llm_response": response['response'],
                                    "chat_summery": conversation_sum.memory.buffer}

            chat_conversation = await self.chat_service.insert_chat_conversation(conversation_details)

            return response
        except Exception as e:
            print(e)

