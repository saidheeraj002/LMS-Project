import vertexai
from fastapi.params import Depends
# from vertexai.generative_models import GenerativeModel
import os
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from app.services.ai_conversation_service import AIChatService
from langchain_google_vertexai import ChatVertexAI
# from vertexai.generative_models import Part, GenerativeModel
from app.services.subject_service import SubjectService
from app.services.lesson_service import LessonService


class GeminiCalling:
    def __init__(self, chat_service: AIChatService = Depends(), subject_service: SubjectService = Depends(),
                 lesson_service: LessonService = Depends()):
        self.vertex_init = vertexai.init(project="genai-432214", location="us-central1")
        self.model_langchain  =  ChatVertexAI(model="gemini-2.0-flash-001", temperature=0, max_retries=2)
        self.gcred = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "*******************"
        self.chat_service = chat_service
        self.subject_service = subject_service
        self.lesson_service = lesson_service

    # async def gemini_api_calling(self, request_body, current_user):
    async def gemini_api_calling(self, request_body):
        try:
            existing_summery = await self.chat_service.get_chat_summery(request_body)
            summary_text = existing_summery.chat_summery if existing_summery else {}

            memory = ConversationSummaryMemory(llm=self.model_langchain)
            memory.buffer = summary_text

            conversation_sum = ConversationChain(
                llm=self.model_langchain,
                memory=memory
            )

            user_query = request_body['user_query']

            contents = [user_query]

            response = conversation_sum(contents)

            conversation_details = {"window_id":request_body['window_id'], "user_query": request_body['user_query'],
                                    "llm_response": response['response'],
                                    "chat_summery":conversation_sum.memory.buffer}

            chat_conversation = await self.chat_service.insert_chat_conversation(conversation_details)
            return response
        except Exception as e:
            print(e)
