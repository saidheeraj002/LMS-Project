import vertexai
from fastapi.params import Depends
# from vertexai.generative_models import GenerativeModel
import os
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from app.services.ai_conversation_service import AIChatService
from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import Part, GenerativeModel
from app.services.subject_service import SubjectService
from app.services.lesson_service import LessonService


class GeminiCalling:
    def __init__(self, chat_service: AIChatService = Depends(), subject_service: SubjectService = Depends(),
                 lesson_service: LessonService = Depends()):
        self.vertex_init = vertexai.init(project="genai-432214", location="us-central1")
        # self.model_name = GenerativeModel("gemini-2.0-pro-exp-02-05")
        # self.model_name = GenerativeModel("gemini-2.0-flash-001")
        self.model_langchain  =  ChatVertexAI(model="gemini-2.0-flash-001", temperature=0, max_retries=2)
        self.gcred = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "*******************"
        self.chat_service = chat_service
        self.subject_service = subject_service
        self.lesson_service = lesson_service

    async def gemini_api_calling(self, request_body, current_user):
        try:
            existing_summery = await self.chat_service.get_chat_summery(request_body)
            summary_text = existing_summery.chat_summery if existing_summery else {}

            memory = ConversationSummaryMemory(llm=self.model_langchain)
            memory.buffer = summary_text

            conversation_sum = ConversationChain(
                llm=self.model_langchain,
                memory=memory
            )

            subject_list = await self.subject_service.get_class_subjects_id(current_user['grade'], request_body['subject'])
            subject_ids_list = [subject.id for subject in subject_list]
            topics_list = await self.lesson_service.get_lesson_pdf_by_subject_id(subject_ids_list, request_body['topic'])
            # pdf_file = [topic.lesson_pdf_link for topic in topics_list]

            pdf_file = [topic.lesson_pdf_link for topic in topics_list]
            pdf_content = Part.from_uri(pdf_file[0], mime_type="application/pdf")
            print("pdf_content", pdf_content)

            system_prompt = """
            You are a professional document analyzer designed to assist users with information extracted from a provided PDF document.

            **Capabilities:**

            1.  **Greeting and Chapter Context:**
                * When a user sends a greeting (e.g., "Hi," "Hello," "Hey"), respond with a friendly greeting and indicate the specific chapter context of the provided PDF.
                * Example responses:
                    * "Hi, How can I assist you today with the Trigonometry Chapter?"
                    * "Hello, How can I help you with the Circles Chapter?"
                    * "Hey, How can I assist you today with the Chemical Reactions and Equations Chapter?"
            2.  **Query Analysis and Response:**
                * Analyze the user's query and search for relevant information within the provided PDF document.
                * If the answer is found, provide a clear and concise response based on the document's content.
                * If the answer is not found, respond with "I do not have information on your query."

            **Workflow:**

            1.  User uploads a PDF document.
            2.  The system identifies the chapter or subject matter of the PDF.
            3.  The system waits for user input.
            4.  Upon receiving user input:
                * If it is a greeting, provide a greeting and chapter context.
                * If it is a query, search the PDF and provide an answer or indicate that the information is not available.
            """

            user_query = request_body['query']

            contents = [pdf_content, system_prompt, user_query]

            # response = model.generate_content(contents)

            response = self.model_gemini.generate_content(contents)

            langchain_response = conversation_sum.run(response.text)

            # response = conversation_sum(contents)
            conversation_details = {"window_id":request_body['window_id'], "user_query": request_body['query'],
                                    "llm_response": langchain_response,
                                    "chat_summery":conversation_sum.memory.buffer}

            chat_conversation = await self.chat_service.insert_chat_conversation(conversation_details)
            return langchain_response
        except Exception as e:
            print(e)
