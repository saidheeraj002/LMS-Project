import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Part
import os
from app.services.virtual_teacher_service import VirtualTeacherService
from app.services.lesson_service import LessonService
from fastapi import Depends


class DocsProcessing:
    def __init__(self, lesson_service: LessonService = Depends()):
        self.vertex_init = vertexai.init(project="genai-432214", location="us-central1")
        # self.model_name = GenerativeModel("gemini-2.0-pro-exp-02-05")
        self.model_name = GenerativeModel("gemini-2.0-flash-001")
        self.gcred = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "*****************"
        # self.virtual_teacher_service = VirtualTeacherService()
        self.lesson_service = lesson_service

    async def doc_ingestion_to_model(self, request_details):
        model = self.model_name

        # lesson_details = await self.virtual_teacher_service.get_lesson_details(request_details)
        lesson_details = await self.lesson_service.get_lesson_details(request_details)

        pdf_files = "\n".join(lesson_details)

        pdf_content = Part.from_uri(pdf_files, mime_type="application/pdf")

        system_prompt = ("You are Helpful assistant for the 10th Grade School Students, Help them with the query that they ask. Only answer from the document that we provide"
                         "Don't Hellucinate anything just answer from the document that we provide,"
                         "If there is no relevant answer for the query you can respond with i don't know.")

        query = request_details.question
        prompt = system_prompt + query

        contents = [pdf_content, prompt]

        response = model.generate_content(contents)

        return response.text