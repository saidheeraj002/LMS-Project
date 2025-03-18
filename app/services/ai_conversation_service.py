from fastapi import Depends
from app.models import models
from app.schemas import ai_chat_schema
from app.db_manager import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


class AIChatService:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def insert_chat_conversation(self, ai_chat_details):
        # result = await self.db.execute(select(models.Subject).filter(models.Subject.title == subject.title))
        # subject_data = result.scalar_one_or_none()
        # if subject_data:
        #     return {"status_code": 400, "detail": "Subject is already Present"}
        new_chat_conversation = models.ChatConversations(window_id=ai_chat_details['window_id'], user_query=ai_chat_details['user_query'],
                                                         llm_response=ai_chat_details['llm_response'], chat_summery=ai_chat_details['chat_summery'])
        self.db.add(new_chat_conversation)
        await self.db.commit()
        await self.db.refresh(new_chat_conversation)
        return new_chat_conversation

    async def get_chat_summery(self, request_data):
        try:
            result = await self.db.execute(select(models.ChatConversations).filter(models.ChatConversations.window_id == request_data['window_id']))
            chat_data = result.scalars().all()

            if chat_data:
                last_conversation = chat_data[-1]
                return last_conversation
            else:
                return None
        except Exception as e:
            print(e)

    async def get_pdf_url(self, request_data):
        try:
            result = await self.db.execute(select(models.Subject).filter(models.Subject.grade == request_data.grade))
            chat_data = result.scalars().all()

            if chat_data:
                last_conversation = chat_data[-1]
                return last_conversation
            else:
                return None
        except Exception as e:
            print(e)

