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
        try:
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
        except Exception as e:
            print(e)

    async def insert_new_chat_window(self, chat_window_details):
        try:
            new_chat_window = models.ChatWindows(window_id=chat_window_details['window_id'],
                                                 username=chat_window_details['username'], subject=chat_window_details['subject'],
                                                 topic=chat_window_details['topic'])
            self.db.add(new_chat_window)
            await self.db.commit()
            await self.db.refresh(new_chat_window)
            return new_chat_window
        except Exception as e:
            print(e)

    async def get_chat_history(self, chat_id):
        try:
            chat_history = await self.db.execute(select(models.ChatConversations).filter(models.ChatConversations.window_id == chat_id))
            chat_history_data = chat_history.scalars().all()
            if chat_history_data:
                return chat_history_data
            else:
                return None
        except Exception as e:
            print(e)

    async def get_recent_chats(self, user_data):
        try:
            recent_chats = await self.db.execute(select(models.ChatWindows).filter(models.ChatWindows.username == user_data))
            chat_history_data = recent_chats.scalars().all()
            if chat_history_data:
                return chat_history_data
            else:
                return None
        except Exception as e:
            print(e)

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

