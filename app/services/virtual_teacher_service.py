from fastapi import Depends
from app.models import models
from app.db_manager import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


class VirtualTeacherService:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def get_lesson_details(self, request_details):
        result = await self.db.execute(select(models.Subject).filter(models.Lesson.id == request_details.lesson_id[0], models.Lesson.subject_id == request_details.subject_id))
        lesson_details = result.scalars().all()
        return lesson_details
