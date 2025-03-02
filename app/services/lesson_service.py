from fastapi import Depends
from app.models import models
from app.db_manager import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.schemas import lessons_schema


class LessonService:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def upload_lessons(self, lesson: lessons_schema.LessonCreate):
        result = await self.db.execute(select(models.Lesson).filter(models.Lesson.name == lesson.name))
        lesson_data = result.scalar_one_or_none()
        if lesson_data:
            return {"status_code": 400, "detail": "Lesson is already Present"}
        new_lesson = models.Lesson(name=lesson.name, lesson_description=lesson.lesson_description, subject_id=lesson.subject_id, lesson_pdf_link=lesson.lesson_pdf_link)
        self.db.add(new_lesson)
        await self.db.commit()
        await self.db.refresh(new_lesson)
        return new_lesson

    async def get_lessons_list(self):
        result = await self.db.execute(select(models.Lesson))
        lessons_list = result.scalars().all()
        return lessons_list

    async def get_lessons_by_subject_id(self, subject_id):
        result = await self.db.execute(select(models.Lesson).filter(models.Lesson.subject_id == subject_id))
        lessons_list = result.scalars().all()
        return lessons_list

    async def get_lesson_details(self, request_details):
        try:
            result = await self.db.execute(select(models.Lesson).with_only_columns(models.Lesson.lesson_pdf_link).filter(
                models.Lesson.id == request_details.lesson_id[0], models.Lesson.subject_id == request_details.subject_id))
            # lesson_id = request_details.lesson_id[0]
            # result = await self.db.execute(
            #     select(models.Lesson).filter(models.Lesson.id == 16, models.Lesson.subject_id == 1))
            lesson_details = result.scalars().all()
            return lesson_details
        except Exception as e:
            print(e)