from fastapi import Depends
from app.models import models
from app.schemas import subjects_schema
from app.db_manager import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


class SubjectService:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def upload_subjects(self, subject: subjects_schema.SubjectCreate):
        result = await self.db.execute(select(models.Subject).filter(models.Subject.title == subject.title))
        subject_data = result.scalar_one_or_none()
        if subject_data:
            return {"status_code": 400, "detail": "Subject is already Present"}
        new_subject = models.Subject(title=subject.title, description=subject.description, grade=subject.grade)
        self.db.add(new_subject)
        await self.db.commit()
        await self.db.refresh(new_subject)
        return new_subject

    async def get_subjects_list(self):
        result = await self.db.execute(select(models.Subject))
        return result.scalars().all()

    async def get_class_subjects_list(self, grade):
        result = await self.db.execute(select(models.Subject).filter(models.Subject.grade == grade))
        subjects_list = result.scalars().all()
        return subjects_list

    async def get_class_subjects_id(self, grade, subject_name):
        result = await self.db.execute(select(models.Subject).filter(models.Subject.grade == grade, models.Subject.title == subject_name))
        subjects_list = result.scalars().all()
        return subjects_list
