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