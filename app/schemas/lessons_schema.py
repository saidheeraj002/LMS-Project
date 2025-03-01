from typing import Optional, List
from pydantic import BaseModel, EmailStr, root_validator, Field
from datetime import datetime

class LessonBase(BaseModel):
    name: str
    lesson_description: Optional[str] = None
    subject_id: int
    lesson_pdf_link: Optional[str] = None

class LessonCreate(LessonBase):
    pass

class LessonUpdate(LessonBase):
    title: Optional[str] = None
    description: Optional[str] = None

class Lesson(LessonBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class LessonBulkCreate(BaseModel):
    lessons: List[LessonCreate]

