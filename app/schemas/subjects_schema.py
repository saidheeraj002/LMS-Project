from typing import Optional
from pydantic import BaseModel, EmailStr, root_validator, Field
from datetime import datetime

class SubjectBase(BaseModel):
    title: str
    description: Optional[str] = None
    grade: str

class SubjectCreate(SubjectBase):
    pass

class SubjectUpdate(SubjectBase):
    title: Optional[str] = None
    description: Optional[str] = None

class Subject(SubjectBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

