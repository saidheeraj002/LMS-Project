from typing import List
from pydantic import BaseModel, Field

class AskVirtualTeacherRequest(BaseModel):
    uuid: str = Field(..., description="User UUID")
    subject_id: int = Field(..., description="Subject ID")
    lesson_id: List[int] = Field(..., description="List of Lesson IDs")
    question: str = Field(..., description="User's question")