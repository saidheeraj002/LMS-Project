from typing import Optional
from pydantic import BaseModel, EmailStr, root_validator, Field
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    # grade: Optional[str]
    role: Optional[str] = "student"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserCreate(UserBase):
    is_active: bool
    password: str

class UserUpdate(UserBase):
    username: Optional[str] = None
    email: Optional[EmailStr] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class UserInDB(User):
    hashed_password: str