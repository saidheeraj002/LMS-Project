from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db_manager import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    grade = Column(Integer, nullable=True)
    role = Column(String, default="student")  # "student", "teacher", "parent"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, index=True, nullable=False)
    description = Column(Text)
    grade = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    lessons = relationship("Lesson", back_populates="subject") # Corrected back_populates



class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True, nullable=False)
    lesson_description = Column(Text)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    lesson_pdf_link = Column(Text)

    subject = relationship("Subject", back_populates="lessons") # Corrected back_populates


class ChatConversations(Base):
    __tablename__ = "chat_conversations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    window_id = Column(Integer, nullable=True)
    user_query = Column(Text)
    llm_response = Column(Text)
    chat_summery = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ChatWindows(Base):
    __tablename__ = "chat_windows"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    window_id = Column(String, index=True, nullable=False)
    username = Column(String, index=True, nullable=False)
    subject = Column(String, index=True, nullable=False)
    topic = Column(String, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
