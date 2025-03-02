from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas import lessons_schema
from app.services.lesson_service import LessonService
from app.models import models
from app.api.auth import get_current_user
from typing import List

router = APIRouter()

@router.post("/upload_lessons/", response_model=lessons_schema.Lesson, status_code=status.HTTP_201_CREATED)
async def upload_lesson(lesson_details: lessons_schema.LessonCreate, lesson_service: LessonService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    uploading new Lesson.
    """
    try:
        return await lesson_service.upload_lessons(lesson_details)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/bulk_upload_lessons/", response_model=List[lessons_schema.Lesson], status_code=status.HTTP_201_CREATED)
async def upload_lessons_bulk(lesson_details: lessons_schema.LessonBulkCreate, lesson_service: LessonService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    Upload multiple lessons.
    """
    try:
        created_lessons = []
        for lesson in lesson_details.lessons:
            created_lesson = await lesson_service.upload_lessons(lesson)
            created_lessons.append(created_lesson)
        return created_lessons
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# @router.get("/lessons_list/", response_model=lessons_schema.Lesson)
# async def get_users(lesson_service: LessonService = Depends(), current_user: models.User = Depends(get_current_user)):
#     try:
#         """
#         Retrieve all the Lesson of the Selected Subject.
#         """
#         lessons_data =  await lesson_service.get_lessons_list()
#         return lessons_data
#     except Exception as e:
#         return e


@router.get("/lesson_data/{subject_id}")
async def get_lessons_list(subject_id: int, lesson_service: LessonService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        """
        Retrieve all the Lessons Data based on the Subject ID.
        """
        return await lesson_service.get_lessons_by_subject_id(subject_id)
    except Exception as e:
        return e



@router.get("/lessons_list/")
async def get_lessons_list(lesson_service: LessonService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        """
        Retrieve all the Lessons Data.
        """
        return await lesson_service.get_lessons_list()
    except Exception as e:
        return e
