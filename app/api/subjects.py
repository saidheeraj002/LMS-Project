from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas import subjects_schema
from app.services.subject_service import SubjectService
from app.models import models
from app.api.auth import get_current_user

router = APIRouter()

@router.post("/upload_subjects/", response_model=subjects_schema.Subject, status_code=status.HTTP_201_CREATED)
async def upload_subject(subject_details: subjects_schema.SubjectCreate, subject_service: SubjectService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    uploading new subject.
    """
    try:
        return await subject_service.upload_subjects(subject_details)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/class_subjects_list/{class_details}")
async def get_subjects_list(class_details: int, subject_service: SubjectService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        print("class_details", class_details)
        """
        Retrieve all the Subjects of specified class.
        """
        return await subject_service.get_class_subjects_list(class_details)
    except Exception as e:
        return e

@router.get("/subjects_list/")
async def get_subjects_list(subject_service: SubjectService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        """
        Retrieve all the Subjects.
        """
        return await subject_service.get_subjects_list()
    except Exception as e:
        return e
