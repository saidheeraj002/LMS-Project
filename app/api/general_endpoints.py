from fastapi import APIRouter, Depends, HTTPException, status
from app.models import models
from app.api.auth import get_current_user
from app.services.subject_service import SubjectService
from app.services.lesson_service import LessonService


router = APIRouter()


@router.get("/get_user_details/")
async def user_details(current_user: models.User = Depends(get_current_user)):
    """
    Get the Logged-In User Details
    """
    try:
        print("current_user", current_user)
        if type(current_user) is dict:
            return current_user
        else:
            current_user_details = current_user
            print("current_user", current_user_details)
            return {"status_code": 200, "response": current_user_details}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/home_data")
async def home_page_data(subject_service: SubjectService = Depends(), lesson_service: LessonService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    Get all the home data based on the student grade
    :return:
    """
    try:
        user_details = {}
        print("home page data")
        if type(current_user) is dict and "status_code" in current_user:
            return current_user

        user_details.update(current_user)

        subject_list = await subject_service.get_class_subjects_list(current_user['grade'])

        subject_data = {subject.id: {"title": subject.title, "topics": []} for subject in subject_list}

        subject_ids_list = list(subject_data.keys())

        topics_list = await lesson_service.get_lessons_by_subject_id(subject_ids_list)

        for topic in topics_list:
            subject_id = topic.subject_id  # Ensure the Lesson model has a subject_id field
            if subject_id in subject_data:
                subject_data[subject_id]["topics"].append(topic.name)

        user_details.update({"subjects": list(subject_data.values())})
        print("response", user_details)
        return user_details
    except Exception as e:
        print(e)



