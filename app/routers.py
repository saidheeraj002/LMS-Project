from fastapi import APIRouter
from app.api import auth, users, subjects, lessons, virtual_teacher

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/api", tags=["Users"])
api_router.include_router(subjects.router, prefix="/api", tags=["Subjects"])
api_router.include_router(lessons.router, prefix="/api", tags=["Lesson"])
api_router.include_router(virtual_teacher.router, prefix="/api", tags=["VirtualTeacher"])

# api_router.include_router(assessments.router, prefix="/api/assessments", tags=["Assessments"])
# api_router.include_router(virtual_teacher.router, prefix="/api/virtual-teacher", tags=["Virtual Teacher"])
# api_router.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
# api_router.include_router(payments.router, prefix="/api/payments", tags=["Payments"])