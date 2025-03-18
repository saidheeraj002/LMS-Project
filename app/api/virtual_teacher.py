from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas import virtual_teacher_schema
from app.models import models
from app.api.auth import get_current_user
from app.ai.docs_ingestion import DocsProcessing
from app.ai.gemini_calling import GeminiCalling

router = APIRouter()

@router.post("/virtual_teacher/")
async def virtual_assistance(request_body: virtual_teacher_schema.AskVirtualTeacherRequest, current_user: models.User = Depends(get_current_user), ai_assistance: DocsProcessing = Depends()):
    """
    Answers the user query.
    """
    try:
        result = await ai_assistance.doc_ingestion_to_model(request_body)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))












