from fastapi import APIRouter, Depends, HTTPException, status
from app.models import models
from app.api.auth import get_current_user
from app.ai.gemini_calling import GeminiCalling
from app.schemas import ai_chat_schema

router = APIRouter()

# @router.post("/ai_chat/")
# async def ask_query(request_body: ai_chat_schema.AIChatRequest, llm_calling: GeminiCalling = Depends()):
#     """
#     Answers the user query with the help of Gemini LLM Model.
#     """
#     try:
#         result = await llm_calling.gemini_api_calling(request_body)
#         return {"answer": result}
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/ai_chat/")
async def ask_query(request_body: dict, llm_calling: GeminiCalling = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    Answers the User with the help of Gemini LLM Model
    :param request_body:
    :param llm_calling:
    :param current_user:
    :return:
    """
    try:
        if type(current_user) is dict and "status_code" in current_user:
            return current_user
        user_details = current_user
        print("request_body", request_body)
        response = await llm_calling.gemini_api_calling(request_body, current_user)
        print("response", response)
        return {"response": response}

    except Exception as e:
        print(e)