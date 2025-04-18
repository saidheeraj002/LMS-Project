from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.models import models
from app.api.auth import get_current_user
from app.ai.gemini_calling import GeminiCalling
from app.ai.user_query_support import UserQuerySupport
from app.ai.langchain_pdf_to_gemini import AIUserQuerySupport
from app.schemas import ai_chat_schema
import uuid

from app.schemas.ai_chat_schema import AIChatDetails
from app.services.ai_conversation_service import AIChatService


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

@router.post("/new_chat/")
async def new_chat(chat_details: ai_chat_schema.AIChatDetails = Body(), chat_service: AIChatService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        if type(current_user) is dict and "status_code" in current_user:
            return current_user
        print("chat_details", chat_details)
        chat_id = uuid.uuid4()
        print(chat_id)
        chat_window_details = {"username": current_user['username'], "window_id": chat_id, "subject": chat_details.subject, "topic": chat_details.topic}
        await chat_service.insert_new_chat_window(chat_window_details)
        return  {"chat_id": chat_id, "subject": chat_details.subject, "topic": chat_details.topic}
    except Exception as e:
        print(e)

@router.get("/chat_details/")
async def get_chat_details(chat_id: str, chat_service: AIChatService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        if type(current_user) is dict and "status_code" in current_user:
            return current_user
        chat_history = await chat_service.get_chat_history(chat_id)
        return {"chat_history": chat_history}
    except Exception as e:
        print(e)

@router.get("/recent_chat_data")
async def get_recent_chats(chat_service: AIChatService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        if type(current_user) is dict and "status_code" in current_user:
            return current_user
        recent_chats = await chat_service.get_recent_chats(current_user['username'])
        return {'recent_chats': recent_chats, "username": current_user['username']}
    except Exception as e:
        print(e)

@router.post("/ai_chat/")
# async def ask_query(request_body: dict, llm_calling: GeminiCalling = Depends(), current_user: models.User = Depends(get_current_user)):
# async def ask_query(request_body: dict, llm_calling: UserQuerySupport = Depends(), current_user: models.User = Depends(get_current_user)):
async def ask_query(request_body: dict, llm_calling: AIUserQuerySupport = Depends(), current_user: models.User = Depends(get_current_user)):
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
        response = await llm_calling.handle_agentic_query(request_body)
        print("response", response)
        return response

    except Exception as e:
        print(e)