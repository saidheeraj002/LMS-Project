from pydoc import describe

from pydantic import BaseModel, Field

class AIChatCreate(BaseModel):
    window_id: str = Field(..., description="Chat Window ID")
    user_query: str = Field(..., description="User Query")
    llm_response: str = Field(..., description="Response of the LLM")
    chat_summery: str = Field(..., description="Summery of the Chat Conversation")

class AIChatRequest(BaseModel):
    window_id: str = Field(..., description="Chat Window ID")
    user_query: str = Field(..., description="User Query")

class AIChatDetails(BaseModel):
    subject: str = Field(..., description="Selected Subject")
    topic: str = Field(..., description="Selected Topic")

