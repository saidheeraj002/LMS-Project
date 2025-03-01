from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas import user_schema
from app.services import user_service
from app.api.auth import get_current_user
from app.models import models

router = APIRouter()

@router.post("/users/", response_model=user_schema.User, status_code=status.HTTP_201_CREATED)
async def create_user(user: user_schema.UserCreate, user_service: user_service.UserService = Depends()):
    """
    Create a new user.
    """
    try:
        return await user_service.create_user(user)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/users/{user_id}", response_model=user_schema.User)
async def get_user(user_id: int, user_service: user_service.UserService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    Retrieve a user by ID.
    """
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

# @router.get("/users_list/", response_model=List[user_schema.User])
@router.get("/users_list/")
async def get_users(user_service: user_service.UserService = Depends(), current_user: models.User = Depends(get_current_user)):
    try:
        """
        Retrieve all users.
        """
        return await user_service.get_users()
    except Exception as e:
        return e

@router.put("/users/{user_id}", response_model=user_schema.User)
async def update_user(user_id: int, user: user_schema.UserUpdate, user_service: user_service.UserService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    Update a user.
    """
    updated_user = await user_service.update_user(user_id, user)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return updated_user

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, user_service: user_service.UserService = Depends(), current_user: models.User = Depends(get_current_user)):
    """
    Delete a user.
    """
    deleted = await user_service.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return None