from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from datetime import timedelta
import sqlalchemy
from app.db_manager import get_db
from app.schemas import user_schema as user_schemas
from app.services import user_service
# from app.database.schemas import generic as generic_schemas
from app.core import security
from app.core.config import settings
from typing import List
from app.models import models
from jose import JWTError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter()


# # @router.post("/register", response_model=user_schemas.User)
# @router.post("/register")
# async def register_user(user: user_schemas.UserCreate, user_service: user_service.UserService = Depends()):
#     try:
#         result = await user_service.create_user(user)
#         # db_user = await db.query(models.User).filter(models.User.email == user.email).first()
#         # result = await db.execute(sqlalchemy.select(models.User).filter(models.User.email == user.email))
#         # db_user = result.scalar_one_or_none()
#         # if db_user:
#         #     raise HTTPException(status_code=400, detail="Email already registered")
#         # hashed_password = security.get_password_hash(user.password)
#         # new_db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password,
#         #                           role=user.role)
#         # db.add(new_db_user)
#         # await db.flush()
#         # await db.commit()
#         # await db.refresh(new_db_user)
#         # return new_db_user
#         return result
#     except Exception as e:
#         print("Error while creating the User", e)
#         return {"400": {"description": "Email Already Registered"}}


# @router.post("/login", response_model=generic_schemas.Message)
@router.post("/login")
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
async def login_for_access_token(user_login: user_schemas.UserLogin = Body(), db: Session = Depends(get_db)):
    try:
        # db_user = await db.query(models.User).filter(models.User.username == form_data.username).first()
        result = await db.execute(
            sqlalchemy.select(models.User).filter(models.User.email == user_login.email))
        db_user = result.scalar_one_or_none()
        if not db_user or not security.verify_password(user_login.password, db_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            data={"sub": db_user.username}, expires_delta=access_token_expires
        )
        # return {"message": f"Bearer {access_token}"}
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        print(e)
        return {"401": {"description": "Incorrect username or password"}}


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # credentials_exception = HTTPException(
    #     status_code=status.HTTP_401_UNAUTHORIZED,
    #     detail="Could not validate credentials",
    #     headers={"WWW-Authenticate": "Bearer"},
    # )
    try:
        payload = security.decode_access_token(token)
        if type(payload) is str and payload == "Signature has expired.":
            return {"status_code": 401, "response": "Token has Expired"}
        else:
            username: str = payload.get("sub")
    except JWTError as e:
        return {"status_code": 401, "response": "Invalid Token"}

    result = await db.execute(sqlalchemy.select(models.User.username, models.User.email).filter(
        models.User.username == username))

    result = await db.execute(
        sqlalchemy.select(models.User.username, models.User.grade).filter(models.User.username == username)
    )

    # result = await db.execute(sqlalchemy.select(models.User).where(models.User.username == username))
    user = result.fetchone()
    if user:
        user = dict(user._mapping)
    if user is None:
        return {"status_code": 401, "response": "User not found"}
    return user


# @router.get("/profile", response_model=user_schemas.User)
async def read_users_me(current_user: user_schemas.User = Depends(get_current_user)):
    return current_user


# @router.put("/profile/update", response_model=user_schemas.User)
async def update_user(user_update: user_schemas.UserUpdate, current_user: user_schemas.User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    try:
        if user_update.username:
            current_user.username = user_update.username
        if user_update.email:
            current_user.email = user_update.email
        if user_update.role:
            current_user.role = user_update.role
        db.commit()
        db.refresh(current_user)

        return current_user
    except Exception as e:
        print(e)


# @router.post("/logout")
async def logout_user():
    # In a real-world scenario, you might want to blacklist the token
    return {"message": "Successfully logged out"}


# @router.get("/users_list", response_model=List[user.User])
# async def users_list(db: Session = Depends(get_db)):
#     # users = db.query(models.User).all()
#     results = await db.execute(sqlalchemy.select(models.User))
#     users = results.scalars().all()
#
#     return users
