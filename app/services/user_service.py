from fastapi import Depends
from app.models import models
from app.schemas import user_schema
from app.db_manager import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.security import get_password_hash

class UserService:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def authenticate_user(self, username, password):
        result = await self.db.execute(select(models.User).where(models.User.username == username))
        user = result.scalars().first()
        if not user:
            return None
        if user.password != password: #In real app, hash and compare.
            return None
        return user

    async def create_user(self, user: user_schema.UserCreate):
        result = await self.db.execute(select(models.User).filter(models.User.email == user.email))
        db_user = result.scalar_one_or_none()
        if db_user:
            return {"status_code":400, "detail":"Email already registered"}
        hashed_password = get_password_hash(user.password)
        new_db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password,
                                                            role=user.role)
        self.db.add(new_db_user)
        await self.db.commit()
        await self.db.refresh(new_db_user)
        return new_db_user

    async def get_user(self, user_id: int):
        return await self.db.get(models.User, user_id)

    async def get_users(self):
        result = await self.db.execute(select(models.User))
        return result.scalars().all()

    async def update_user(self, user_id: int, user: user_schema.UserUpdate):
        db_user = await self.db.get(models.User, user_id)
        if db_user:
            for key, value in user.dict(exclude_unset=True).items():
                setattr(db_user, key, value)
            await self.db.commit()
            await self.db.refresh(db_user)
            return db_user
        return None

    async def delete_user(self, user_id: int):
        db_user = await self.db.get(models.User, user_id)
        if db_user:
            await self.db.delete(db_user)
            await self.db.commit()
            return True
        return False

