from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings
from fastapi import HTTPException

engine = create_async_engine(
    settings.DATABASE_URL)
AsyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

Base = declarative_base()

async def get_db():
    print("Entering get_db()")
    try:
        db = AsyncSessionLocal()
        yield db
    except Exception as e:
        print(f"Exception in get_db: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        await db.close()
        print("Exiting get_db()")