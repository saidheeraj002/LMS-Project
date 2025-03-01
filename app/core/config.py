from pydantic_settings import BaseSettings
from datetime import timedelta

class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+asyncmy://dheeraj:dheeraj@127.0.0.1/deal_db"  # Replace with your database URL
    SECRET_KEY: str = "secret"  # Replace with your secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"

settings = Settings()