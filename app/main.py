from fastapi import FastAPI
from app.routers import api_router
from app.db_manager import engine, Base


app = FastAPI(title="AI-Based LMS Backend", version="0.1.0")

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Based LMS Backend!"}

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:  # Use async with engine.begin()
        await conn.run_sync(Base.metadata.create_all) # Use await conn.run_sync



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)