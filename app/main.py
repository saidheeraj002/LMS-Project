from fastapi import FastAPI
from app.routers import api_router
from app.db_manager import engine, Base
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="AI-Based LMS Backend", version="0.1.0")

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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




# main.py

# import os
# from contextlib import asynccontextmanager
# from typing import Dict, Any
# from fastapi import FastAPI, Depends as FastAPIDepends
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
#
# # --- App Specific Imports ---
# from app.routers import api_router
# from app.db_manager import engine, Base # Assuming async setup
#
# # --- Import the initializer and shared services dict from the new module ---
# from app.core.ai_services import initialize_ai_services, shared_services
#
# # --- Load Environment Variables ---
# # Load .env file at the very beginning, before any initializations
# load_dotenv()
# print("Attempted to load environment variables from .env file.")
#
#
# # --- FastAPI Lifespan Function ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Initialize shared AI/Langchain resources on startup using the dedicated function.
#     """
#     print("Application startup: Initializing shared AI resources via lifespan...")
#     # Call the initialization function from the ai_services module
#     await initialize_ai_services()
#     print("--- Lifespan: Shared AI resources initialization complete ---")
#
#     yield # Application runs here
#
#     # --- Cleanup ---
#     print("Application shutdown: Cleaning up resources...")
#     shared_services.clear() # Clear the dict on shutdown
#
#
# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="AI-Based LMS Backend",
#     version="0.1.0",
#     lifespan=lifespan # Register the lifespan handler
#     )
#
# # --- Include Routers ---
# app.include_router(api_router)
#
# # --- CORS Middleware ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Be more specific in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # --- Root Endpoint ---
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the AI-Based LMS Backend!"}
#
# # --- Database Startup Event (Keep as is) ---
# @app.on_event("startup")
# async def startup_db_event():
#     print("Running database startup event...")
#     try:
#         async with engine.begin() as conn:
#             await conn.run_sync(Base.metadata.create_all)
#             print("Database tables checked/created.")
#     except Exception as e:
#         print(f"Error during database startup: {e}")
#
# # --- Main Execution ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Use reload=True for dev