import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
import asyncio

from fastapi import FastAPI
from llama_cpp import Llama

# Import configurations | Logger is loaded here
from core.config import (
    MODEL_PATH,
    N_GPU_LAYERS,
    N_CTX,
    N_THREADS,
    VERBOSE_LLAMA,
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
)

from api.v1 import endpoints as api_v1_endpoints

# Get a logger for this module
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the LLM application.
    """
    logger.info("Application Startup")
    logger.info("Attempting to load LLM.")
    logger.info("Configuration:")
    logger.info(f"  MODEL_PATH: {MODEL_PATH}")
    logger.info(f"  N_GPU_LAYERS: {N_GPU_LAYERS}")
    logger.info(f"  N_CTX: {N_CTX}")
    logger.info(f"  N_THREADS: {N_THREADS}")
    logger.info(f"  VERBOSE_LLAMA: {VERBOSE_LLAMA}")

    llm_instance: Optional[Llama] = None
    if not os.path.exists(MODEL_PATH):
        error_message = (
            f"Model file not found at {MODEL_PATH}. LLM will not be available."
        )
        logger.error(error_message)
        # app.state.llm will remain None or not be set
    else:
        logger.info("Model file found. Initializing Llama model...")
        try:
            start_time = time.time_ns()
            llm_instance = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=N_GPU_LAYERS,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                verbose=VERBOSE_LLAMA,
            )
            load_time = round((time.time_ns() - start_time) * 1e-6, 3)
            logger.info(f"LLM loaded successfully in {load_time} ms.")
            logger.info(f"Model Name: {os.path.basename(MODEL_PATH)}")
        except Exception as e:
            logger.exception(f"Failed to load LLM: {e}. LLM will not be available.")
            # app.state.llm will remain None or not be set

    app.state.llm = llm_instance  # Store llm_instance (or None) in app.state
    if llm_instance:
        app.state.llm_lock = asyncio.Lock()  # Create a lock
        logger.info("LLM Lock initialized.")
    else:
        app.state.llm_lock = None  # If llm not loaded, set lock as None
        logger.info("LLM Lock not initialized as LLM failed to load or was not found.")

    logger.info("Application startup complete. Model loading is done.")

    # Code above yield is executed on application start

    yield  # FastAPI application starts serving requests after this point

    # Code below yield is executed when the application exits (ex. ctrl+c)

    logger.info("Application Shutdown Sequence")
    logger.info("Cleaning up LLM instance...")
    if hasattr(app.state, "llm") and app.state.llm:
        del app.state.llm
        logger.info("LLM instance reference removed from app.state.")
    else:
        logger.info("No LLM instance was loaded or it was already cleaned up.")
    logger.info("Shutdown complete.")


app_fastapi = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)


# Include API routers
app_fastapi.include_router(api_v1_endpoints.router, prefix="/v1", tags=["v1"])


# Basic root endpoint
@app_fastapi.get("/", tags=["General"])
async def read_root():
    return {"message": f"Welcome to {APP_TITLE}. Visit /docs for API details."}


if __name__ == "__main__":
    # If this is run through python main.py then we will import uvicorn here
    import uvicorn

    logger.info("Starting Uvicorn server for Llama.cpp microservice...")
    logger.info(
        "Access API documentation at http://localhost:8000/docs or http://localhost:8000/redoc"
    )

    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)
