import os
import time
import logging
import asyncio
import functools
from fastapi import APIRouter, HTTPException, Request

# Import schemas, dependencies, and config
from core.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageOutput,
)

from core.dependencies import LLMDependency, LLMLockDependency
from core.config import MODEL_PATH

# Get a logger for this module
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check_status(request: Request) -> dict:
    """
    Provides a health check endpoint for the LLM service.
    Verifies if the LLM is loaded.
    """
    logger.debug("Health check endpoint called.")
    # Check app.state directly for the health status
    llm_instance_available = (
        hasattr(request.app.state, "llm") and request.app.state.llm is not None
    )

    model_name_display = "N/A"
    if llm_instance_available and MODEL_PATH and os.path.exists(MODEL_PATH):
        model_name_display = os.path.basename(MODEL_PATH)
    elif MODEL_PATH and not os.path.exists(MODEL_PATH):
        model_name_display = f"{os.path.basename(MODEL_PATH)} (file not found)"

    if llm_instance_available:
        logger.info("Health check: OK - Model loaded.")
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": model_name_display,
        }
    else:
        logger.warning("Health check: WARNING - Model not loaded.")
        return {
            "status": "warning",
            "model_loaded": False,
            "message": "LLM not loaded or failed to load. Check server logs.",
            "configured_model_path": MODEL_PATH,
        }


@router.post("/chat/completions", response_model=ChatCompletionResponse, tags=["LLM"])
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    llm: LLMDependency,
    llm_lock: LLMLockDependency,
) -> ChatCompletionResponse:
    """
    Generates a chat completion response from the loaded Large Language Model.
    """
    logger.info(
        f"Received chat completion request with {len(request_data.messages)} messages."
    )

    if not request_data.messages:
        logger.warning("Received empty messages list for chat completion.")
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    logger.debug(
        f"Chat completion request details: {request_data.model_dump_json(indent=2)}"
    )

    # Prepare messages for the Llama model
    messages_for_llm = [message.model_dump() for message in request_data.messages]

    async with llm_lock:
        logger.info("Acquired lock.")

        try:
            logger.info("Offloading LLM inference to thread pool executor.")
            start_inference_time = time.time_ns()

            # functools.partial is used to pre-fill KEY\VALUE arguments
            blocking_task = functools.partial(
                llm.create_chat_completion,  # The blocking function from llama_cpp
                messages=messages_for_llm,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                top_p=request_data.top_p,
                stop=request_data.stop,
            )

            # Run the blocking LLM call in a separate thread
            loop = asyncio.get_event_loop()
            completion_result = await loop.run_in_executor(
                None, blocking_task
            )  # ThreadPoolExecutor

            total_inference_time_ms = round(
                (time.time_ns() - start_inference_time) * 1e-6, 3
            )
            logger.info(f"LLM processing completed in {total_inference_time_ms} ms.")

            # Process LLM response
            if not isinstance(completion_result, dict):
                logger.error(f"Unexpected LLM response structure: {completion_result}")
                raise HTTPException(
                    status_code=500, detail="Invalid response structure from LLM."
                )

            first_choice = completion_result["choices"][0]
            llm_message = first_choice["message"]

            response_message = ChatMessageOutput(
                role=llm_message.get(
                    "role", "assistant"
                ),  # Default to assistant if role is missing
                content=llm_message.get(
                    "content", ""
                ),  # Default to empty string if content is missing
            )

            choice = ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason=first_choice.get(
                    "finish_reason", "stop"
                ),  # Default to stop
            )

            model_name_display = "unknown_model"
            if MODEL_PATH and os.path.exists(MODEL_PATH):
                model_name_display = os.path.basename(MODEL_PATH)

            logger.info("Successfully generated chat completion response.")
            return ChatCompletionResponse(
                id=completion_result.get(
                    "id", f"chatcmpl-{int(time.time())}"
                ),  # Use LLM id or generate one
                created=completion_result.get(
                    "created", int(time.time())
                ),  # Use LLM created or generate one
                model=model_name_display,
                choices=[choice],
            )

        except HTTPException:  # Re-raise HTTPExceptions
            raise
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during chat completion: {e}"
            )  # Get tracebacl
