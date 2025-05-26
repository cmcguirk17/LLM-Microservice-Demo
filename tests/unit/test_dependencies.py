import pytest
from unittest.mock import MagicMock
from fastapi import Request, HTTPException, FastAPI  # Added FastAPI for app instance
from llama_cpp import Llama  # For type hinting

from core.dependencies import get_llm_instance  # Function to test


@pytest.mark.asyncio
async def test_get_llm_instance_success():
    """Test LLM is returned when available in app.state."""
    mock_llm_object = MagicMock(spec=Llama)

    test_app = FastAPI()
    test_app.state.llm = mock_llm_object  # Set the llm on the app's state

    # The scope needs a reference to the application instance
    mock_scope = {"type": "http", "app": test_app}
    mock_request = Request(scope=mock_scope)

    retrieved_llm = await get_llm_instance(mock_request)
    assert retrieved_llm is mock_llm_object


@pytest.mark.asyncio
async def test_get_llm_instance_llm_is_none():
    """Test HTTPException 503 if app.state.llm is None."""
    test_app = FastAPI()
    test_app.state.llm = None  # LLM attribute exists but is None

    mock_scope = {"type": "http", "app": test_app}
    mock_request = Request(scope=mock_scope)

    with pytest.raises(HTTPException) as exc_info:
        await get_llm_instance(mock_request)

    assert exc_info.value.status_code == 503
    assert "LLM model is not loaded" in exc_info.value.detail
