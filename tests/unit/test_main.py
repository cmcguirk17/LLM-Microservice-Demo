import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app_fastapi, lifespan, APP_TITLE
from app.core import config as app_config


def test_read_root():
    client = TestClient(app_fastapi)  # TestClient will run the lifespan
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": f"Welcome to {APP_TITLE}. Visit /docs for API details."
    }


@pytest.mark.asyncio
async def test_lifespan_model_loads_successfully(mocker):
    """
    Test the lifespan when the model file exists and Llama loads successfully.
    """

    # Mock os.path.exists to return True
    mocker.patch("os.path.exists", return_value=True)

    # Mock the Llama class and instance
    mock_llama_instance = MagicMock()
    mock_Llama_class = mocker.patch(
        "app.main.Llama", return_value=mock_llama_instance
    )  # Patch Llama in app.main

    # Mock time.time_ns for consistent time
    mocker.patch(
        "time.time_ns", side_effect=[0, 1_000_000_000]
    )  # Start and end time (1 second)

    # Create a dummy FastAPI app instance for the lifespan context
    test_app = FastAPI()

    async with lifespan(test_app):
        mock_Llama_class.assert_called_once_with(
            model_path=app_config.MODEL_PATH,
            n_gpu_layers=app_config.N_GPU_LAYERS,
            n_ctx=app_config.N_CTX,
            n_threads=app_config.N_THREADS,
            verbose=app_config.VERBOSE_LLAMA,
        )
        assert hasattr(test_app.state, "llm")
        assert test_app.state.llm is mock_llama_instance
        # print("Lifespan startup with successful mock load completed.")

    # Check if app.state.llm was cleaned up (it would be None or attribute deleted)
    assert not hasattr(test_app.state, "llm") or test_app.state.llm is None
    # print("Lifespan shutdown completed.")


@pytest.mark.asyncio
async def test_lifespan_model_file_not_found(mocker):
    """
    Test the lifespan when the model file does NOT exist.
    """

    # Mock os.path.exists to return False
    mocker.patch("os.path.exists", return_value=False)

    # Mock the Llama class
    mock_Llama_class = mocker.patch("app.main.Llama")

    test_app = FastAPI()

    async with lifespan(test_app):
        mock_Llama_class.assert_not_called()  # Llama should not be instantiated
        assert hasattr(test_app.state, "llm")  # app.state.llm is set
        assert test_app.state.llm is None  # but it's None
        # print("Lifespan startup with model not found completed.")

    assert not hasattr(test_app.state, "llm") or test_app.state.llm is None
    # print("Lifespan shutdown completed.")


@pytest.mark.asyncio
async def test_lifespan_model_load_fails_exception(mocker):
    """
    Test the lifespan when Llama class raises an exception.
    """

    mocker.patch("os.path.exists", return_value=True)  # Model file "exists"
    mock_Llama_class = mocker.patch(
        "app.main.Llama", side_effect=Exception("Test LLM Load Error")
    )  # Llama init fails

    test_app = FastAPI()

    async with lifespan(test_app):
        mock_Llama_class.assert_called_once()  # Attempted to load
        assert hasattr(test_app.state, "llm")
        assert test_app.state.llm is None  # LLM is None due to exception
        # print("Lifespan startup with model load exception completed.")

    assert not hasattr(test_app.state, "llm") or test_app.state.llm is None
    # print("Lifespan shutdown completed.")
