import pytest
from core.schemas import (
    ClientConfig,
    GenerationParams,
    ChatMessageInput,
    ChatCompletionRequest,
    ChatMessageOutput,
    ChatCompletionChoice,
    ChatCompletionResponse,
)


### Tests for ClientConfig and Generation Params for the client_chat.py script
def test_client_config_defaults():
    config = ClientConfig()
    assert config.service_url == "http://localhost:8000/v1/chat/completions"
    assert config.request_timeout == 120
    assert config.generation_params.temperature == 0.7
    assert config.client_log_level == "INFO"
    assert config.default_system_prompt is None


def test_generation_params_override():
    params = GenerationParams(temperature=0.5, max_tokens=100)
    assert params.temperature == 0.5
    assert params.max_tokens == 100


###


### Tests for schemas used in main.py for the microservice
def test_valid_chat_message_input():
    msg = ChatMessageInput(role="user", content="Hello!")
    assert msg.role == "user"
    assert msg.content == "Hello!"


def test_invalid_role_chat_message_input():
    with pytest.raises(ValueError):
        ChatMessageInput(role="bot", content="Invalid role")


def test_chat_completion_request_defaults():
    request = ChatCompletionRequest(
        messages=[ChatMessageInput(role="user", content="Hi")]
    )
    assert request.temperature == 0.7
    assert request.max_tokens == 512
    assert request.top_p == 1.0
    assert request.stop is None


def test_chat_completion_response():
    response = ChatCompletionResponse(
        model="mistral",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessageOutput(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
    )
    assert response.model == "mistral"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Hello!"
    assert response.choices[0].finish_reason == "stop"
