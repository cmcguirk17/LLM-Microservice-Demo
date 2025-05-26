import pytest
from unittest.mock import MagicMock
import logging

from client_chat import LLMChatClient

logger = logging.getLogger(__name__)


# Fixtures


@pytest.fixture
def mock_requests_post(mocker):
    """Fixture to mock requests.post."""
    return mocker.patch("app.client_chat.requests.post")


@pytest.fixture
def base_client():
    """LLMChatClien fixture."""
    return LLMChatClient(
        service_url="http://fake-llm-service.com/api", request_timeout=10
    )


# Tests


def test_client_initialization(base_client):
    """Test basic client initialization."""
    assert base_client.service_url == "http://fake-llm-service.com/api"
    assert base_client.request_timeout == 10
    assert base_client.conversation_history == []


def test_client_initialization_with_system_prompt():
    """Test client initialization with a default system prompt."""
    client = LLMChatClient("http://url.com", 5, default_system_prompt="Be concise.")
    assert client.conversation_history == [{"role": "system", "content": "Be concise."}]


def test_add_user_message(base_client):
    """Test adding a user message."""
    base_client.add_user_message("Hello, world!")
    assert base_client.conversation_history == [
        {"role": "user", "content": "Hello, world!"}
    ]


def test_add_assistant_message(base_client):
    """Test adding an assistant message."""
    base_client.add_user_message("User question")
    base_client.add_assistant_message("AI answer")
    assert base_client.conversation_history == [
        {"role": "user", "content": "User question"},
        {"role": "assistant", "content": "AI answer"},
    ]


def test_get_llm_response_successful(base_client, mock_requests_post):
    """Test a successful call to get_llm_response."""
    base_client.add_user_message("What does AI stand for?")

    mock_api_response = MagicMock()  # simulate the 'response'
    mock_api_response.status_code = 200
    # Define what response.json() should return
    mock_api_response.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Artificial Intelligence."}}
        ]
    }
    mock_requests_post.return_value = (
        mock_api_response  # Make requests.post return mock
    )

    llm_answer = base_client.get_llm_response(temperature=0.5, max_tokens=50)

    assert llm_answer == "Artificial Intelligence."
    mock_requests_post.assert_called_once()  # Ensure requests.post was actually called
    # Check that the assistant's response was added to the history
    assert base_client.conversation_history[-1] == {
        "role": "assistant",
        "content": "Artificial Intelligence.",
    }


def test_get_llm_response_without_user_message_first(base_client, mock_requests_post):
    """Test calling get_llm_response when the last message isn't from the user."""
    # (Covers no history, or last message was 'system' or 'assistant')
    llm_answer = base_client.get_llm_response(temperature=0.5, max_tokens=50)

    assert "Error: Internal client error - no user message to respond to." in llm_answer
    mock_requests_post.assert_not_called()  # API should not have been called


def test_get_llm_response_bad_json_structure(base_client, mock_requests_post):
    """Test handling of unexpected JSON structure from the LLM service."""
    base_client.add_user_message("Ask something.")
    user_messages_before_call = len(base_client.conversation_history)

    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.json.return_value = {
        "error": "unexpected format",
        "data": None,
    }  # Missing 'choices'
    mock_requests_post.return_value = mock_api_response

    llm_answer = base_client.get_llm_response(temperature=0.5, max_tokens=50)

    assert "Error: Received an invalid response from the AI." in llm_answer
    assert len(base_client.conversation_history) == user_messages_before_call - 1


def test_clear_history(base_client):
    """Test clearing the conversation history."""
    base_client.add_user_message("Message 1")
    base_client.add_assistant_message("Response 1")

    base_client.clear_history()
    assert base_client.conversation_history == []


def test_clear_history_with_new_system_prompt(base_client):
    """Test clearing history and setting a new system prompt."""
    base_client.add_user_message("Old context")

    base_client.clear_history(system_prompt="This is a new context.")
    assert base_client.conversation_history == [
        {"role": "system", "content": "This is a new context."}
    ]
