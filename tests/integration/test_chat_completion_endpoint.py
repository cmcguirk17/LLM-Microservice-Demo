import requests


BASE_URL = "http://localhost:8000"


def test_chat_completion_success():
    request_payload = {
        "messages": [
            {"role": "user", "content": "Hello! Who won the Stanley Cup in 2021?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0,
    }

    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=request_payload)

    if response.status_code == 503:
        print("LLM not loaded. Skipping chat completion test.")
        return

    assert response.status_code == 200

    data = response.json()
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) > 0

    first_choice = data["choices"][0]
    assert "message" in first_choice
    assert first_choice["message"]["role"] == "assistant"
    assert len(first_choice["message"]["content"]) > 0


def test_chat_completion_empty_messages():
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert "Messages list cannot be empty." in response.text
