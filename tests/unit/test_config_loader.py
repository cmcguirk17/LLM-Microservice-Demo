import pytest
from yaml import YAMLError
from pydantic import HttpUrl, ValidationError
from core.general import load_config_from_yaml, ConfigError
from core.schemas import ClientConfig


def test_load_valid_config(tmp_path):
    yaml_content = """
    service_url: "http://localhost:8000/v1/chat/completions"
    request_timeout: 100
    generation_params:
      temperature: 0.5
      max_tokens: 150
    client_log_level: "DEBUG"
    default_system_prompt: "You are an assistant."
    """

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    config = load_config_from_yaml(str(config_file))
    assert isinstance(config, ClientConfig)
    assert config.service_url == HttpUrl("http://localhost:8000/v1/chat/completions")
    assert config.request_timeout == 100
    assert config.generation_params.temperature == 0.5
    assert config.generation_params.max_tokens == 150
    assert config.client_log_level == "DEBUG"
    assert config.default_system_prompt == "You are an assistant."


def test_load_config_malformed_yaml(tmp_path):
    # YAML that cannot be parsed
    yaml_content = """
    service_url: "http://localhost:1234"
    unclosed_key: { # Malformed YAML
    """
    config_file = tmp_path / "malformed_config.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ConfigError) as excinfo:  # custom ConfigError
        load_config_from_yaml(str(config_file))

    assert "Error parsing YAML" in str(excinfo.value)  # validate expected msg
    assert isinstance(excinfo.value.__cause__, YAMLError)  # check the wrapped error


def test_config_with_defaults(tmp_path):
    yaml_content = """
    request_timeout: 200
    """

    config_file = tmp_path / "default_config.yaml"
    config_file.write_text(yaml_content)

    config = load_config_from_yaml(str(config_file))
    assert config.request_timeout == 200
    assert config.generation_params.temperature == 0.7  # Default
    assert config.generation_params.max_tokens == 250  # Default
    assert config.client_log_level == "INFO"  # Default
