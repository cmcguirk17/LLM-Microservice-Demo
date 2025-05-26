import yaml
import os
import logging
from pydantic import ValidationError  # Important to import for specific catching

from core.schemas import ClientConfig


logger = logging.getLogger(__name__)


# Define some custom exceptions
class ConfigError(Exception):
    """Custom exception for configuration loading errors."""

    pass


class ConfigFileNotFoundError(ConfigError, FileNotFoundError):
    """Custom exception for when the config file is not found."""

    pass


def load_config_from_yaml(
    config_path: str = "app/client_chat_config.yaml",
) -> ClientConfig:
    """
    Loads configuration from a YAML file into a Pydantic model.
    Raises ConfigFileNotFoundError, ConfigError (wrapping YAMLError or ValidationError),
    or RuntimeError for other issues.
    """

    if not os.path.exists(config_path):
        msg = f"Configuration file '{config_path}' not found."
        logger.error(msg)
        raise ConfigFileNotFoundError(msg)

    try:
        with open(config_path, "r") as f:
            raw_config_data = yaml.safe_load(f)

        if raw_config_data is None:
            raw_config_data = {}
            logger.warning(
                f"Configuration file '{config_path}' is empty or contains only comments. Proceeding with empty config data."
            )

        logger.info(f"Attempting to load configuration from '{config_path}'.")
        config_model = ClientConfig(
            **raw_config_data
        )  # valErr if we don't match pydantic model
        logger.info(f"Configuration loaded successfully from '{config_path}'.")
        return config_model

    except yaml.YAMLError as e:
        msg = f"Error parsing YAML in configuration file '{config_path}': {e}"
        logger.error(msg)
        raise ConfigError(msg) from e

    except ValidationError as e:
        msg = f"Configuration validation error for '{config_path}': {e}"
        logger.error(msg)
        raise ConfigError(msg) from e

    except Exception as e:  # Catch other potential err
        msg = f"An unexpected error occurred while loading configuration from '{config_path}': {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
