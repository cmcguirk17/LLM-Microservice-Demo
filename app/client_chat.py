import requests
import json
import logging
import time
import sys
from typing import Optional

from core.general import load_config_from_yaml
from core.schemas import ClientConfig


# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LLMChatClient:
    """
    A client for interacting with an LLM service and maintaining conversation history.
    """

    def __init__(
        self,
        service_url: str,
        request_timeout: int,
        default_system_prompt: Optional[str] = None,
    ):
        """
        Initializes the LLM chat client.

        Args:
            service_url (str): The base URL of the LLM inference service.
            request_timeout (int): Timeout in seconds for requests to the LLM service.
            default_system_prompt (Optional[str], optional): An optional initial system prompt, default None.
        """
        self.service_url = service_url
        self.request_timeout = request_timeout
        self.conversation_history = []

        if default_system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": default_system_prompt}
            )
            logger.info(f'Using system prompt: "{default_system_prompt}"')

        logger.info(
            f"LLM Chat Client initialized. Service URL: {self.service_url}, Timeout: {self.request_timeout}s"
        )

    def add_user_message(self, content: str) -> None:
        """
        Adds a message to the user role conversation history.

        Args:
            content (str): The text content of the message.
        """
        self.conversation_history.append({"role": "user", "content": content})
        logger.debug(f"Added user message: {content}")

    def add_assistant_message(self, content: str) -> None:
        """
        Adds a message to the assistant role conversation history.

        Args:
            content (str): The text content of the message.
        """
        self.conversation_history.append({"role": "assistant", "content": content})
        logger.debug(f"Added assistant message: {content}")

    def get_llm_response(self, temperature: float, max_tokens: int) -> str:
        """
        Sends the current conversation history to the LLM service and returns the response.

        Args:
            temperature (float): The temperature setting for the LLM generation.
            max_tokens (int): The maximum number of tokens for the LLM to generate.

        Returns:
            str: The content of the response, or an error message
        """
        if (
            not self.conversation_history
            or self.conversation_history[-1]["role"] != "user"
        ):
            logger.warning("Cannot get LLM response without a preceding user message.")
            return "Error: Internal client error - no user message to respond to."

        payload = {
            "messages": self.conversation_history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(
            f"Sending request to LLM. Temp: {temperature}, Max Tokens: {max_tokens}. History len: {len(self.conversation_history)}"
        )
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            start_response_time = time.time_ns()
            response = requests.post(
                self.service_url, json=payload, timeout=self.request_timeout
            )
            response_time = round((time.time_ns() - start_response_time) * 1e-6, 3)
            logger.info(
                f"Received response from LLM service in {response_time} ms. Status: {response.status_code}"
            )

            response.raise_for_status()

            response_data = response.json()
            logger.debug(f"Response Data: {json.dumps(response_data, indent=2)}")

            assistant_message_data = response_data["choices"][0]["message"]
            self.add_assistant_message(assistant_message_data["content"])
            return assistant_message_data["content"]

        except (KeyError, IndexError, TypeError) as e:
            logger.exception(f"Error parsing LLM response: {e}")
            self._revert_last_user_message()
            return "Error: Received an invalid response from the AI."
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            self._revert_last_user_message()
            return "Error: An unexpected problem occurred."

    def _revert_last_user_message(self) -> None:
        """
        Removes the last user message from history if an error occurred
        """
        if (
            self.conversation_history
            and self.conversation_history[-1]["role"] == "user"
        ):
            self.conversation_history.pop()
            logger.info("Reverted last user message due to error.")

    def clear_history(self, system_prompt: Optional[str] = None) -> None:
        """
        Clears the current conversation history.

        Args:
            system_prompt (Optional[str]): A new system prompt to start
        """
        self.conversation_history = []
        if system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": system_prompt}
            )
            logger.info(f'History cleared. Using new system prompt: "{system_prompt}"')
        else:
            logger.info(
                "Conversation history cleared. No new system prompt applied (client will use its initial one if any)."
            )

    def print_history(self) -> None:
        """
        Prints the current conversation history as JSON format.
        """
        print("\n--- Conversation History ---")
        print(json.dumps(self.conversation_history, indent=2))
        print("---------------------------\n")


def run_chat_loop(config: ClientConfig):
    """
    Runs the main interactive chat loop.
    The loop continues until the user types '/exit'.

    Commands:
        /exit: Terminates the chat session.
        /history: Prints the current conversation history.
        /clear: Clears the conversation history and allows setting a new system prompt.
        /temp <value>: Sets the temperature for the next LLM generation.
        /tokens <value>: Sets the maximum tokens for the next LLM generation.

    Args:
        config (ClientConfig): The loaded configuration object.
    """

    # Determine initial system prompt: user input > config default > none
    system_prompt_from_config = config.default_system_prompt

    prompt_message = "Enter a system prompt for this session"
    if system_prompt_from_config:
        prompt_message += (
            f" (or press Enter to use default: '{system_prompt_from_config[:50]}...') "
        )
    else:
        prompt_message += " (or press Enter for none): "

    user_system_prompt_input = input(prompt_message).strip()

    chosen_initial_system_prompt = None
    if user_system_prompt_input:
        chosen_initial_system_prompt = user_system_prompt_input
    elif system_prompt_from_config:
        chosen_initial_system_prompt = system_prompt_from_config

    # Initialize client with values from config
    client = LLMChatClient(
        service_url=str(
            config.service_url
        ),  # Ensure str after Pydantic validate with HttpUrl
        request_timeout=config.request_timeout,
        default_system_prompt=chosen_initial_system_prompt,
    )

    print(
        "\nStarting chat. Type '/exit' to end, '/history' to show history, '/clear' to reset."
    )
    print(
        "You can also use '/temp <value>' or '/tokens <value>' to change generation parameters for the next turn."
    )

    # Use generation defaults from config for the first turn if not overridden by commands
    current_turn_temp: Optional[float] = config.generation_params.temperature
    current_turn_tokens: Optional[int] = config.generation_params.max_tokens

    while True:
        try:
            user_input_raw = input("You: ")
        except EOFError:
            print("\nExiting chat (EOF).")
            break
        except KeyboardInterrupt:
            print("\nExiting chat (Interrupted).")
            break

        user_input = user_input_raw.strip()
        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Exiting chat.")
            break
        elif user_input.lower() == "/history":
            client.print_history()
            continue
        elif user_input.lower() == "/clear":
            clear_prompt_message = "Enter new system prompt for cleared chat"
            if config.default_system_prompt:
                clear_prompt_message += f" (or Enter to use default: '{config.default_system_prompt[:50]}...')"
            else:
                clear_prompt_message += " (or Enter for none)"

            new_sys_prompt_input = input(f"{clear_prompt_message}: ").strip()

            new_chosen_sys_prompt_for_clear = None
            if new_sys_prompt_input:
                new_chosen_sys_prompt_for_clear = new_sys_prompt_input
            elif config.default_system_prompt:  # Re-apply default from config on clear
                new_chosen_sys_prompt_for_clear = config.default_system_prompt

            client.clear_history(system_prompt=new_chosen_sys_prompt_for_clear)
            # Reset turn-specific params to defaults from config
            current_turn_temp = config.generation_params.temperature
            current_turn_tokens = config.generation_params.max_tokens
            print("Chat history cleared.")
            continue
        elif user_input.lower().startswith("/temp "):
            try:
                temp_val = float(user_input.split(" ", 1)[1])
                if 0.0 <= temp_val <= 2.0:  # could validate from pydantic schema?
                    current_turn_temp = temp_val
                    print(f"Temperature for next turn set to: {current_turn_temp}")
                else:
                    print("Invalid temperature. Must be between 0.0 and 2.0.")
            except (ValueError, IndexError):
                print("Invalid format. Use: /temp <value_float>")
            continue
        elif user_input.lower().startswith("/tokens "):
            try:
                tokens_val = int(user_input.split(" ", 1)[1])
                if tokens_val > 0:
                    current_turn_tokens = tokens_val
                    print(f"Max tokens for next turn set to: {current_turn_tokens}")
                else:
                    print("Invalid max_tokens. Must be a positive integer.")
            except (ValueError, IndexError):
                print("Invalid format. Use: /tokens <value_int>")
            continue

        client.add_user_message(user_input)
        assistant_response = client.get_llm_response(
            temperature=(
                current_turn_temp
                if current_turn_temp is not None
                else config.generation_params.temperature
            ),
            max_tokens=(
                current_turn_tokens
                if current_turn_tokens is not None
                else config.generation_params.max_tokens
            ),
        )

        current_turn_temp = config.generation_params.temperature
        current_turn_tokens = config.generation_params.max_tokens

        print(f"AI: {assistant_response}")

    client.print_history()


if __name__ == "__main__":
    # Configuration file loading
    try:
        config_data: ClientConfig = load_config_from_yaml()
    except Exception as e:
        logger.exception(
            f"Unhandled exception during configuration loading: {e}. Exiting."
        )
        sys.exit(1)

    # Set logger
    try:
        configured_log_level = config_data.client_log_level.upper()
        logger.setLevel(configured_log_level)
        # Set root logger level to affect other loggers (e.g., from 'util' package)
        logging.getLogger().setLevel(configured_log_level)
        logger.info(f"Client log level set to: {configured_log_level} from config.")
    except Exception as e:
        logger.exception(
            f"Error setting log level from config: {e}. Using initial basicConfig level."
        )

    # Run the application
    try:
        run_chat_loop(config_data)

    except Exception as e:
        logger.exception(f"An unexpected error occurred in the main chat loop: {e}")
        sys.exit(1)
