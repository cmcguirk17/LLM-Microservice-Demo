import os
import logging


# Logger Setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# configure root logger
setup_logging()
logger = logging.getLogger(__name__)

# Model Configuration
# Default settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
N_CTX = int(os.getenv("N_CTX", 4096))

# Default N_THREADS to a safe value if os.cpu_count() returns None
_default_threads = 4
try:
    _cpu_count = os.cpu_count()
    if _cpu_count is not None:
        _default_threads = _cpu_count
except NotImplementedError:
    logger.warning("os.cpu_count() not implemented, defaulting N_THREADS to 4.")

N_THREADS = int(os.getenv("N_THREADS", _default_threads))
VERBOSE_LLAMA = os.getenv("VERBOSE_LLAMA", "true").lower() in ("true", "1", "t")

# Application Metadata
APP_TITLE = "Llama.cpp FastAPI Microservice"
APP_DESCRIPTION = "A RESTful API to interact with a GGUF model via llama-cpp-python."
APP_VERSION = "0.1.0"
