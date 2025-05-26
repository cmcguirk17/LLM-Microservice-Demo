FROM python:3.10-slim

# Prevent apt-get from asking questions during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Dependencies
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    curl \
    build-essential \
    git && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV POETRY_VERSION=2.0.0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# Copy the application code into the container
WORKDIR /src
COPY app/ /src/app/
# Copy pyproject and lock file
COPY ./README.md ./pyproject.toml ./poetry.lock /src/


# Install/Setup Poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Build venvironment
RUN poetry install --no-interaction --no-root

# Expose the port the app runs on
EXPOSE 8000

WORKDIR /src/app/

# Command to run the application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --workers for production concurrency
CMD ["poetry", "run", "uvicorn", "main:app_fastapi", "--host", "0.0.0.0", "--port", "8000"]
