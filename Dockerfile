FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*

# Install uv for reproducible dependency installation.
RUN pip install --no-cache-dir uv

# Install locked project dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source (filtered by .gcloudignore).
COPY . .

# Cloud Run supplies PORT, normally 8080.
ENV PORT=8080
ENV PATH="/app/.venv/bin:$PATH"
ENV DTALE_APP_ROOT=/explorer

EXPOSE 8080

CMD ["python", "deploy/serve.py"]
