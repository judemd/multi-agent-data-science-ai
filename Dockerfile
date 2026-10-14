FROM python:3.11-slim

WORKDIR /app

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

EXPOSE 8080

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=true"]
