# Multi-Agent Data Science AI

A Streamlit proof of concept for a human-reviewed data science workflow: dataset
understanding, preparation, model selection, evaluation, and handoff. AI agents
interpret deterministic evidence; human decisions control progression.

## Run locally

Install Python 3.11 and uv, then run:

```powershell
uv sync --frozen
uv run streamlit run app.py --server.port 8502
```

Configure Google Application Default Credentials and the environment variables
`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, and
`GOOGLE_CLOUD_LOCATION` for agent calls. Set `GCS_BUCKET` to use the existing
Cloud Storage persistence; otherwise project data is saved locally under `data/`.
Do not commit credentials or local project exports.

The D-Tale Reference tab starts a separate exploration copy in the background.
Locally its link uses a loopback port. D-Tale changes never become approved
preparation actions or workflow evidence.

## Cloud Run

The container starts Nginx on `$PORT`, routes the workflow to internal Streamlit
port 8501, and routes `/explorer/` to internal D-Tale port 40000. `DTALE_APP_ROOT`
is set by the Dockerfile so explorer links and assets share the app's HTTPS origin.
Use session affinity and one instance because explorer sessions are in memory
and the application currently maintains one shared active workflow.

`cloudbuild.yaml` builds the image and smoke-tests Streamlit health, the D-Tale
page, and its prefixed assets before publishing the image. Supply the full
Artifact Registry image URI as the `_IMAGE` substitution:

```powershell
gcloud builds submit --config=cloudbuild.yaml --substitutions=_IMAGE=IMAGE_URI --region=europe-west2
gcloud run deploy multi-agent-ds-poc --image=IMAGE_URI --region=europe-west2 --session-affinity --max-instances=1
```

Deployment to an existing service preserves its runtime identity and environment
configuration. Handoff approval in the application is separate from deployment.

## Tests

```powershell
uv run pytest -q
```
