"""Client for the private LLM: the `hospital-private-llm` Model Serving endpoint, which runs a small
open-weight model inside this workspace (see 04_serving/01_private_llm_endpoint.ipynb).

PHI prompts and retrieved records go only to this endpoint. There is deliberately no fallback to any other
model: if it fails, the request fails."""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

PRIVATE_ENDPOINT = os.environ.get("PRIVATE_LLM_ENDPOINT", "hospital-private-llm")

# The endpoint scales to zero, so the first call after idle can take minutes. A long HTTP timeout plus a few
# retries covers the cold start; the app answers asynchronously, so this never blocks a web request.
_client = WorkspaceClient(config=Config(http_timeout_seconds=300))

status = {"state": "unknown", "endpoint": PRIVATE_ENDPOINT, "error": None}

_RETRYABLE = ("503", "429", "timed out", "timeout", "scal", "not ready", "temporarily", "bad gateway", "502", "504")


def refresh_status() -> None:
    """Best-effort: needs permission to read the endpoint, which the app may not have."""
    try:
        ep = _client.serving_endpoints.get(PRIVATE_ENDPOINT)
        status.update(state=ep.state.ready.value.lower() if ep.state and ep.state.ready else "unknown", error=None)
    except Exception:
        status.update(state="unknown")


def generate(question: str, context: list[str]) -> str:
    # Delimited so the model treats the records as data, not instructions.
    records = "<records>\n" + "\n".join(f"- {c}" for c in context) + "\n</records>"
    last = None
    for attempt in range(1, 4):
        try:
            r = _client.serving_endpoints.query(
                name=PRIVATE_ENDPOINT,
                dataframe_records=[{"question": question, "records": records}],
            )
            status.update(state="ready", error=None)
            return str(r.predictions[0]).strip()
        except Exception as e:
            last = e
            if attempt == 3 or not any(k in str(e).lower() for k in _RETRYABLE):
                break
            time.sleep(20)
    status.update(state="failed", error=type(last).__name__)
    raise RuntimeError(f"Private model call failed: {type(last).__name__}") from None
