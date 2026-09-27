# Databricks notebook source
# MAGIC %md
# MAGIC # AI Gateway — the safety net behind the router
# MAGIC
# MAGIC The router (`03_routing`) makes the primary PHI/non-PHI decision. The AI Gateway is what
# MAGIC catches the case where that decision was wrong — a misclassified intent, a follow-up
# MAGIC question that drags PHI into an otherwise-general thread, a user pasting a chart note into
# MAGIC the box. Every Model Serving endpoint in this project (`04_serving`) is configured with:
# MAGIC
# MAGIC - **Inference tables** — full request/response logging, required for HIPAA audit trails. On the private
# MAGIC   (custom-model) endpoint these come from endpoint telemetry; Free Edition doesn't offer them for the
# MAGIC   external endpoint, where the app's own audit log covers it.
# MAGIC - **PII/PHI detection on input and output** — scans the outbound prompt *before* it leaves
# MAGIC   for an external model, and scans the inbound response for anything that looks like it
# MAGIC   leaked from a private data source.
# MAGIC - **Rate limits and per-role access policies** — so a compromised or over-scoped credential
# MAGIC   can't be used to exfiltrate data at volume even if a single query slips through.

# COMMAND ----------

import requests
from databricks.sdk import WorkspaceClient

# Auth comes from the SDK config so this runs both in a notebook and via Databricks Connect
# (dbutils' notebook context only exists in a real notebook).
_w = WorkspaceClient()
WORKSPACE_URL = _w.config.host.rstrip("/")
HEADERS = _w.config.authenticate()
EXISTING_ENDPOINTS = {e.name for e in _w.serving_endpoints.list()}


# Workspace features a PUT can be refused for (404 FEATURE_DISABLED), matched on the error message.
# Free Edition refuses inference tables on every endpoint type, and AI Guardrails on custom models.
_OPTIONAL = {"Inference table": "inference_table_config", "AI Guardrails": "guardrails"}


def apply_gateway(endpoint: str, config: dict):
    """PUT replaces the endpoint's whole AI Gateway config, so everything is sent in one call. Features the
    workspace refuses are dropped one at a time and the rest re-sent, so what is supported still applies."""
    if endpoint not in EXISTING_ENDPOINTS:
        print(f"{endpoint}: skipped, endpoint does not exist")
        return
    config, skipped = dict(config), []
    while True:
        r = requests.put(f"{WORKSPACE_URL}/api/2.0/serving-endpoints/{endpoint}/ai-gateway", headers=HEADERS, json=config)
        if r.ok:
            break
        msg = r.json().get("message", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        feature = next((f for f in _OPTIONAL if f in msg and _OPTIONAL[f] in config), None)
        if r.status_code != 404 or feature is None:
            raise RuntimeError(f"{endpoint}: AI Gateway update failed ({r.status_code}): {msg[:300]}")
        config.pop(_OPTIONAL[feature])
        skipped.append(feature.lower())
    enabled = []
    if "guardrails" in config:
        g = config["guardrails"]
        enabled += [f"{side} {'+'.join(k for k in ('safety', 'pii') if g.get(side, {}).get(k))}"
                    for side in ("input", "output") if g.get(side)]
    if config.get("usage_tracking_config", {}).get("enabled"):
        enabled.append("usage tracking")
    enabled += [f"rate limit {rl['calls']}/{rl['renewal_period']} per {rl['key']}" for rl in config.get("rate_limits", [])]
    if "inference_table_config" in config:
        enabled.append("inference tables")
    print(f"{endpoint}: enabled {', '.join(enabled) or 'nothing'}"
          + (f"; not supported in this workspace: {', '.join(skipped)}" if skipped else ""))


def inference_table(endpoint: str) -> dict:
    return {"enabled": True, "catalog_name": "hospital_lakehouse", "schema_name": "audit",
            "table_name_prefix": endpoint.replace("-", "_")}

# COMMAND ----------

# MAGIC %md
# MAGIC ## External endpoint — the boundary that matters
# MAGIC Every request leaves the workspace, so it gets the full set: **Safety** and **PII detection (block)** on
# MAGIC both input and output, a per-user rate limit, usage tracking, and an inference table as the audit trail.
# MAGIC Free Edition refuses inference tables (`FEATURE_DISABLED`); the app's own `audit.chat_log` and
# MAGIC `audit.compliance_log` cover the audit trail instead.
# MAGIC
# MAGIC There's no keyword blocklist: the API accepts `invalid_keywords` but doesn't save it. Identifier and
# MAGIC keyword screening happens in the app (`preflight_leak_check` in `app/main.py`) before any hosted call.

# COMMAND ----------

apply_gateway("hospital-external-openai", {
    "guardrails": {
        "input": {"safety": True, "pii": {"behavior": "BLOCK"}},
        "output": {"safety": True, "pii": {"behavior": "BLOCK"}},
    },
    "rate_limits": [{"calls": 30, "key": "user", "renewal_period": "minute"}],
    "usage_tracking_config": {"enabled": True},
    "inference_table_config": inference_table("hospital-external-openai"),
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Private endpoint — inside the boundary
# MAGIC No outbound-leak guard is needed, but input safety would stop an injection attempt in retrieved context.
# MAGIC Free Edition doesn't support AI Guardrails on custom (CPU pyfunc) models, so the gateway gives it a rate
# MAGIC limit and usage tracking; the requested guardrails are dropped automatically if refused. Its inference
# MAGIC table comes from endpoint telemetry instead (next cell), not from the gateway.

# COMMAND ----------

apply_gateway("hospital-private-llm", {
    "guardrails": {"input": {"safety": True}},
    "rate_limits": [{"calls": 60, "key": "user", "renewal_period": "minute"}],
    "usage_tracking_config": {"enabled": True},
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference tables and telemetry — custom-model endpoints
# MAGIC Custom-model endpoints log every request and response through OpenTelemetry, a separate API from the AI
# MAGIC Gateway. That works on Free Edition for the private endpoint: payloads go to
# MAGIC `audit.hospital_private_llm_payload`, and logs, traces and metrics to `audit.hospital_private_llm_otel_*`.
# MAGIC Rows are exported in batches and can take up to an hour to appear. External-model endpoints don't
# MAGIC support this API.
# MAGIC Changing telemetry redeploys the endpoint (a few minutes), so it's only applied when not already set.

# COMMAND ----------

from databricks.sdk.errors import InvalidParameterValue, ResourceConflict
from databricks.sdk.service.serving import TelemetryConfig, TelemetryInferenceTableConfig, UnityCatalogTableNames


# The telemetry tables live in the audit schema; create it if a reset (or a fresh catalog) removed it.
spark.sql("CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.audit COMMENT 'Audit and monitoring tables.'")


def enable_telemetry(endpoint: str):
    if endpoint not in EXISTING_ENDPOINTS:
        print(f"{endpoint}: telemetry skipped, endpoint does not exist")
        return
    prefix = f"hospital_lakehouse.audit.{endpoint.replace('-', '_')}"
    payload_table = f"{prefix}_payload"
    current = _w.serving_endpoints.get(endpoint).telemetry_config
    if current and current.inference_table_config and current.inference_table_config.name == payload_table:
        print(f"{endpoint}: inference table and telemetry already enabled ({payload_table})")
        return
    try:
        _w.serving_endpoints.patch_telemetry_config(endpoint, telemetry_config=TelemetryConfig(
            table_names=UnityCatalogTableNames(logs_table=f"{prefix}_otel_logs", metrics_table=f"{prefix}_otel_metrics",
                                               traces_table=f"{prefix}_otel_spans"),
            inference_table_config=TelemetryInferenceTableConfig(name=payload_table, sampling_fraction=1.0),
        ))
        print(f"{endpoint}: enabled inference table and telemetry ({payload_table}, {prefix}_otel_*); "
              "the endpoint redeploys for a few minutes")
    except InvalidParameterValue as e:
        print(f"{endpoint}: telemetry not supported for this endpoint type ({str(e)[:100]})")
    except ResourceConflict:
        print(f"{endpoint}: endpoint is updating; run this cell again once it's READY")


for endpoint in ["hospital-private-llm", "hospital-external-openai"]:
    enable_telemetry(endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Independent leak check on the general path
# MAGIC Belt-and-suspenders: re-run Presidio against the exact prompt text right before the call to
# MAGIC the external endpoint fires, using the same detector as the ingestion-time auto-tagger
# MAGIC (`01_ingestion/02_pii_phi_autotagger.py`) so the two checks can't silently drift apart.

# COMMAND ----------

# The check that actually runs in production is `preflight_leak_check` in `app/main.py` (deterministic:
# known patient names from the PHI tables + SSN/phone/email/MRN-shaped patterns), applied to the
# question and the retrieved context before any hosted-model call; a hit re-routes to the private
# model. The Presidio version below is the heavier alternative for notebook use.
_analyzer = None

BLOCK_ENTITIES = {"PERSON", "US_SSN", "MEDICAL_LICENSE", "PHONE_NUMBER", "DATE_TIME"}


def pre_flight_leak_check(prompt: str) -> tuple[bool, list[str]]:
    """Returns (is_safe_to_send_externally, flagged_entity_types)."""
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        _analyzer = AnalyzerEngine(
            nlp_engine=NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }).create_engine(),
            supported_languages=["en"],
        )
    results = _analyzer.analyze(text=prompt, language="en")
    flagged = sorted({r.entity_type for r in results if r.entity_type in BLOCK_ENTITIES})
    return (len(flagged) == 0, flagged)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Role-based access to the chat app itself
# MAGIC The AI Gateway governs the *model endpoints*; this grants govern who can reach the app
# MAGIC (`app/chat_app.py`) that fronts them at all.

# COMMAND ----------

# %sql magic is a no-op when this file is run via "Run file", so use spark.sql directly.
# The app reads PHI chunks *as the signed-in user*, so clinical staff need explicit read access.
# Row filters from 00_governance then decide which rows each person sees.
for grant in [
    "GRANT USE CATALOG ON CATALOG hospital_lakehouse TO `clinical_staff`",
    "GRANT USE SCHEMA ON SCHEMA hospital_lakehouse.clinical TO `clinical_staff`",
    "GRANT SELECT ON TABLE hospital_lakehouse.clinical.phi_chunks TO `clinical_staff`",
    "GRANT USE CATALOG ON CATALOG hospital_lakehouse TO `operational_staff`",
]:
    try:
        spark.sql(grant)
        print("ok:", grant)
    except Exception as e:  # e.g. a group that doesn't exist, or a workspace-local group
        reason = ("Unity Catalog doesn't accept grants to workspace-local groups on Free Edition; "
                  "users get these grants individually (INSTALLATION.md, step 14a)"
                  if "PRINCIPAL_DOES_NOT_EXIST" in str(e) or "not exist" in str(e).lower() else e.__class__.__name__)
        print(f"skipped: {grant}\n   reason: {reason}")
