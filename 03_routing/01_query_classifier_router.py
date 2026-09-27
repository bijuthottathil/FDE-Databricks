# Databricks notebook source
# MAGIC %md
# MAGIC # The core FDE build: classification-driven router
# MAGIC
# MAGIC For every incoming user query, decide — **before any retrieval or generation happens** —
# MAGIC whether answering it requires touching a Unity-Catalog-tagged `phi` table/column. That
# MAGIC decision, not the query's wording alone, determines the entire downstream path:
# MAGIC
# MAGIC - **PHI path**: retrieve from the PHI Vector Search index → generate with the private,
# MAGIC   self-hosted LLM (Model Serving on customer-managed compute). Nothing leaves the VPC.
# MAGIC - **General path**: retrieve from the general Vector Search index → generate with an
# MAGIC   External Model (Azure OpenAI with a BAA, or standard OpenAI for truly non-PHI use).
# MAGIC
# MAGIC The classifier is deliberately two-layered: a fast deterministic layer that looks up which
# MAGIC tables/columns the query's own retrieval step would need to touch (schema + tag lookup, not
# MAGIC an LLM's judgment call), backed by a lightweight semantic classifier for queries that don't
# MAGIC map cleanly to a known table (e.g. "does she have any allergies?" implies clinical tables
# MAGIC without naming them). Unity Catalog tags are authoritative; the semantic layer only decides
# MAGIC *which* tagged resource is implicated, never whether the tag itself should be trusted.

# COMMAND ----------

from dataclasses import dataclass
from enum import Enum


class RoutingDecision(str, Enum):
    PHI_PRIVATE_LLM = "phi_private_llm"
    GENERAL_EXTERNAL_LLM = "general_external_llm"


@dataclass
class RouteResult:
    decision: RoutingDecision
    matched_tables: list[str]
    vector_index: str
    model_endpoint: str
    reason: str

# COMMAND ----------

# MAGIC %md ## Layer 1 — deterministic: which tagged tables does this query implicate?
# MAGIC
# MAGIC Built once per session (or cached with a short TTL) from `system.information_schema`, so the
# MAGIC router's notion of "what's PHI" can never drift from Unity Catalog's.

# COMMAND ----------


class PhiRegistry(set):
    """Set of PHI/PII-tagged table names, plus `.known`: every table that exists in Unity
    Catalog. `route()` uses `.known` to fail closed on tables it can't find."""

    known: set[str] | None = None


def load_phi_table_registry(spark) -> PhiRegistry:
    rows = spark.sql(
        """
        SELECT DISTINCT catalog_name || '.' || schema_name || '.' || table_name AS full_name
        FROM system.information_schema.table_tags
        WHERE tag_name = 'classification' AND tag_value IN ('phi', 'pii')
        UNION
        SELECT DISTINCT catalog_name || '.' || schema_name || '.' || table_name AS full_name
        FROM system.information_schema.column_tags
        WHERE tag_name = 'classification' AND tag_value IN ('phi', 'pii')
        """
    ).collect()
    registry = PhiRegistry(r["full_name"] for r in rows)
    known_rows = spark.sql(
        """
        SELECT table_catalog || '.' || table_schema || '.' || table_name AS full_name
        FROM system.information_schema.tables
        WHERE table_catalog = 'hospital_lakehouse'
        """
    ).collect()
    registry.known = {r["full_name"] for r in known_rows}
    return registry

# COMMAND ----------

# MAGIC %md ## Layer 2 — semantic intent classifier
# MAGIC
# MAGIC A small, cheap model call (served privately — the classifier itself never sees confirmed PHI,
# MAGIC only the user's free-text question) maps the query to a small set of known intents, each of
# MAGIC which is pre-mapped to a table list. This keeps the *decision boundary* declarative and
# MAGIC auditable rather than trusting an LLM's live judgment of "is this PHI".

# COMMAND ----------

INTENT_TABLE_MAP = {
    "patient_clinical_lookup": ["hospital_lakehouse.clinical.silver_patient_encounters",
                                 "hospital_lakehouse.clinical.gold_patient_summary"],
    "lab_result_lookup": ["hospital_lakehouse.clinical.silver_lab_results"],
    "bed_availability": ["hospital_lakehouse.operational.gold_bed_availability_by_unit"],
    "cafeteria_hours": ["hospital_lakehouse.operational.facilities_info"],
    "general_policy": ["hospital_lakehouse.operational.policy_docs"],
}

CLASSIFIER_ENDPOINT = "hospital_intent_classifier"  # small model served privately; see 04_serving


def classify_intent(query: str) -> str:
    """Calls a lightweight, privately-served classifier endpoint. Never calls an external API —
    the query text itself may reference PHI even before we know the answer will."""
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint=CLASSIFIER_ENDPOINT,
        inputs={
            "dataframe_records": [{"query": query, "labels": list(INTENT_TABLE_MAP.keys())}]
        },
    )
    return response["predictions"][0]["label"]

# COMMAND ----------

# MAGIC %md ## Router

# COMMAND ----------

PHI_MODEL_ENDPOINT = "hospital-private-llm"          # 04_serving/01_private_llm_endpoint.py
GENERAL_MODEL_ENDPOINT = "hospital-external-openai"  # 04_serving/02_external_model_endpoint.py

PHI_VECTOR_INDEX = "hospital_lakehouse.clinical.phi_chunks_index"
GENERAL_VECTOR_INDEX = "hospital_lakehouse.operational.general_chunks_index"


def route(query: str, phi_registry: set[str]) -> RouteResult:
    intent = classify_intent(query)
    tables = INTENT_TABLE_MAP.get(intent, [])
    touches_phi = any(t in phi_registry for t in tables)

    # A mapped table that isn't in Unity Catalog has no tag we can trust — an absent table
    # is not the same as a general one. Fail closed.
    known = getattr(phi_registry, "known", None)
    missing = [t for t in tables if known is not None and t not in known]
    if missing:
        return RouteResult(
            decision=RoutingDecision.PHI_PRIVATE_LLM,
            matched_tables=tables,
            vector_index=PHI_VECTOR_INDEX,
            model_endpoint=PHI_MODEL_ENDPOINT,
            reason=f"intent='{intent}' maps to table(s) not found in Unity Catalog {missing} — defaulting to PHI-safe path",
        )

    if touches_phi or not tables:
        # Unknown intent defaults to the PHI-safe path — fail closed, not open.
        return RouteResult(
            decision=RoutingDecision.PHI_PRIVATE_LLM,
            matched_tables=tables,
            vector_index=PHI_VECTOR_INDEX,
            model_endpoint=PHI_MODEL_ENDPOINT,
            reason=f"intent='{intent}' matches PHI-tagged table(s) {tables}" if tables
                   else f"intent='{intent}' unrecognized — defaulting to PHI-safe path",
        )

    return RouteResult(
        decision=RoutingDecision.GENERAL_EXTERNAL_LLM,
        matched_tables=tables,
        vector_index=GENERAL_VECTOR_INDEX,
        model_endpoint=GENERAL_MODEL_ENDPOINT,
        reason=f"intent='{intent}' only matches general-tagged table(s) {tables}",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## End-to-end handler
# MAGIC This is what `app/chat_app.py` calls per turn. Every decision is logged with its reasoning —
# MAGIC that log is what feeds the AI Gateway audit trail (`05_guardrails`) and the compliance
# MAGIC review dashboard (`06_monitoring`).

# COMMAND ----------

import logging

logger = logging.getLogger("router.audit")


def handle_query(query: str, user_email: str, phi_registry: set[str]) -> dict:
    result = route(query, phi_registry)

    logger.info(
        "route_decision",
        extra={
            "user": user_email,
            "decision": result.decision.value,
            "matched_tables": result.matched_tables,
            "reason": result.reason,
        },
    )

    from importlib import import_module
    vector_search = import_module("02_vector_search.01_create_indexes")  # illustrative import
    context_chunks = vector_search.retrieve(query, result.vector_index)

    return {
        "decision": result.decision.value,
        "model_endpoint": result.model_endpoint,
        "context": context_chunks,
        "reason": result.reason,
    }
