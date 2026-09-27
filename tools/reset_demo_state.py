# Databricks notebook source
# MAGIC %md
# MAGIC # Reset everything from `05_guardrails` onward (for re-recording)
# MAGIC
# MAGIC Undoes INSTALLATION.md steps 13–15 so they can be run again from a clean state:
# MAGIC
# MAGIC | Step | What is removed |
# MAGIC |---|---|
# MAGIC | 13b `05_guardrails` | AI Gateway config on both endpoints; telemetry on `hospital-private-llm` (it redeploys) |
# MAGIC | 13c–13d `06_monitoring`, `07_compliance/02` | The whole `hospital_lakehouse.audit` schema: chat log, compliance log, views, telemetry tables |
# MAGIC | 13e `07_compliance/01` | The runs in the MLflow experiment `/Shared/hospital_chat/routing_and_prompts` (the experiment stays) |
# MAGIC | 14 test-user access | Only with `RESET_TEST_USERS=yes`: the test users' catalog grants |
# MAGIC | 15 chat app | The `hospital-chat` app, and with it its service principal and every grant step 15b gave it |
# MAGIC
# MAGIC Kept: the catalog and its data, chunks, indexes, the model and both endpoints, groups, and the steps 1–12 setup.
# MAGIC
# MAGIC **Dry run by default**: it only prints what it would do. To actually reset, run with `RESET_CONFIRM=yes`.
# MAGIC **Deletes the audit history.** Only for demo or test workspaces.

# COMMAND ----------

import os
import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

CONFIRM = os.environ.get("RESET_CONFIRM", "").lower() == "yes"
RESET_TEST_USERS = os.environ.get("RESET_TEST_USERS", "").lower() == "yes"
TEST_USERS = ["bijumathewt@gmail.com", "bijuawsazure@gmail.com"]
ENDPOINTS = ["hospital-external-openai", "hospital-private-llm"]
AUDIT = "hospital_lakehouse.audit"
EXPERIMENT = "/Shared/hospital_chat/routing_and_prompts"
APP = "hospital-chat"

w = WorkspaceClient()
HOST, HEADERS = w.config.host.rstrip("/"), w.config.authenticate()


def do(description: str, action):
    """Run `action` only when confirmed; always say what it is."""
    if not CONFIRM:
        print(f"[dry run] would {description}")
        return
    action()
    print(f"done: {description}")


print("RESET_CONFIRM=yes: resetting now." if CONFIRM else "Dry run: nothing will change. Re-run with RESET_CONFIRM=yes to reset.")

# COMMAND ----------

# 15. The app first, so nothing writes to the audit tables while they're being removed. Deleting the app also
#     deletes its service principal, which removes its catalog grants, endpoint permissions and group membership.
def delete_app():
    w.apps.delete(APP)
    for _ in range(40):  # deletion runs in the background; wait until the app is gone
        try:
            w.apps.get(APP)
        except NotFound:
            return
        time.sleep(15)
    raise TimeoutError(f"{APP} is still being deleted after 10 minutes")


try:
    w.apps.get(APP)
    do(f"delete the app {APP} (and its service principal and grants)", delete_app)
except NotFound:
    print(f"{APP}: not installed")

# COMMAND ----------

# 13b. AI Gateway. An empty config is rejected, and a PUT replaces the whole config, so sending only
#      "usage tracking off" clears guardrails and rate limits too.
def clear_gateway(ep: str):
    r = requests.put(f"{HOST}/api/2.0/serving-endpoints/{ep}/ai-gateway", headers=HEADERS,
                     json={"usage_tracking_config": {"enabled": False}})
    r.raise_for_status()


for ep in ENDPOINTS:
    gw = w.api_client.do("GET", f"/api/2.0/serving-endpoints/{ep}").get("ai_gateway") or {}
    if gw.get("guardrails") or gw.get("rate_limits") or (gw.get("usage_tracking_config") or {}).get("enabled"):
        do(f"clear the AI Gateway config on {ep} (guardrails, rate limits, usage tracking)", lambda ep=ep: clear_gateway(ep))
    else:
        print(f"{ep}: AI Gateway already clear")

# COMMAND ----------

# 13b. Telemetry on the private endpoint. It has to go before the audit schema: an endpoint whose telemetry points
#      at a missing schema fails to deploy. Removing it redeploys the endpoint; wait for that to finish.
def remove_telemetry():
    ep = "hospital-private-llm"
    w.serving_endpoints.patch_telemetry_config(ep)  # no config = remove telemetry
    for _ in range(60):
        if w.serving_endpoints.get(ep).state.config_update.value == "NOT_UPDATING":
            break
        time.sleep(15)
    print(f"   hospital-private-llm: {w.serving_endpoints.get(ep).state.ready.value}")


if w.serving_endpoints.get("hospital-private-llm").telemetry_config:
    do("remove telemetry from hospital-private-llm (the endpoint redeploys for a few minutes)", remove_telemetry)
else:
    print("hospital-private-llm: telemetry already off")

# COMMAND ----------

# 13c–13d. The audit schema and everything in it.
def drop_audit():
    if w.serving_endpoints.get("hospital-private-llm").telemetry_config:
        raise RuntimeError("hospital-private-llm still has telemetry configured; not dropping the audit schema")
    spark.sql(f"DROP SCHEMA IF EXISTS {AUDIT} CASCADE")


try:
    tables = [t.name for t in w.tables.list(catalog_name="hospital_lakehouse", schema_name="audit")]
    do(f"drop the schema {AUDIT} and its {len(tables)} tables/views ({', '.join(tables) or 'none'})", drop_audit)
except NotFound:
    print(f"{AUDIT}: already gone")

# COMMAND ----------

# 13e. MLflow runs. Only the runs are deleted: MLflow won't let a new experiment reuse a deleted experiment's name,
#      so deleting the experiment itself would make 07_compliance/01 fail.
try:
    exp = w.experiments.get_by_name(EXPERIMENT).experiment
    runs = list(w.experiments.search_runs(experiment_ids=[exp.experiment_id]))
    if runs:
        do(f"delete {len(runs)} MLflow run(s) in {EXPERIMENT}",
           lambda: [w.experiments.delete_run(r.info.run_id) for r in runs])
    else:
        print(f"{EXPERIMENT}: no runs")
except NotFound:
    print(f"{EXPERIMENT}: experiment doesn't exist")

# COMMAND ----------

# 14. Test users' catalog grants (optional). Their entitlements, warehouse access and group membership stay:
#     step 14 re-applying them is harmless, and removing entitlements could lock the accounts out of the workspace.
if RESET_TEST_USERS:
    for u in TEST_USERS:
        do(f"revoke {u}'s catalog grants on hospital_lakehouse",
           lambda u=u: spark.sql(f"REVOKE USE CATALOG, USE SCHEMA, SELECT ON CATALOG hospital_lakehouse FROM `{u}`"))
else:
    print("test users: left as they are (set RESET_TEST_USERS=yes to revoke their catalog grants)")

# COMMAND ----------

print("\nReset complete. Start recording at INSTALLATION.md step 13." if CONFIRM
      else "\nDry run complete. To reset: RESET_CONFIRM=yes python3 <bootstrap> tools/reset_demo_state.py")
