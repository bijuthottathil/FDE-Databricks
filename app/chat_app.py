# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks App — unified clinical chat UI
# MAGIC
# MAGIC The one surface clinicians and staff actually talk to. It never calls a Model Serving
# MAGIC endpoint directly — every turn goes through the router (`03_routing`), which is the only
# MAGIC component allowed to decide PHI-vs-general and to hold credentials for both endpoints. This
# MAGIC file is the Databricks Apps entry point (`app.yaml` + this Flask app).

# COMMAND ----------

# MAGIC %md
# MAGIC ```yaml
# MAGIC # app.yaml
# MAGIC command: ["python", "chat_app.py"]
# MAGIC env:
# MAGIC   - name: "PHI_MODEL_ENDPOINT"
# MAGIC     value: "hospital-private-llm"
# MAGIC   - name: "GENERAL_MODEL_ENDPOINT"
# MAGIC     value: "hospital-external-openai"
# MAGIC ```

# COMMAND ----------

import os
import sys

from flask import Flask, request, jsonify, render_template_string

# __file__ is unreliable in this execution context (Databricks Apps / file-editor "Run"),
# so resolve the project root via the notebook path the runtime gives us instead, with a
# cwd-based fallback for anywhere that doesn't apply.
try:
    _notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
    _project_root = "/Workspace" + "/".join(_notebook_path.split("/")[:-2])
except Exception:
    _project_root = os.path.dirname(os.getcwd())

sys.path.append(_project_root)

from importlib import import_module

router_mod = import_module("03_routing.01_query_classifier_router")
compliance_mod = import_module("07_compliance.01_mlflow_tracking")

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Hospital Chat Assistant</title>
<h2>Ask a clinical or operational question</h2>
<form method="post" action="/ask">
  <input name="query" style="width:400px" placeholder="e.g. Any beds available on 4 West?">
  <button type="submit">Ask</button>
</form>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query") or (request.json or {}).get("query", "")
    user_email = request.headers.get("X-Forwarded-Email", "unknown@hospital.org")

    phi_registry = router_mod.load_phi_table_registry(spark)  # noqa: F821 (spark injected by Databricks runtime)
    result = router_mod.handle_query(query, user_email, phi_registry)

    compliance_mod.log_routing_decision(
        decision=result["decision"],
        model_endpoint=result["model_endpoint"],
        matched_tables=[c["source_table"] for c in result["context"]],
        reason=result["reason"],
    )

    return jsonify(
        {
            "answer_source": result["decision"],
            "context_used": result["context"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
