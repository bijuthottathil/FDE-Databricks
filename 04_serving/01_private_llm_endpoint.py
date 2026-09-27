# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Private LLM — small open-weight model, served inside the workspace
# MAGIC
# MAGIC Registers Qwen2.5-1.5B-Instruct (weights kept in a Unity Catalog volume) as an MLflow model in
# MAGIC Unity Catalog and serves it on Databricks Model Serving. Prompts and retrieved PHI context are only
# MAGIC ever sent to this endpoint, which runs in this workspace — not to a third-party or Databricks-hosted
# MAGIC foundation model.
# MAGIC
# MAGIC **Free Edition build:** CPU compute (Small workload), scale-to-zero on. The original design used a dedicated GPU pool
# MAGIC on customer-managed compute with `scale_to_zero_enabled=False` and a fine-tuned model; neither is
# MAGIC available here, and this model is not fine-tuned. Expect cold starts and slow generation.
# MAGIC
# MAGIC The weights must already be in `/Volumes/hospital_lakehouse/models/weights/qwen2.5-1.5b-instruct`.

# COMMAND ----------

import os
import tempfile

import mlflow
import mlflow.pyfunc
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

CATALOG_MODEL = "hospital_lakehouse.models.clinical_llm"
PRIVATE_LLM_ENDPOINT = "hospital-private-llm"
WEIGHTS_VOLUME_DIR = "/Volumes/hospital_lakehouse/models/weights/qwen2.5-1.5b-instruct"

w = WorkspaceClient()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{w.current_user.me().user_name}/private_llm")

# COMMAND ----------

# MAGIC %md ## Model wrapper
# MAGIC Input: `question` and `records` (newline-joined retrieved chunks). Output: the answer string.
# MAGIC The system prompt keeps answers grounded in the supplied records only.
# MAGIC

# COMMAND ----------

class PrivateLLM(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = context.artifacts["weights"]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).eval()

    def predict(self, context, model_input, params=None):
        import torch

        out = []
        for question, records in zip(model_input["question"], model_input["records"]):
            messages = [
                {"role": "system", "content": "You are a clinical assistant. Answer ONLY from the records provided. "
                                               "If the records do not contain the answer, say so. Be brief."},
                {"role": "user", "content": f"Records:\n{records or '(no matching records)'}\n\nQuestion: {question}"},
            ]
            prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ids = self.tok(prompt, return_tensors="pt")
            with torch.no_grad():
                gen = self.model.generate(**ids, max_new_tokens=160, do_sample=False)
            out.append(self.tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return out

# COMMAND ----------

# MAGIC %md ## Register in Unity Catalog

# COMMAND ----------

local = WEIGHTS_VOLUME_DIR

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "question"), ColSpec("string", "records")]),
    outputs=Schema([ColSpec("string")]),
)
with mlflow.start_run():
    info = mlflow.pyfunc.log_model(
        "model",
        python_model=PrivateLLM(),
        artifacts={"weights": local},
        signature=signature,
        input_example=pd.DataFrame({"question": ["Which unit?"], "records": ["Patient A is in ICU."]}),
        pip_requirements=["torch==2.14.0", "transformers==5.17.0", "accelerate", "pandas"],
        registered_model_name=CATALOG_MODEL,
    )
VERSION = str(info.registered_model_version)
print("registered", CATALOG_MODEL, "version", VERSION)

# COMMAND ----------

# MAGIC %md ## Serving endpoint

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

config = EndpointCoreConfigInput(
    name=PRIVATE_LLM_ENDPOINT,
    served_entities=[
        ServedEntityInput(
            name="hospital-clinical-llm",
            entity_name=CATALOG_MODEL,
            entity_version=VERSION,
            workload_size="Small",        # Medium exceeds the free-usage provisioned-concurrency quota; Small fits this model
            scale_to_zero_enabled=True,   # Free Edition: no always-on capacity
        )
    ],
)
from databricks.sdk.errors import NotFound

# Decide create vs update up front, so a failed update surfaces its own error instead of being
# retried as a create (which only reports "already exists").
try:
    w.serving_endpoints.get(name=PRIVATE_LLM_ENDPOINT)
    exists = True
except NotFound:
    exists = False
if exists:
    w.serving_endpoints.update_config_and_wait(name=PRIVATE_LLM_ENDPOINT, served_entities=config.served_entities)
else:
    w.serving_endpoints.create_and_wait(name=PRIVATE_LLM_ENDPOINT, config=config)
print("endpoint ready:", PRIVATE_LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access policy
# MAGIC Only the chat app's service principal should be able to query this endpoint directly. Grant it via the
# MAGIC app resource (`resources/hospital_chat_app.yml`, `serving_endpoint` with `CAN_QUERY`) — endpoint
# MAGIC permissions are managed through the permissions API, not SQL `GRANT`.