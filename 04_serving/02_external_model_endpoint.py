# Databricks notebook source
# MAGIC %md
# MAGIC # External Model — OpenAI, for non-PHI queries only
# MAGIC
# MAGIC Used exclusively for the router's `general_external_llm` path — bed availability, cafeteria
# MAGIC hours, general policy. The API key lives in a Databricks secret
# MAGIC (`hospital_chat/openai_api_key`) and the endpoint config only references it, so the key never
# MAGIC appears in code or config. Store it with:
# MAGIC
# MAGIC     databricks secrets put-secret hospital_chat openai_api_key
# MAGIC
# MAGIC **Compliance note.** Standard OpenAI has no BAA. The original design defaulted to Azure OpenAI with a
# MAGIC BAA (template below) precisely so that a wrong PHI classification wouldn't become a HIPAA incident. This
# MAGIC endpoint is only acceptable for workspaces where compliance has signed off that the general path carries
# MAGIC no PHI — here that rests on the router, the pre-flight leak check in `app/main.py`, and the demo data
# MAGIC being fake. For real PHI, use the Azure template.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ExternalModel,
    ExternalModelProvider,
    OpenAiConfig,
    ServedEntityInput,
)

w = WorkspaceClient()
EXTERNAL_MODEL_ENDPOINT = "hospital-external-openai"

# COMMAND ----------

config = EndpointCoreConfigInput(
    name=EXTERNAL_MODEL_ENDPOINT,
    served_entities=[
        ServedEntityInput(
            name="openai-gpt4o-mini-general",
            external_model=ExternalModel(
                name="gpt-4o-mini",
                provider=ExternalModelProvider.OPENAI,
                task="llm/v1/chat",
                openai_config=OpenAiConfig(openai_api_key="{{secrets/hospital_chat/openai_api_key}}"),
            ),
        )
    ],
)
try:
    w.serving_endpoints.create_and_wait(name=EXTERNAL_MODEL_ENDPOINT, config=config)
except Exception as e:
    if "already exists" not in str(e):
        raise
    w.serving_endpoints.update_config_and_wait(name=EXTERNAL_MODEL_ENDPOINT, served_entities=config.served_entities)
print("endpoint ready:", EXTERNAL_MODEL_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure OpenAI with a BAA — template, not deployed
# MAGIC Needs a real Azure OpenAI resource and a secret `hospital_chat/azure_openai_api_key`. Azure settings live
# MAGIC in `openai_config` with `openai_api_type="azure"`:
# MAGIC
# MAGIC ```python
# MAGIC OpenAiConfig(
# MAGIC     openai_api_type="azure",
# MAGIC     openai_api_base="https://<resource>.openai.azure.com/",
# MAGIC     openai_api_version="2024-08-01-preview",
# MAGIC     openai_deployment_name="<deployment>",
# MAGIC     openai_api_key="{{secrets/hospital_chat/azure_openai_api_key}}",
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access
# MAGIC Give the chat app's service principal `CAN_QUERY` on this endpoint through the app resource
# MAGIC (`resources/hospital_chat_app.yml`); endpoint permissions are managed through the permissions API,
# MAGIC not SQL `GRANT`.
