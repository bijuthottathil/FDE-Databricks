# Installation Steps (quick reference)

The commands only, in order. For what each step does, expected output and troubleshooting, see `INSTALLATION.md`
(same step numbers).

**Rules:** run from the project root with the virtual environment active, one step at a time, and wait for each
to finish (Free Edition allows one serverless job at a time).

---

## Setup (every new terminal)

- Go to the project and activate the environment:
```zsh
cd /Volumes/D/Projects/FDE/medical_fde
source .venv/bin/activate
BOOT=/Users/bijum/.vscode/extensions/databricks.databricks-2.19.0-darwin-arm64/resources/python/dbconnect-bootstrap.py
```
- Make sure you're in the `phi_service_principals` group (Settings → Identity and access → Groups) before step 6.

## Data and governance

- **1. Governance, first pass** (catalog, schemas, groups, tags):
```zsh
python3 $BOOT 00_governance/01_unity_catalog_tags_and_abac.py
```
- **2. Landing schema and volumes:**
```zsh
python3 -c "
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.serverless().getOrCreate()
spark.sql(\"CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.landing COMMENT 'Raw file drops from source systems. Read only by the ingestion pipeline.'\")
for v in ['ehr_fhir', 'lab_results', 'scheduling']:
    spark.sql(f'CREATE VOLUME IF NOT EXISTS hospital_lakehouse.landing.{v}')
print('Landing schema and volumes ready: hospital_lakehouse.landing.{ehr_fhir, lab_results, scheduling}')
"
```
- **3. Synthetic landing data:**
```zsh
python3 01_ingestion/04_synthetic_landing_data.py
```
- **4. Ingestion pipeline, full refresh** (`hospital_ingestion`, not `FDE-Medical-ETL-Pipeline`):
```zsh
databricks bundle run hospital_ingestion --full-refresh-all
```
- **5. Reference tables**, then assign the test clinician to ICU:
```zsh
python3 $BOOT 01_ingestion/03_operational_reference_tables.py
```
```zsh
python3 -c "from databricks.connect import DatabricksSession as S; S.builder.serverless().getOrCreate().sql(\"INSERT INTO hospital_lakehouse.operational.staff_assignments VALUES ('bijumathewt@gmail.com', 'ICU')\"); print('Clinician assignment added: bijumathewt@gmail.com -> ICU')"
```
- **6. PHI/PII autotagger:**
```zsh
python3 $BOOT 01_ingestion/02_pii_phi_autotagger.py
```
- **7. Governance, second pass** (row filters and masks):
```zsh
python3 $BOOT 00_governance/01_unity_catalog_tags_and_abac.py
```

## Retrieval

- **8. Chunk tables and general index:**
```zsh
python3 $BOOT 02_vector_search/01_create_indexes.py
```
- **9. Fill the chunk tables:**
```zsh
python3 $BOOT 02_vector_search/02_populate_chunks.py
```
- **10. Unfiltered copy for the PHI index** (demo data only):
```zsh
python3 -c "
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.serverless().getOrCreate()
spark.sql(\"CREATE TABLE IF NOT EXISTS hospital_lakehouse.clinical.phi_chunks_vs_source (chunk_id STRING, source_table STRING, mrn STRING, unit STRING, content STRING, updated_at TIMESTAMP) COMMENT 'Unfiltered copy of phi_chunks for the PHI vector index. NOT row-filtered.' TBLPROPERTIES (delta.enableChangeDataFeed = true)\")
spark.sql(\"ALTER TABLE hospital_lakehouse.clinical.phi_chunks_vs_source SET TAGS ('classification' = 'phi')\")
spark.sql('INSERT OVERWRITE hospital_lakehouse.clinical.phi_chunks_vs_source SELECT chunk_id, source_table, mrn, unit, content, updated_at FROM hospital_lakehouse.clinical.phi_chunks')
print('PHI index source ready: phi_chunks_vs_source has', spark.table('hospital_lakehouse.clinical.phi_chunks_vs_source').count(), 'rows')
"
```
- **11. PHI index** (re-run step 8):
```zsh
python3 $BOOT 02_vector_search/01_create_indexes.py
```

## Private model

- **12a. `models` schema and `weights` volume:**
```zsh
python3 -c "
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.serverless().getOrCreate()
spark.sql(\"CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.models COMMENT 'Private LLM weights and registered models.'\")
spark.sql('CREATE VOLUME IF NOT EXISTS hospital_lakehouse.models.weights')
print('Models schema and weights volume ready: hospital_lakehouse.models.weights')
"
```
- **12b. Download and upload the Qwen weights** (about 3.1 GB):
```zsh
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir /tmp/qwen2.5-1.5b-instruct \
  --include "*.json" --include "*.safetensors" --include "merges.txt"
rm -rf /tmp/qwen2.5-1.5b-instruct/.cache   # download metadata; not part of the model
databricks fs cp -r --overwrite /tmp/qwen2.5-1.5b-instruct \
  dbfs:/Volumes/hospital_lakehouse/models/weights/qwen2.5-1.5b-instruct
databricks fs ls dbfs:/Volumes/hospital_lakehouse/models/weights/qwen2.5-1.5b-instruct \
  && echo "Weights uploaded to /Volumes/hospital_lakehouse/models/weights/qwen2.5-1.5b-instruct"
```
- **12c. `audit` schema:**
```zsh
python3 -c "from databricks.connect import DatabricksSession as S; S.builder.serverless().getOrCreate().sql(\"CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.audit COMMENT 'Audit and monitoring tables.'\"); print('Audit schema ready: hospital_lakehouse.audit')"
```
- **12d. Register the model and deploy the endpoint** (in the Databricks UI):
  - Workspace → Users → btucker5543@gmail.com → Databricks → FDE-Medical → files → 04_serving → `01_private_llm_endpoint`
  - Compute: **Serverless** → **Run all**
  - Wait until the endpoint is `READY`:
```zsh
databricks serving-endpoints get hospital-private-llm -o json | grep -A2 '"state"'
```

## Guardrails, monitoring and compliance

- **13b. AI Gateway guardrails and telemetry:**
```zsh
python3 $BOOT 05_guardrails/01_ai_gateway_guardrails.py
```
- **13c. Monitoring:**
```zsh
python3 $BOOT 06_monitoring/01_lakehouse_monitoring.py
```
- **13d. Compliance log:**
```zsh
python3 $BOOT 07_compliance/02_compliance_log.py
```
- **13e. MLflow lineage:**
```zsh
python3 $BOOT 07_compliance/01_mlflow_tracking.py
```

## Test users

- **14a. Catalog grants:**
```zsh
python3 -c "
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.serverless().getOrCreate()
for u in ['bijumathewt@gmail.com', 'bijuawsazure@gmail.com']:
    spark.sql(f'GRANT USE CATALOG, USE SCHEMA, SELECT ON CATALOG hospital_lakehouse TO \`{u}\`')
    print('Catalog access granted:', u)
"
```
- **14b. Workspace entitlements:**
```zsh
python3 -c "
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import iam
w = WorkspaceClient()
for u in ['bijumathewt@gmail.com', 'bijuawsazure@gmail.com']:
    user = next(iter(w.users.list(filter=f\"userName eq '{u}'\")))
    w.users.patch(user.id, schemas=[iam.PatchSchema.URN_IETF_PARAMS_SCIM_API_MESSAGES_2_0_PATCH_OP],
        operations=[iam.Patch(op=iam.PatchOp.ADD, path='entitlements',
                              value=[{'value': 'workspace-access'}, {'value': 'databricks-sql-access'}])])
    print('Entitlements added:', u)
"
```
- **14c. SQL warehouse access:**
```zsh
databricks warehouses update-permissions 73c6a216d830bde0 --json '{"access_control_list":[
  {"user_name":"bijumathewt@gmail.com","permission_level":"CAN_USE"},
  {"user_name":"bijuawsazure@gmail.com","permission_level":"CAN_USE"}]}' >/dev/null \
  && echo "Warehouse access granted: Serverless Starter Warehouse"
```

## Chat app

- **15a. Deploy** (creates the app and its service principal):
```zsh
databricks bundle deploy && echo "App deployed: hospital-chat"
```
- **15b. Grant the app's service principal** catalog access, `phi_service_principals` membership and `CAN_QUERY` on
  both endpoints:
```zsh
python3 -c "
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import iam
from databricks.sdk.service.serving import ServingEndpointAccessControlRequest as R, ServingEndpointPermissionLevel as L
import sys
from databricks.sdk.errors import NotFound
w = WorkspaceClient()
try:
    app = w.apps.get('hospital-chat')
except NotFound:
    sys.exit('hospital-chat does not exist yet: run step 15a (databricks bundle deploy) first, then this.')
sp, sp_id = app.service_principal_client_id, str(app.service_principal_id)
spark = DatabricksSession.builder.serverless().getOrCreate()
for g in [f'GRANT USE CATALOG ON CATALOG hospital_lakehouse TO \`{sp}\`',
          f'GRANT USE SCHEMA, SELECT ON SCHEMA hospital_lakehouse.clinical TO \`{sp}\`',
          f'GRANT USE SCHEMA, SELECT ON SCHEMA hospital_lakehouse.operational TO \`{sp}\`',
          f'GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA hospital_lakehouse.audit TO \`{sp}\`']:
    spark.sql(g)
print('Catalog access granted to', app.service_principal_name)
grp = next(iter(w.groups.list(filter=\"displayName eq 'phi_service_principals'\")))
w.groups.patch(grp.id, schemas=[iam.PatchSchema.URN_IETF_PARAMS_SCIM_API_MESSAGES_2_0_PATCH_OP],
               operations=[iam.Patch(op=iam.PatchOp.ADD, value={'members': [{'value': sp_id}]})])
print('Added to phi_service_principals:', app.service_principal_name)
for ep in ['hospital-private-llm', 'hospital-external-openai']:
    w.serving_endpoints.update_permissions(w.serving_endpoints.get(ep).id,
        access_control_list=[R(service_principal_name=sp, permission_level=L.CAN_QUERY)])
    print('CAN_QUERY granted on', ep)
print('App service principal ready.')
"
```
- Wait about **4 minutes** for the group membership to apply.
- **15c. Start the app:**
```zsh
databricks bundle run hospital_chat && echo "App running."
```
- Open the app URL in a **private window** and approve the `sql` scope:
```zsh
databricks apps get hospital-chat -o json | grep '"url"'
```

## Verify

- **16.** Ask the questions in `INSTALLATION.md` step 16, for example:
  - *Are any beds available in Oncology?* → general, 4 of 35
  - *What is Nadia Karim's diagnosis?* → private model, O80
  - *Which patients are in the ICU?* → records listed: Daniel Okafor, Tomas Novak
  - *How many patients are admitted?* → 14
- Check the audit log in the SQL editor:
```sql
SELECT event_time, user_email, status, path, model FROM hospital_lakehouse.audit.chat_log ORDER BY event_time DESC LIMIT 20;
```
