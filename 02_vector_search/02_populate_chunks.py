# Databricks notebook source
# MAGIC %md
# MAGIC # Populate the chunk tables and sync the indexes
# MAGIC
# MAGIC Builds one text chunk per source row and refreshes both Delta Sync indexes.
# MAGIC - `operational.general_chunks` ← facilities, policies, bed availability (non-PHI)
# MAGIC - `clinical.phi_chunks` ← `gold_patient_summary` (PHI; never mixed into the general table).
# MAGIC   Carries the `unit` column so the ABAC unit-scoped row filter applies to the chunks too.

# COMMAND ----------

import subprocess
import sys
import time

try:
    import databricks.vector_search  # noqa: F401
except ImportError:
    # databricks-vectorsearch pulls in a pydantic/pydantic-core that needs
    # typing_extensions>=4.13 (for Sentinel) — upgrade it explicitly, since the runtime's
    # preinstalled version is older and a plain install won't touch an already-satisfied dep.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                            "typing_extensions>=4.13.0", "databricks-vectorsearch"])
    # The kernel already imported `databricks` (namespace package, e.g. via databricks.sdk) and
    # the runtime's older typing_extensions before this cell ran, so their cached module/path
    # state doesn't reflect what was just installed — evict both so the import below is fresh.
    import importlib
    for _mod in [m for m in sys.modules
                 if m in ("databricks", "typing_extensions")
                 or m.startswith("databricks.") or m.startswith("typing_extensions.")]:
        del sys.modules[_mod]
    importlib.invalidate_caches()

from databricks.vector_search.client import VectorSearchClient

ENDPOINT_NAME = "hospital_chat_vs_endpoint"
PHI_INDEX = "hospital_lakehouse.clinical.phi_chunks_index"
GENERAL_INDEX = "hospital_lakehouse.operational.general_chunks_index"

# COMMAND ----------

# -- General chunks. Full refresh: chunk_id is deterministic, so replace the table contents.
spark.sql("""
    INSERT OVERWRITE hospital_lakehouse.operational.general_chunks
    SELECT concat('facility-', facility_id) AS chunk_id,
           'facilities_info' AS source_table,
           concat(name, ' (', location, '): open ', opens, ' to ', closes, '. ', notes) AS content,
           current_timestamp() AS updated_at
    FROM hospital_lakehouse.operational.facilities_info
    UNION ALL
    SELECT concat('policy-', doc_id), 'policy_docs', concat(title, ': ', content), current_timestamp()
    FROM hospital_lakehouse.operational.policy_docs
    UNION ALL
    SELECT concat('beds-', unit), 'gold_bed_availability_by_unit',
           concat(unit, ' unit has ', beds_available, ' of ', beds_total, ' beds available as of ', as_of, '.'),
           current_timestamp()
    FROM hospital_lakehouse.operational.gold_bed_availability_by_unit
""")

# -- PHI chunks: one per encounter, with that visit's labs listed in time order. (Gold has one row per
# -- encounter x lab, so without the GROUP BY a single visit would become up to 15 near-duplicate
# -- chunks.) Test codes are spelled out so questions like "sodium" or "white blood cells" match.
# -- MRN and unit are kept as columns for the unit-scoped row filter and for auditing.
spark.sql("""
    INSERT OVERWRITE hospital_lakehouse.clinical.phi_chunks
    SELECT concat('enc-', encounter_id) AS chunk_id,
           'gold_patient_summary' AS source_table,
           mrn,
           unit,
           concat('Patient ', patient_name, ' (MRN ', mrn, '), ', encounter_class, ' encounter ', encounter_id,
                  ' on ', period_start, ' in ', unit, ', diagnosis code ', diagnosis_code,
                  CASE WHEN count(test_code) = 0 THEN '' ELSE concat('. Labs: ', array_join(transform(
                      array_sort(collect_list(struct(observed_at, test_code, result_value, result_flag))),
                      l -> concat(CASE l.test_code WHEN 'GLU' THEN 'glucose (GLU)'
                                                   WHEN 'CR'  THEN 'creatinine (CR)'
                                                   WHEN 'HGB' THEN 'hemoglobin (HGB)'
                                                   WHEN 'NA'  THEN 'sodium (NA)'
                                                   WHEN 'WBC' THEN 'white blood cells (WBC)'
                                                   ELSE l.test_code END,
                                  ' = ', l.result_value, ' (flag ', l.result_flag, ') at ', l.observed_at)), '; ')) END,
                  '.') AS content,
           current_timestamp() AS updated_at
    FROM hospital_lakehouse.clinical.gold_patient_summary
    GROUP BY encounter_id, mrn, unit, patient_name, encounter_class, period_start, diagnosis_code
""")

for t in ["clinical.phi_chunks", "operational.general_chunks"]:
    print(t, spark.table(f"hospital_lakehouse.{t}").count(), "chunks")

# COMMAND ----------

# -- Triggered sync of the indexes that exist. The PHI index reads
# -- the unfiltered copy phi_chunks_vs_source, which isn't refreshed here: re-run its INSERT OVERWRITE
# -- (INSTALLATION.md, step 10) first if you want the PHI index to pick up these chunks.
vsc = VectorSearchClient(disable_notice=True)
existing = {i["name"] for i in vsc.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])}
indexes = [n for n in [GENERAL_INDEX, PHI_INDEX] if n in existing]
for name in [GENERAL_INDEX, PHI_INDEX]:
    if name not in existing:
        print(f"{name}: not created yet, skipping sync (run 01_create_indexes.py)")

# Start the syncs without waiting: on Free Edition this session holds the serverless compute the
# syncs need. Check progress as shown in INSTALLATION.md.
synced = []
for name in indexes:
    index = vsc.get_index(ENDPOINT_NAME, name)
    if not index.describe().get("status", {}).get("ready"):
        print(f"{name}: still building. Once it's ready, run this notebook again to sync these chunks")
        continue
    index.sync()
    synced.append(name)

print(f"Chunks loaded. Index sync started for {len(synced)} index(es): {', '.join(synced) or 'none'}.")
