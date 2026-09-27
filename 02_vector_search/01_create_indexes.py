# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search — PHI and general indexes, kept separate
# MAGIC
# MAGIC Retrieval must respect the PHI boundary *before* generation happens — a single mixed index
# MAGIC would let a "general" query's retrieval step pull PHI context into a prompt that's about to
# MAGIC go to a third-party model. So there are two Vector Search indexes, backed by two Delta
# MAGIC tables that Unity Catalog tags as `phi` and `general` respectively, and the router
# MAGIC (`03_routing`) picks the index the same way it picks the model.

# COMMAND ----------

import subprocess
import sys

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
from databricks.vector_search.exceptions import ResourceConflict

vsc = VectorSearchClient()

ENDPOINT_NAME = "hospital_chat_vs_endpoint"
PHI_INDEX = "hospital_lakehouse.clinical.phi_chunks_index"
GENERAL_INDEX = "hospital_lakehouse.operational.general_chunks_index"
# Vector Search can't index a row-filtered table, so the PHI index reads this unfiltered copy of
# phi_chunks. It is created by hand (INSTALLATION.md, step 10) because it bypasses row filtering.
PHI_INDEX_SOURCE = "hospital_lakehouse.clinical.phi_chunks_vs_source"

# COMMAND ----------

# MAGIC %md ## Endpoint (shared compute, index-level isolation is enforced by UC, not by endpoint)

# COMMAND ----------

try:
    vsc.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type="STANDARD",
    )
except ResourceConflict:
    print(f"Endpoint {ENDPOINT_NAME} already exists")

# COMMAND ----------

# -- Chunked, embeddable source tables. Built from Gold, inheriting Gold's UC tags.
# -- %sql magic is a no-op when this file is run via "Run file", so use spark.sql directly.
spark.sql("""
    CREATE TABLE IF NOT EXISTS hospital_lakehouse.clinical.phi_chunks (
      chunk_id STRING,
      source_table STRING,
      mrn STRING,
      unit STRING,
      content STRING,
      updated_at TIMESTAMP
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
# Tables created before `unit` was added to the schema above.
if "unit" not in spark.table("hospital_lakehouse.clinical.phi_chunks").columns:
    spark.sql("ALTER TABLE hospital_lakehouse.clinical.phi_chunks ADD COLUMN unit STRING AFTER mrn")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.phi_chunks SET TAGS ('classification' = 'phi', 'row_scope' = 'unit')")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.phi_chunks ALTER COLUMN unit SET TAGS ('unit_key' = 'true')")

spark.sql("""
    CREATE TABLE IF NOT EXISTS hospital_lakehouse.operational.general_chunks (
      chunk_id STRING,
      source_table STRING,
      content STRING,
      updated_at TIMESTAMP
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
spark.sql("ALTER TABLE hospital_lakehouse.operational.general_chunks SET TAGS ('classification' = 'general')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Indexes
# MAGIC Delta Sync indexes stay current with their source tables via change data feed, embedded by
# MAGIC `databricks-bge-large-en` inside the workspace. General first: it has no restrictions. The PHI
# MAGIC index can't use `phi_chunks` directly (Vector Search rejects row-filtered tables), so it's
# MAGIC built from `phi_chunks_vs_source` when that copy exists, and skipped otherwise.

# COMMAND ----------


def create_index(index_name, source_table):
    existing = {i["name"] for i in vsc.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])}
    if index_name in existing:
        print(f"{index_name}: already exists")
        return
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=ENDPOINT_NAME,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="content",
        embedding_model_endpoint_name="databricks-bge-large-en",  # served within the workspace
    )
    print(f"{index_name}: created from {source_table}")


create_index(GENERAL_INDEX, "hospital_lakehouse.operational.general_chunks")

if spark.catalog.tableExists(PHI_INDEX_SOURCE):
    create_index(PHI_INDEX, PHI_INDEX_SOURCE)
else:
    print(f"{PHI_INDEX}: skipped. Vector Search can't index the row-filtered phi_chunks table, and the "
          f"unfiltered copy {PHI_INDEX_SOURCE} doesn't exist yet. Create it (INSTALLATION.md, step 10), "
          "then run this notebook again. The app doesn't need this index: it reads phi_chunks directly "
          "as the signed-in user.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval helper used by the router
# MAGIC The router never chooses an index based on free-text guesswork — it passes the index name
# MAGIC through, driven by the same UC-tag lookup used for model routing.

# COMMAND ----------


def retrieve(query: str, index_name: str, num_results: int = 5):
    return vsc.get_index(ENDPOINT_NAME, index_name).similarity_search(
        query_text=query,
        columns=["chunk_id", "source_table", "content"],
        num_results=num_results,
    )
