# Databricks notebook source
# MAGIC %md
# MAGIC # Compliance log — append-only, hash-chained
# MAGIC
# MAGIC `hospital_lakehouse.audit.compliance_log` is the restricted record of every chat request: who asked,
# MAGIC how it was routed, which model and chunks answered, which guardrails fired. The app writes one row per
# MAGIC request. It is separate from `audit.chat_log`, which stays the safe operational log for monitoring.
# MAGIC
# MAGIC - **Append-only** (`delta.appendOnly`): `UPDATE` and `DELETE` are rejected; `DESCRIBE HISTORY` records any change.
# MAGIC - **Hash chain**: each row carries the hash of the previous row, so an edited or removed row is detected
# MAGIC   by `verify_chain` below. This is tamper-*evident*, not tamper-proof: the owner can still drop the table.
# MAGIC - **Question/answer text is OFF by default.** Set the app env var `COMPLIANCE_LOG_TEXT=true` to store
# MAGIC   it — the table then holds PHI and must be handled as such. Retention means temporarily unsetting
# MAGIC   `delta.appendOnly`; document who may do that.

# COMMAND ----------

import os
import sys

from databricks.sdk import WorkspaceClient

# guards.py lives with the app: ./app from the project root (terminal), ../app from this folder (workspace).
sys.path.insert(0, next(p for p in (os.path.join(os.getcwd(), "app"), os.path.join(os.getcwd(), "..", "app"))
                        if os.path.exists(os.path.join(p, "guards.py"))))
import guards  # noqa: E402

TABLE = "hospital_lakehouse.audit.compliance_log"

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
      seq BIGINT COMMENT 'position in the hash chain, starting at 1',
      event_id STRING,
      event_time STRING COMMENT 'UTC ISO-8601; a string because it is part of the hashed content',
      user_email STRING,
      status STRING COMMENT 'ok | failed | blocked | denied',
      path STRING,
      intent STRING,
      model STRING,
      model_version STRING,
      chunk_ids STRING,
      flags STRING COMMENT 'guardrail outcomes, comma-separated',
      question STRING COMMENT 'NULL unless COMPLIANCE_LOG_TEXT is enabled',
      answer STRING COMMENT 'NULL unless COMPLIANCE_LOG_TEXT is enabled',
      prev_hash STRING,
      row_hash STRING
    )
    TBLPROPERTIES (delta.appendOnly = true)
    COMMENT 'Append-only, hash-chained compliance record of chat requests. Restricted: contains user identities and, if enabled, PHI.'
""")
spark.sql(f"ALTER TABLE {TABLE} SET TAGS ('classification' = 'phi')")  # conservative: the text columns can hold PHI

# COMMAND ----------

# MAGIC %md ## Access
# MAGIC The app's service principal inserts (and reads the last row to extend the chain). Nobody else is granted
# MAGIC anything here; grant read to individual compliance reviewers by email (workspace groups can't be granted).

# COMMAND ----------

from databricks.sdk.errors import NotFound

_w = WorkspaceClient()
try:
    _app_sp = _w.apps.get("hospital-chat").service_principal_client_id
    spark.sql(f"GRANT SELECT, MODIFY ON TABLE {TABLE} TO `{_app_sp}`")
    print("granted SELECT, MODIFY to the app service principal")
except NotFound:
    # The app (and so its service principal) is created by `databricks bundle deploy`; its grants are
    # applied in INSTALLATION.md step 15, which covers this table too.
    print("hospital-chat isn't deployed yet: its access to this table is granted in INSTALLATION.md step 15")
# Example for a reviewer:  GRANT SELECT ON TABLE hospital_lakehouse.audit.compliance_log TO `reviewer@hospital.org`

# COMMAND ----------

# MAGIC %md ## Verify the chain

# COMMAND ----------


def verify() -> list[str]:
    rows = [r.asDict() for r in spark.table(TABLE).orderBy("seq").collect()]
    problems = guards.verify_chain(rows)
    print(f"{len(rows)} rows checked —", "chain intact" if not problems else f"{len(problems)} problem(s)")
    for p in problems:
        print("  ", p)
    return problems


verify()
