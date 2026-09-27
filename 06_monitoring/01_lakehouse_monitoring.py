# Databricks notebook source
# MAGIC %md
# MAGIC # Monitoring — routing mix, grounding, latency, access-log review
# MAGIC
# MAGIC The chat app (`app/main.py`) writes one row per question to
# MAGIC `hospital_lakehouse.audit.chat_log`. It records the routing decision, the model used, which
# MAGIC chunks grounded the answer, and timings — **never the question or answer text**, which on the
# MAGIC PHI path may contain PHI. Everything below is built on that table.
# MAGIC
# MAGIC 1. **Routing mix** — share of PHI vs general questions over time, so a shift is visible on its own.
# MAGIC 2. **Grounding** — share of questions where no record matched. With no answer text logged this is
# MAGIC    the available proxy for "the model had nothing to be grounded in"; the app refuses to
# MAGIC    generate in that case, so a rising rate means retrieval is missing content.
# MAGIC 3. **Latency** — per path and model.
# MAGIC 4. **Access log review** — who queried what, cross-referenced against the row/column
# MAGIC    filters from `00_governance`, to catch policy misconfigurations rather than just
# MAGIC    successful attacks.

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.audit COMMENT 'Audit and monitoring tables.'")

# COMMAND ----------

# MAGIC %md ## Audit table (written by the app)

# COMMAND ----------

spark.sql("""
    CREATE TABLE IF NOT EXISTS hospital_lakehouse.audit.chat_log (
      event_time TIMESTAMP,
      user_email STRING,
      status STRING COMMENT 'ok | failed | blocked (guardrail) | denied (no access)',
      path STRING COMMENT 'phi | general',
      intent STRING,
      model STRING,
      matched_chunks INT,
      chunk_ids STRING,
      route_s DOUBLE,
      retrieve_s DOUBLE,
      generate_s DOUBLE,
      error STRING
    ) COMMENT 'One row per chat question. No question or answer text — it may contain PHI.'
""")
# user_email identifies a person, so the table itself is PII (never PHI: no clinical content).
spark.sql("ALTER TABLE hospital_lakehouse.audit.chat_log SET TAGS ('classification' = 'pii')")

# COMMAND ----------

# MAGIC %md ## Monitoring views

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE VIEW hospital_lakehouse.audit.routing_mix_daily AS
    SELECT date(event_time) AS day, path,
           count(*) AS questions,
           round(100 * count(*) / sum(count(*)) OVER (PARTITION BY date(event_time)), 1) AS pct_of_day
    FROM hospital_lakehouse.audit.chat_log
    WHERE status = 'ok'
    GROUP BY date(event_time), path
""")

spark.sql("""
    CREATE OR REPLACE VIEW hospital_lakehouse.audit.grounding_daily AS
    SELECT date(event_time) AS day, coalesce(path, 'n/a (stopped before routing)') AS path,
           count(*) AS questions,
           sum(CASE WHEN matched_chunks = 0 THEN 1 ELSE 0 END) AS no_records_matched,
           round(100 * avg(CASE WHEN matched_chunks = 0 THEN 1.0 ELSE 0.0 END), 1) AS pct_no_records,
           sum(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
           sum(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) AS blocked_by_guardrail,
           sum(CASE WHEN status = 'denied' THEN 1 ELSE 0 END) AS access_denied
    FROM hospital_lakehouse.audit.chat_log
    GROUP BY date(event_time), coalesce(path, 'n/a (stopped before routing)')
""")

spark.sql("""
    CREATE OR REPLACE VIEW hospital_lakehouse.audit.latency_hourly AS
    SELECT date_trunc('hour', event_time) AS hour, path, model,
           count(*) AS questions,
           round(avg(route_s + retrieve_s + generate_s), 1) AS avg_total_s,
           round(percentile_approx(route_s + retrieve_s + generate_s, 0.95), 1) AS p95_total_s,
           round(avg(generate_s), 1) AS avg_generate_s
    FROM hospital_lakehouse.audit.chat_log
    WHERE status = 'ok'
    GROUP BY date_trunc('hour', event_time), path, model
""")

# COMMAND ----------

# -- Quick look. Prints instead of display() so it also works via Databricks Connect.
for view in ["routing_mix_daily", "grounding_daily", "latency_hourly"]:
    print(f"\n== {view}")
    spark.table(f"hospital_lakehouse.audit.{view}").show(20, truncate=False)

# COMMAND ----------

# MAGIC %md ## Access log review — cross-reference against row/column policy

# COMMAND ----------

# -- Surfaces any PHI-tagged read NOT covered by an expected ABAC grant — i.e. a policy
# -- misconfiguration, since actual unauthorized reads should already be blocked at the source.
# %sql magic is a no-op when this file is run via "Run file", so use spark.sql directly.
access_review_df = spark.sql("""
    SELECT
      audit.event_time,
      audit.user_identity.email AS user_email,
      audit.request_params.full_name_arg AS table_name,
      audit.action_name
    FROM system.access.audit audit
    WHERE audit.service_name = 'unityCatalog'
      AND audit.action_name IN ('getTable', 'generateTemporaryTableCredential')
      AND audit.request_params.full_name_arg LIKE 'hospital_lakehouse.clinical.%'
      AND audit.user_identity.email NOT IN (
        SELECT staff_email FROM hospital_lakehouse.operational.staff_assignments
      )
    ORDER BY audit.event_time DESC
""")
try:
    display(access_review_df)
except NameError:  # running via Databricks Connect, not a notebook
    access_review_df.show(20, truncate=False)
