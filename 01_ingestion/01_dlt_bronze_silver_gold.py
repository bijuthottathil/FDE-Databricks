# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Delta Live Tables — EHR / Lab / Scheduling ingestion
# MAGIC
# MAGIC Bronze → Silver → Gold for three source systems:
# MAGIC - EHR extracts in HL7/FHIR
# MAGIC - Lab systems (HL7 ORU or flat feeds)
# MAGIC - Scheduling / bed-management systems (operational, non-PHI)
# MAGIC
# MAGIC PII/PHI detection and Unity Catalog auto-tagging happens at the Bronze → Silver hop,
# MAGIC in `02_pii_phi_autotagger.py`, which this pipeline calls as a Silver-layer expectation.

# COMMAND ----------

# Pipeline default schema is `clinical`; operational (non-PHI) tables use schema-qualified names.
import dlt
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Bronze — raw landing, schema-on-read, no filtering

# COMMAND ----------

@dlt.table(
    name="bronze_ehr_fhir",
    comment="Raw FHIR bundles landed from the EHR extract feed.",
    table_properties={"quality": "bronze"},
)
def bronze_ehr_fhir():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")  # one row per file, raw JSON in `value`
        .load("/Volumes/hospital_lakehouse/landing/ehr_fhir/")
    )


@dlt.table(
    name="bronze_lab_results",
    comment="Raw HL7 ORU lab result messages, pre-parsed to JSON by the interface engine.",
    table_properties={"quality": "bronze"},
)
def bronze_lab_results():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")  # one row per file, raw JSON in `value`
        .load("/Volumes/hospital_lakehouse/landing/lab_results/")
    )


@dlt.table(
    name="operational.bronze_scheduling",
    comment="Raw bed/scheduling feed. Operational, not clinical.",
    table_properties={"quality": "bronze"},
)
def bronze_scheduling():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")  # one row per file, raw JSON in `value`
        .load("/Volumes/hospital_lakehouse/landing/scheduling/")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver — parsed, deduped, PII/PHI-flagged
# MAGIC
# MAGIC `flag_pii_phi_columns()` (see `02_pii_phi_autotagger.py`) runs a regex pass plus a
# MAGIC Presidio analyzer pass over each new/changed column and returns the columns that look
# MAGIC like direct identifiers or clinical content. Those columns get tagged in Unity Catalog
# MAGIC immediately — classification never lags behind ingestion.

# COMMAND ----------

@dlt.table(
    name="silver_patient_encounters",
    comment="Parsed FHIR Encounter + Patient resources, deduplicated on encounter id.",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("valid_encounter_id", "encounter_id IS NOT NULL")
def silver_patient_encounters():
    bronze = dlt.read_stream("bronze_ehr_fhir")
    parsed = (
        bronze.select(
            F.get_json_object("value", "$.id").alias("encounter_id"),
            F.get_json_object("value", "$.subject.reference").alias("patient_ref"),
            F.get_json_object("value", "$.subject.display").alias("patient_name"),
            F.get_json_object("value", "$.identifier[0].value").alias("mrn"),
            F.get_json_object("value", "$.class.display").alias("encounter_class"),
            F.get_json_object("value", "$.period.start").alias("period_start"),
            F.get_json_object("value", "$.location[0].location.display").alias("unit"),
            F.get_json_object("value", "$.reasonCode[0].coding[0].code").alias("diagnosis_code"),
            F.current_timestamp().alias("_ingested_at"),
        )
        .dropDuplicates(["encounter_id"])
    )
    return parsed


@dlt.table(
    name="silver_lab_results",
    comment="Parsed lab results linked to encounter/patient.",
    table_properties={"quality": "silver"},
)
def silver_lab_results():
    bronze = dlt.read_stream("bronze_lab_results")
    return bronze.select(
        F.get_json_object("value", "$.patient_mrn").alias("mrn"),
        # Optional: which visit the lab belongs to. Null for feeds (and older files) that don't send it.
        F.get_json_object("value", "$.encounter_id").alias("lab_encounter_id"),
        F.get_json_object("value", "$.test_code").alias("test_code"),
        F.get_json_object("value", "$.result_value").alias("result_value"),
        F.get_json_object("value", "$.result_flag").alias("result_flag"),
        F.get_json_object("value", "$.observed_at").alias("observed_at"),
        F.current_timestamp().alias("_ingested_at"),
    )


@dlt.table(
    name="operational.silver_bed_availability",
    comment="Operational bed availability by unit. Non-PHI by construction — no patient identifiers.",
    table_properties={"quality": "silver"},
)
def silver_bed_availability():
    bronze = dlt.read_stream("operational.bronze_scheduling")
    return bronze.select(
        F.get_json_object("value", "$.unit").alias("unit"),
        F.get_json_object("value", "$.beds_total").cast("int").alias("beds_total"),
        F.get_json_object("value", "$.beds_available").cast("int").alias("beds_available"),
        F.get_json_object("value", "$.as_of").alias("as_of"),
    )

# COMMAND ----------

# MAGIC %md ## Gold — aggregated / query-ready

# COMMAND ----------

@dlt.table(
    name="gold_patient_summary",
    comment="One row per patient encounter enriched with latest labs. PHI — Gold does not remove sensitivity, it only shapes the data for retrieval.",
    table_properties={"quality": "gold"},
)
def gold_patient_summary():
    enc = dlt.read("silver_patient_encounters")
    labs = dlt.read("silver_lab_results").withColumnRenamed("_ingested_at", "_lab_ingested_at").withColumnRenamed("mrn", "lab_mrn")
    # A lab that names its encounter attaches only to that visit, so repeat visits don't each collect
    # all of the patient's labs. Labs without an encounter id fall back to matching on MRN.
    return enc.join(
        labs,
        (enc.mrn == labs.lab_mrn)
        & (labs.lab_encounter_id.isNull() | (labs.lab_encounter_id == enc.encounter_id)),
        how="left",
    ).drop("lab_mrn")


@dlt.table(
    name="operational.gold_bed_availability_by_unit",
    comment="Current bed availability per unit, refreshed continuously. Feeds the general-purpose vector index and the ops chat path directly.",
    table_properties={"quality": "gold"},
)
def gold_bed_availability_by_unit():
    # Latest snapshot per unit. (Taking max() of each column separately would mix snapshots and
    # report the highest availability ever seen, not the current one.)
    latest = F.max_by(F.struct("beds_total", "beds_available", "as_of"), "as_of")
    return (
        dlt.read("operational.silver_bed_availability")
        .groupBy("unit")
        .agg(latest.alias("s"))
        .select("unit", "s.beds_total", "s.beds_available", "s.as_of")
    )
