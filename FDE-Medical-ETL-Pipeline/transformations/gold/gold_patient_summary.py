from pyspark import pipelines as dp


@dp.materialized_view(
    name="gold_patient_summary",
    comment="One row per patient encounter enriched with latest labs. PHI — Gold does not remove sensitivity, it only shapes the data for retrieval.",
    table_properties={"quality": "gold"},
)
def gold_patient_summary():
    enc = spark.read.table("silver_patient_encounters")
    labs = spark.read.table("silver_lab_results").withColumnRenamed("_ingested_at", "_lab_ingested_at").withColumnRenamed("mrn", "lab_mrn")
    # A lab that names its encounter attaches only to that visit, so repeat visits don't each collect
    # all of the patient's labs. Labs without an encounter id fall back to matching on MRN.
    return enc.join(
        labs,
        (enc.mrn == labs.lab_mrn)
        & (labs.lab_encounter_id.isNull() | (labs.lab_encounter_id == enc.encounter_id)),
        how="left",
    ).drop("lab_mrn")
