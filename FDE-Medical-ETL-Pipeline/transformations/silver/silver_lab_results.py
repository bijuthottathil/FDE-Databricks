from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(
    name="silver_lab_results",
    comment="Parsed lab results linked to encounter/patient.",
    table_properties={"quality": "silver"},
)
def silver_lab_results():
    bronze = spark.readStream.table("bronze_lab_results")
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
