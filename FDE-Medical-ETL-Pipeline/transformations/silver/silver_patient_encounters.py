from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(
    name="silver_patient_encounters",
    comment="Parsed FHIR Encounter + Patient resources, deduplicated on encounter id.",
    table_properties={"quality": "silver"},
)
@dp.expect_or_drop("valid_encounter_id", "encounter_id IS NOT NULL")
def silver_patient_encounters():
    bronze = spark.readStream.table("bronze_ehr_fhir")
    return (
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
