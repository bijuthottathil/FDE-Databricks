from pyspark import pipelines as dp


@dp.table(
    name="bronze_ehr_fhir",
    comment="Raw FHIR bundles landed from the EHR extract feed.",
    table_properties={"quality": "bronze"},
)
def bronze_ehr_fhir():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")
        .load("/Volumes/hospital_lakehouse/landing/ehr_fhir/")
    )
