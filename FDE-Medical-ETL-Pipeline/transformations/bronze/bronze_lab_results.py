from pyspark import pipelines as dp


@dp.table(
    name="bronze_lab_results",
    comment="Raw HL7 ORU lab result messages, pre-parsed to JSON by the interface engine.",
    table_properties={"quality": "bronze"},
)
def bronze_lab_results():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")
        .load("/Volumes/hospital_lakehouse/landing/lab_results/")
    )
