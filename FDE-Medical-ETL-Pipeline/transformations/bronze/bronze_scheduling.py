from pyspark import pipelines as dp


@dp.table(
    name="operational.bronze_scheduling",
    comment="Raw bed/scheduling feed. Operational, not clinical.",
    table_properties={"quality": "bronze"},
)
def bronze_scheduling():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")
        .load("/Volumes/hospital_lakehouse/landing/scheduling/")
    )
