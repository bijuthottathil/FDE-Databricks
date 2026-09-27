from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(
    name="operational.silver_bed_availability",
    comment="Operational bed availability by unit. Non-PHI by construction — no patient identifiers.",
    table_properties={"quality": "silver"},
)
def silver_bed_availability():
    bronze = spark.readStream.table("operational.bronze_scheduling")
    return bronze.select(
        F.get_json_object("value", "$.unit").alias("unit"),
        F.get_json_object("value", "$.beds_total").cast("int").alias("beds_total"),
        F.get_json_object("value", "$.beds_available").cast("int").alias("beds_available"),
        F.get_json_object("value", "$.as_of").alias("as_of"),
    )
