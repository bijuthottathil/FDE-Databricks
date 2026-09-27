from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.materialized_view(
    name="operational.gold_bed_availability_by_unit",
    comment="Current bed availability per unit, refreshed continuously. Feeds the general-purpose vector index and the ops chat path directly.",
    table_properties={"quality": "gold"},
)
def gold_bed_availability_by_unit():
    # Latest snapshot per unit. (Taking max() of each column separately would mix snapshots and
    # report the highest availability ever seen, not the current one.)
    latest = F.max_by(F.struct("beds_total", "beds_available", "as_of"), "as_of")
    return (
        spark.read.table("operational.silver_bed_availability")
        .groupBy("unit")
        .agg(latest.alias("s"))
        .select("unit", "s.beds_total", "s.beds_available", "s.as_of")
    )
