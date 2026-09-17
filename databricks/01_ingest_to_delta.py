# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest Criteo Uplift v2.1 into Delta
# MAGIC Reads the compressed CSV from a Unity Catalog volume with PySpark, writes a Delta table,
# MAGIC and checks it against the figures the local pipeline reports.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

SOURCE = "/Volumes/workspace/upliftbench/raw/criteo-research-uplift-v2.1.csv.gz"
TABLE = "workspace.upliftbench.criteo_uplift"
EXPECTED_ROWS = 13_979_592

schema = StructType(
    [StructField(f"f{i}", DoubleType(), False) for i in range(12)]
    + [StructField(c, IntegerType(), False) for c in ("treatment", "conversion", "visit", "exposure")]
)

# COMMAND ----------

# A .gz file is not splittable, so Spark reads it as one partition in file order.
# That makes monotonically_increasing_id() a plain 0..n-1 row number, the same row_id the local loader assigns.
raw = spark.read.csv(SOURCE, header=True, schema=schema, mode="FAILFAST")
df = raw.withColumn("row_id", F.monotonically_increasing_id())

(df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE))

# COMMAND ----------

t = spark.table(TABLE)
stats = t.agg(
    F.count("*").alias("rows"),
    F.min("row_id").alias("min_row_id"),
    F.max("row_id").alias("max_row_id"),
    F.countDistinct("row_id").alias("distinct_row_ids"),
    F.avg("treatment").alias("treatment_rate"),
    F.avg("visit").alias("visit_rate"),
    F.avg("conversion").alias("conversion_rate"),
    F.sum(F.when(F.col("f0").isNull(), 1).otherwise(0)).alias("null_f0"),
).first().asDict()

assert stats["rows"] == EXPECTED_ROWS, stats
assert stats["min_row_id"] == 0 and stats["max_row_id"] == EXPECTED_ROWS - 1, stats
assert stats["distinct_row_ids"] == EXPECTED_ROWS, stats
assert stats["null_f0"] == 0, stats

# COMMAND ----------

by_arm = (
    t.groupBy("treatment")
    .agg(F.count("*").alias("rows"), F.avg("visit").alias("visit_rate"), F.avg("conversion").alias("conversion_rate"))
    .orderBy("treatment")
)
display(by_arm)

# COMMAND ----------

spark.sql(f"OPTIMIZE {TABLE}")
detail = spark.sql(f"DESCRIBE DETAIL {TABLE}").select("numFiles", "sizeInBytes").first().asDict()

import json
result = {"table": TABLE, **{k: (float(v) if isinstance(v, float) else v) for k, v in stats.items()},
          "by_arm": [r.asDict() for r in by_arm.collect()], **detail}
dbutils.notebook.exit(json.dumps(result))
