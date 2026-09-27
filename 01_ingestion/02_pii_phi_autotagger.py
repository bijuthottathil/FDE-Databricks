# Databricks notebook source
# MAGIC %md
# MAGIC # PII/PHI auto-tagger — Bronze → Silver
# MAGIC
# MAGIC Runs against every Silver table's schema after each DLT update. Two detectors:
# MAGIC 1. **Regex** for structured identifiers Presidio tends to miss in this domain (MRN formats,
# MAGIC    HL7-style encounter ids, internal accession numbers).
# MAGIC 2. **Presidio** (`AnalyzerEngine`) sampled over column values, for names, dates of birth,
# MAGIC    addresses, phone numbers, and clinical free text.
# MAGIC
# MAGIC Columns that trip either detector are tagged `phi` (clinical context) or `pii` (identifying
# MAGIC but non-clinical, e.g. a billing contact phone number) in Unity Catalog immediately — this
# MAGIC is what lets `00_governance` policies and the `03_routing` classifier stay correct as new
# MAGIC columns land, without a human remembering to tag them.

# COMMAND ----------

# %pip install + dbutils.library.restartPython() only work in a real notebook cell —
# they're no-ops when this file is executed via "Run file", so install directly instead.
import subprocess
import sys

try:
    import presidio_analyzer  # noqa: F401
except ImportError:
    # presidio-analyzer pulls in a pydantic/pydantic-core that needs typing_extensions>=4.13
    # (for Sentinel) — upgrade it explicitly, since the runtime's preinstalled version is older
    # and a plain install of presidio-analyzer alone won't touch an already-satisfied dependency.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                            "typing_extensions>=4.13.0", "presidio-analyzer", "presidio-anonymizer"])
    # The kernel already imported the runtime's older typing_extensions before this cell ran;
    # drop it from the module cache so the import below picks up the just-installed version
    # (pydantic-core, a presidio-analyzer dependency, needs >=4.13 for `Sentinel`).
    for _mod in [m for m in sys.modules if m == "typing_extensions" or m.startswith("typing_extensions.")]:
        del sys.modules[_mod]

# COMMAND ----------

import re
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# en_core_web_sm rather than Presidio's default en_core_web_lg: the default model is
# auto-downloaded via pip at first use, which isn't available in every runtime.
# Install once with: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
_nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()
analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=["en"])

MRN_PATTERN = re.compile(r"^\d{7,10}$")
ENCOUNTER_ID_PATTERN = re.compile(r"^[A-Z]{2,4}-\d{6,}$")

# Column-name heuristics catch obvious cases cheaply before spending Presidio calls.
PHI_NAME_HINTS = {"diagnosis", "icd", "cpt", "mrn", "encounter", "lab", "result", "medication", "allergy"}
PII_NAME_HINTS = {"phone", "email", "address", "ssn", "dob", "birth"}

PRESIDIO_ENTITIES_PHI = {"MEDICAL_LICENSE", "US_SSN", "PERSON", "DATE_TIME", "LOCATION"}
# Entities strong enough to mark a whole *operational* table sensitive on their own.
STRONG_ENTITIES = {"MEDICAL_LICENSE", "US_SSN"}
MIN_SCORE = 0.6

# Known non-identifying columns: pipeline metadata, codes, unit names, bed counts.
# Presidio reads short strings like "ICU" or numbers as entities, so skip these outright.
NEVER_SENSITIVE = {"unit", "test_code", "beds_total", "beds_available", "as_of"}


def classify_column(schema: str, column: str, sample_values) -> tuple[str | None, bool]:
    """Return (label, strong) where label is 'phi', 'pii', or None for a single column.

    `strong` means the match is an unambiguous identifier (MRN format, SSN, license) and
    may mark an operational table sensitive; weak NER hits alone never do.
    """
    col_lower = column.lower()

    if col_lower.startswith("_") or col_lower in NEVER_SENSITIVE:
        return None, False

    if sample_values and MRN_PATTERN.match(str(sample_values[0])):
        return "phi", True
    if any(h in col_lower for h in PHI_NAME_HINTS):
        return "phi", False
    if any(h in col_lower for h in PII_NAME_HINTS):
        return "pii", False

    # Presidio pass over a small sample — cheap enough to run per-column, per-refresh.
    hits = []
    for v in sample_values[:25]:
        if v is None:
            continue
        results = analyzer.analyze(text=str(v), language="en")
        hits.extend(r.entity_type for r in results if r.score >= MIN_SCORE)

    # Dates are only PHI in a clinical context (dates of service); in operational data
    # they're timestamps like `as_of`.
    if schema == "operational":
        hits = [h for h in hits if h != "DATE_TIME"]

    strong = any(h in STRONG_ENTITIES for h in hits)
    if any(h in PRESIDIO_ENTITIES_PHI for h in hits):
        return "phi", strong
    if hits:
        return "pii", strong
    return None, False

# COMMAND ----------


def autotag_table(catalog: str, schema: str, table: str):
    full_name = f"{catalog}.{schema}.{table}"
    df = spark.table(full_name)

    # Guard: an empty read is indistinguishable from "nothing to classify" — but it also happens
    # when the ABAC row filter hides every row from the caller (e.g. not yet in
    # phi_service_principals, or group membership still propagating). Classifying that would
    # UNSET every existing tag, so leave the table's tags untouched instead.
    if df.limit(1).count() == 0:
        print(f"WARNING: {full_name} returned no rows (empty, or hidden by a row filter) — skipping, existing tags left as-is")
        return None

    tagged = []
    any_strong = False
    for field in df.schema.fields:
        # Clear any stale label first so re-runs converge instead of only ever adding tags.
        spark.sql(f"ALTER TABLE {full_name} ALTER COLUMN `{field.name}` UNSET TAGS ('classification')")
        sample = [r[0] for r in df.select(field.name).limit(200).na.drop().collect()]
        label, strong = classify_column(schema, field.name, sample)
        if label:
            spark.sql(
                f"ALTER TABLE {full_name} ALTER COLUMN `{field.name}` SET TAGS ('classification' = '{label}')"
            )
            tagged.append((field.name, label))
            any_strong = any_strong or strong

    # A table with any phi/pii column is itself tagged with the strictest label present,
    # so table-level ABAC policies (00_governance) apply even to a table-level SELECT *.
    # Operational tables stay 'general' unless a column is an unambiguous identifier.
    if schema == "operational" and not any_strong:
        table_label = "general"
    elif any(l == "phi" for _, l in tagged):
        table_label = "phi"
    elif any(l == "pii" for _, l in tagged):
        table_label = "pii"
    else:
        table_label = "general"
    spark.sql(f"ALTER TABLE {full_name} SET TAGS ('classification' = '{table_label}')")

    # Row-level ABAC (00_governance) needs two more tags on PHI tables: `unit_key` on the care-unit
    # column, and a table-level `row_scope` that picks which policy applies. Tables with a unit
    # column are scoped per unit; PHI tables without one get the role gate only.
    if table_label == "phi":
        has_unit = "unit" in df.columns
        if has_unit:
            spark.sql(f"ALTER TABLE {full_name} ALTER COLUMN `unit` SET TAGS ('unit_key' = 'true')")
        spark.sql(f"ALTER TABLE {full_name} SET TAGS ('row_scope' = '{'unit' if has_unit else 'clinical'}')")

    return tagged

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run after each Silver/Gold refresh
# MAGIC Scheduled as a task in the same Databricks Job as the DLT pipeline, downstream of it,
# MAGIC so tags are always current before the routing layer or Vector Search reads them.

# COMMAND ----------

silver_tables = [
    ("hospital_lakehouse", "clinical", "silver_patient_encounters"),
    ("hospital_lakehouse", "clinical", "silver_lab_results"),
    ("hospital_lakehouse", "operational", "silver_bed_availability"),
    # Gold inherits sensitivity from Silver — tag it too so the router's registry sees it.
    ("hospital_lakehouse", "clinical", "gold_patient_summary"),
    ("hospital_lakehouse", "operational", "gold_bed_availability_by_unit"),
]

for cat, sch, tbl in silver_tables:
    result = autotag_table(cat, sch, tbl)
    print(f"{cat}.{sch}.{tbl}: {result}")

print(f"Autotagging complete: {len(silver_tables)} tables classified. PHI tables also carry unit_key/row_scope "
      "tags, so the row-filter policies in 00_governance apply to them.")
