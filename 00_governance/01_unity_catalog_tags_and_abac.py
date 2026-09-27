# Databricks notebook source
# ---
# # Unity Catalog — the PHI boundary
#
# This notebook is the single source of truth for sensitivity classification. Everything
# downstream (retrieval, routing, guardrails) reads these tags rather than re-deriving
# sensitivity itself.
#
# 1. Define classification tags: `phi`, `pii`, `general`.
# 2. Apply tags to catalogs/schemas/tables/columns as data lands in Silver/Gold.
# 3. Define attribute-based access control (ABAC) policies keyed off those tags.
# 4. Apply row-level and column-level masking functions for PHI/PII.

# -- Catalog / schema scaffold. One catalog per environment; a dedicated schema for
# -- clinical (PHI) data, separate from operational (general) data, even before tagging.
spark.sql("CREATE CATALOG IF NOT EXISTS hospital_lakehouse COMMENT 'HIPAA-compliant catalog. BAA-covered workspace only.'")
spark.sql("CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.clinical COMMENT 'EHR / HL7-FHIR / lab data. Expect PHI tags at table+column level.'")
spark.sql("CREATE SCHEMA IF NOT EXISTS hospital_lakehouse.operational COMMENT 'Bed availability, cafeteria hours, general policy. Non-sensitive by default.'")

# -- Groups referenced by the ABAC policy and masking functions below don't exist yet
# -- on a fresh account — create them here so this notebook is runnable standalone.
# -- (In production these are provisioned once, centrally, via the account console
# --  or SCIM, not from inside a data-governance notebook.)
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceConflict

_w = WorkspaceClient()
_created_any = False
for _group_name in ["clinical_staff", "phi_service_principals"]:
    try:
        _w.groups.create(display_name=_group_name)
        _created_any = True
    except ResourceConflict:
        pass  # Group already exists

if _created_any:
    time.sleep(15)  # let Unity Catalog's principal cache pick up the newly created group(s)

# -- Governed tags. Unity Catalog tags are the vocabulary every other layer
# -- (DLT auto-tagging, the router, the AI Gateway) reads.
# -- Governed tags cannot be created via SQL DDL — use the Python SDK.
from databricks.sdk.service.tags import TagPolicy, Value

try:
    _w.tag_policies.create_tag_policy(
        tag_policy=TagPolicy(
            tag_key="classification",
            description="Data sensitivity classification used by ABAC, masking, and the LLM router.",
            values=[
                Value(name="phi"),
                Value(name="pii"),
                Value(name="general"),
            ],
        )
    )
except Exception:
    pass  # Tag already exists

try:
    _w.tag_policies.create_tag_policy(
        tag_policy=TagPolicy(
            tag_key="unit_key",
            description="Marks the column holding the care unit/service, used by the ABAC row filter.",
            values=[Value(name="true")],
        )
    )
except Exception:
    pass  # Tag already exists

try:
    _w.tag_policies.create_tag_policy(
        tag_policy=TagPolicy(
            tag_key="row_scope",
            description="How a PHI table's rows are scoped: 'unit' (per care unit) or 'clinical' (role gate only).",
            values=[Value(name="unit"), Value(name="clinical")],
        )
    )
except Exception:
    pass  # Tag already exists

# -- Everything below tags and protects tables created later: by the ingestion pipeline
# -- (silver/gold) and 01_ingestion/03_operational_reference_tables.py (staff_assignments,
# -- which the row filter reads). On a fresh workspace they don't exist yet, so stop cleanly
# -- here and run this notebook again once they do.
_required = [
    "hospital_lakehouse.clinical.silver_patient_encounters",
    "hospital_lakehouse.operational.gold_bed_availability_by_unit",
    "hospital_lakehouse.operational.staff_assignments",
]
_missing = [t for t in _required if not spark.catalog.tableExists(t)]
if _missing:
    _msg = ("Governance setup complete: catalog, schemas, groups and governed tags are ready.\n"
            "Table tags, row filters and column masks were skipped because these tables don't exist yet:\n  "
            + "\n  ".join(_missing)
            + "\nRun the ingestion pipeline and 01_ingestion/03_operational_reference_tables.py, "
              "then run this notebook again.")
    print(_msg)
    if "DATABRICKS_RUNTIME_VERSION" in __import__("os").environ:
        dbutils.notebook.exit(_msg)  # noqa: F821 (defined in Databricks notebooks)
    raise SystemExit(0)

# -- Example: tag a clinical table and its sensitive columns.
# -- (In production this is applied programmatically by the DLT auto-tagger in
# --  01_ingestion/02_pii_phi_autotagger.py — this block shows the manual equivalent.)
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters SET TAGS ('classification' = 'phi')")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters ALTER COLUMN patient_name SET TAGS ('classification' = 'phi')")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters ALTER COLUMN diagnosis_code SET TAGS ('classification' = 'phi')")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters ALTER COLUMN mrn SET TAGS ('classification' = 'phi')")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters ALTER COLUMN unit SET TAGS ('unit_key' = 'true')")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters SET TAGS ('row_scope' = 'unit')")

# -- Operational data stays general.
spark.sql("ALTER TABLE hospital_lakehouse.operational.gold_bed_availability_by_unit SET TAGS ('classification' = 'general')")

# -- Attribute-based access control: row-filter on the tag, not the table.
# -- Any table/column later tagged 'phi' automatically inherits this policy —
# -- new PHI columns are governed by default, not by someone remembering to grant on them.
# -- Row filter UDF (single ABAC row filter; UC allows only one per table, so PHI gating
# -- and per-unit scoping live together): service principals see everything, clinical staff
# -- only see encounters for units they are assigned to, everyone else sees nothing.
# -- Exemptions use is_member() (workspace groups) — Free Edition has no account-level groups.
# -- Two disjoint policies, chosen by the table-level `row_scope` tag, so a table never gets
# -- two row filters: 'unit' tables (have a unit_key column) are scoped per unit; 'clinical'
# -- tables (e.g. lab results, no unit column) get the role gate only.
for _p in ["phi_read_restricted", "phi_role_gate"]:
    try:
        spark.sql(f"DROP POLICY {_p} ON SCHEMA hospital_lakehouse.clinical")
    except Exception:
        pass  # Policy doesn't exist yet
spark.sql("""
    CREATE OR REPLACE FUNCTION hospital_lakehouse.clinical.phi_access_filter(unit STRING)
    RETURNS BOOLEAN
    RETURN is_member('phi_service_principals')
      OR (is_member('clinical_staff')
          AND unit IN (SELECT assigned_unit FROM hospital_lakehouse.operational.staff_assignments
                       WHERE staff_email = current_user()))
""")

spark.sql("""
    CREATE OR REPLACE FUNCTION hospital_lakehouse.clinical.phi_role_gate()
    RETURNS BOOLEAN
    RETURN is_member('phi_service_principals') OR is_member('clinical_staff')
""")

# CREATE POLICY's principal-registration path can lag a UC-governance-level group's
# creation by a few seconds even after `is_member()` already resolves
# it — retry with backoff instead of failing on the first PRINCIPAL_DOES_NOT_EXIST.
_max_attempts = 5
for _attempt in range(1, _max_attempts + 1):
    try:
        spark.sql("""
            CREATE OR REPLACE POLICY phi_read_restricted
            ON SCHEMA hospital_lakehouse.clinical
            COMMENT 'Only clinical roles with an active treatment relationship may read PHI-tagged data.'
            ROW FILTER hospital_lakehouse.clinical.phi_access_filter
            TO `account users`
            FOR TABLES
            WHEN has_tag_value('classification', 'phi') AND has_tag_value('row_scope', 'unit')
            MATCH COLUMNS has_tag('unit_key') AS u
            USING COLUMNS (u)
        """)
        spark.sql("""
            CREATE OR REPLACE POLICY phi_role_gate
            ON SCHEMA hospital_lakehouse.clinical
            COMMENT 'PHI tables without a unit column: clinical roles only.'
            ROW FILTER hospital_lakehouse.clinical.phi_role_gate
            TO `account users`
            FOR TABLES
            WHEN has_tag_value('classification', 'phi') AND has_tag_value('row_scope', 'clinical')
        """)
        print(f"CREATE POLICY succeeded on attempt {_attempt}")
        break
    except Exception as e:
        if _attempt == _max_attempts:
            raise
        print(f"Attempt {_attempt} failed ({e.__class__.__name__}): {e}. Retrying in {_attempt * 5}s...")
        time.sleep(_attempt * 5)

# -- Per-unit row scoping is handled by the ABAC policy above (UC allows only one row filter per table),
# -- so clear any native filter left over from earlier runs.
try:
    spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters DROP ROW FILTER")
except Exception:
    pass  # No native row filter set

# -- Column masking: PHI columns are masked by default for any principal without
# -- explicit clinical-role membership. The routing layer's "does this touch PHI?"
# -- check queries these same tags — see 03_routing/01_query_classifier_router.py.
spark.sql("""
    CREATE OR REPLACE FUNCTION hospital_lakehouse.clinical.mask_phi_string(value STRING)
    RETURNS STRING
    RETURN CASE
      WHEN is_member('clinical_staff') THEN value
      ELSE '***REDACTED-PHI***'
    END
""")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters ALTER COLUMN patient_name SET MASK hospital_lakehouse.clinical.mask_phi_string")
spark.sql("ALTER TABLE hospital_lakehouse.clinical.silver_patient_encounters ALTER COLUMN mrn SET MASK hospital_lakehouse.clinical.mask_phi_string")


# ---
# ## Verification query
# Sanity-check that classification tags are queryable — the router and the AI Gateway
# both depend on `information_schema` tag lookups like this one at request time.
df = spark.sql("""
    SELECT catalog_name, schema_name, table_name, column_name, tag_name, tag_value
    FROM system.information_schema.column_tags
    WHERE tag_name = 'classification'
    ORDER BY catalog_name, schema_name, table_name
""")
try:
    display(df)
except NameError:  # running via Databricks Connect, not a notebook
    df.show(truncate=False)

print("Governance setup complete: tags, row-filter policies (phi_read_restricted, phi_role_gate) "
      "and column masks (patient_name, mrn) are in place.")
