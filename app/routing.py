"""In-app router for the PHI boundary.

Everything here runs inside the Databricks App: the intent classifier is deterministic
(keyword rules, no model call), so the raw question — which may contain PHI — is never sent
to a hosted model before the PHI/general decision is made. Table sensitivity still comes from
Unity Catalog tags, as in 03_routing/01_query_classifier_router.py (INTENT_TABLE_MAP is a copy
of the map there; keep the two in sync).
"""
import re
import time
from dataclasses import dataclass

CATALOG = "hospital_lakehouse"

INTENT_TABLE_MAP = {
    "patient_clinical_lookup": [f"{CATALOG}.clinical.silver_patient_encounters",
                                f"{CATALOG}.clinical.gold_patient_summary"],
    "lab_result_lookup": [f"{CATALOG}.clinical.silver_lab_results"],
    "bed_availability": [f"{CATALOG}.operational.gold_bed_availability_by_unit"],
    "cafeteria_hours": [f"{CATALOG}.operational.facilities_info"],
    "general_policy": [f"{CATALOG}.operational.policy_docs"],
}

# Order matters: clinical intents are checked first so a mixed question ("is Jane Roe's ICU
# bed free?") lands on the PHI side.
INTENT_KEYWORDS = [
    ("lab_result_lookup", ["lab", "glucose", "result", "blood", "test", "a1c", "hemoglobin"]),
    ("patient_clinical_lookup", ["patient", "diagnos", "allerg", "medication", "mrn", "encounter",
                                 "admitted", "discharge", "condition", "treatment", "history", "her ", "his "]),
    ("bed_availability", ["bed", "capacity", "occupan"]),
    ("cafeteria_hours", ["cafeteria", "coffee", "pharmacy", "chapel", "hours", "open", "close", "food"]),
    ("general_policy", ["policy", "visitor", "visiting", "parking", "mask"]),
]


def _has_word(text: str, keyword: str) -> bool:
    """Keyword match at a word start ("lab" matches "labs", not "avai-lab-le"). A keyword written with a
    trailing space ("her ") must be a whole word."""
    if keyword.endswith(" "):
        return re.search(rf"\b{re.escape(keyword.strip())}\b", text) is not None
    return re.search(rf"\b{re.escape(keyword)}", text) is not None


def classify_intent(query: str, known_names: set[str]) -> str:
    q = f" {query.lower()} "
    # A known patient name or MRN-looking number always means clinical, whatever else is asked.
    if any(n in q for n in known_names) or any(tok.isdigit() and 7 <= len(tok) <= 10 for tok in q.split()):
        return "patient_clinical_lookup"
    for intent, words in INTENT_KEYWORDS:
        if any(_has_word(q, w) for w in words):
            return intent
    return "unknown"


@dataclass
class RouteResult:
    phi: bool
    intent: str
    tables: list[str]
    reason: str


class Registry:
    """PHI/PII-tagged tables and all existing tables, read from Unity Catalog (cached briefly)."""

    def __init__(self, sql, ttl=60):
        self._sql, self._ttl, self._at = sql, ttl, 0.0
        self.phi: set[str] = set()
        self.known: set[str] = set()
        self.patient_names: set[str] = set()

    def refresh(self):
        if time.time() - self._at < self._ttl:
            return
        rows = self._sql(f"""
            SELECT catalog_name||'.'||schema_name||'.'||table_name FROM {CATALOG}.information_schema.table_tags
            WHERE catalog_name='{CATALOG}' AND tag_name='classification' AND tag_value IN ('phi','pii')
            UNION
            SELECT catalog_name||'.'||schema_name||'.'||table_name FROM {CATALOG}.information_schema.column_tags
            WHERE catalog_name='{CATALOG}' AND tag_name='classification' AND tag_value IN ('phi','pii')""")
        self.phi = {r[0] for r in rows}
        rows = self._sql(f"SELECT table_catalog||'.'||table_schema||'.'||table_name "
                         f"FROM {CATALOG}.information_schema.tables WHERE table_catalog='{CATALOG}'")
        self.known = {r[0] for r in rows}
        # Names come from the PHI table itself, so new patients are recognised automatically.
        # Gold, not Silver: Silver masks patient_name for everyone outside clinical_staff (the app's service
        # principal included), which would leave only "***redacted-phi***" here and blind the name check.
        rows = self._sql(f"SELECT DISTINCT lower(patient_name) FROM {CATALOG}.clinical.gold_patient_summary")
        self.patient_names = {r[0] for r in rows if r[0]}
        self._at = time.time()


def route(query: str, reg: Registry) -> RouteResult:
    reg.refresh()
    intent = classify_intent(query, reg.patient_names)
    tables = INTENT_TABLE_MAP.get(intent, [])
    missing = [t for t in tables if t not in reg.known]
    if not tables:
        return RouteResult(True, intent, tables, "unrecognized intent — defaulting to PHI-safe path")
    if missing:
        return RouteResult(True, intent, tables, f"table(s) {missing} not found in Unity Catalog — PHI-safe path")
    if any(t in reg.phi for t in tables):
        return RouteResult(True, intent, tables, "intent maps to PHI-tagged table(s)")
    return RouteResult(False, intent, tables, "intent maps only to general-tagged table(s)")
