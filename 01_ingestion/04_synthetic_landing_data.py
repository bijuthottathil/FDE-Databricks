# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic landing data, batch 2
# MAGIC
# MAGIC Adds variety the first batch (26 single-visit inpatients) lacks:
# MAGIC - 14 new patients, 8 of them with repeat visits (24 encounters), so patient-history questions work
# MAGIC - a mix of `inpatient`, `emergency` and `outpatient` encounters, spread over Aug–Sep 2026
# MAGIC - more encounters in the thinly populated units (Maternity, Neurology, Orthopedics, Pediatrics)
# MAGIC - 2–4 labs per encounter, each carrying `encounter_id` so Gold links it to the right visit
# MAGIC - six more days of bed-availability snapshots for all eight units
# MAGIC
# MAGIC Entirely synthetic. Deterministic (fixed seed) and idempotent: files are named `*_s2_*` and
# MAGIC overwritten on re-run, so running it twice doesn't duplicate data. Runs as a notebook or
# MAGIC locally (Databricks SDK only, no Spark). Run the `hospital_ingestion` pipeline afterwards.

# COMMAND ----------

import io
import json
import random
from datetime import datetime, timedelta, timezone

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
BASE = "/Volumes/hospital_lakehouse/landing"
rng = random.Random(20260926)

# -- Unit → plausible ICD-10 codes and encounter classes.
UNITS = {
    "Cardiology":  (["I21.4", "I48.91", "I50.9", "I10"], ["inpatient", "outpatient"]),
    "ED":          (["S52.501A", "R07.9", "J45.901", "S06.0X0A"], ["emergency"]),
    "ICU":         (["J96.01", "A41.9", "J18.9"], ["inpatient"]),
    "Maternity":   (["O80", "O24.419", "O13.9"], ["inpatient", "outpatient"]),
    "Neurology":   (["I63.9", "G40.909", "G43.909"], ["inpatient", "outpatient"]),
    "Oncology":    (["C34.90", "C50.911", "C18.9"], ["outpatient", "inpatient"]),
    "Orthopedics": (["M17.11", "S72.001A", "M54.50"], ["inpatient", "outpatient"]),
    "Pediatrics":  (["J21.9", "J45.21", "A08.4"], ["inpatient", "emergency"]),
}
BED_TOTALS = {"Cardiology": 30, "ED": 25, "ICU": 20, "Maternity": 18, "Neurology": 15,
              "Oncology": 35, "Orthopedics": 20, "Pediatrics": 22}

# -- Lab reference ranges (low, high) and the spread values are drawn from.
LABS = {"GLU": (70, 99, 55, 180, 0), "CR": (0.6, 1.2, 0.4, 2.5, 2), "HGB": (12.0, 17.0, 8.0, 18.5, 1),
        "NA": (135, 145, 125, 150, 0), "WBC": (4.0, 11.0, 2.5, 18.0, 1)}

# -- (name, unit of each visit). Names don't overlap the first batch, so retrieval stays unambiguous.
PATIENTS = [
    ("Daniel Okafor",   ["Cardiology", "Cardiology", "ICU"]),
    ("Hana Suzuki",     ["Maternity", "Maternity"]),
    ("Omar Haddad",     ["ED", "Orthopedics"]),
    ("Chloe Martin",    ["Pediatrics", "Pediatrics", "ED"]),
    ("Ravi Menon",      ["Neurology", "Neurology"]),
    ("Elena Popescu",   ["Oncology", "Oncology"]),
    ("Samuel Mensah",   ["Orthopedics", "Orthopedics"]),
    ("Nadia Karim",     ["Maternity"]),
    ("Tomas Novak",     ["Neurology", "ICU"]),
    ("Ingrid Larsen",   ["Pediatrics"]),
    ("Kwame Boateng",   ["ED"]),
    ("Leila Farahani",  ["Orthopedics"]),
    ("Marco Bianchi",   ["Cardiology"]),
    ("Anya Volkova",    ["Neurology"]),
]


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def lab_value(code):
    lo, hi, vmin, vmax, dp = LABS[code]
    v = round(rng.uniform(lo, hi) if rng.random() < 0.7 else rng.uniform(vmin, vmax), dp)
    flag = "L" if v < lo else "H" if v > hi else "N"
    return (str(int(v)) if dp == 0 else str(v)), flag


def put(path, obj):
    w.files.upload(path, io.BytesIO(json.dumps(obj).encode()), overwrite=True)

# COMMAND ----------

# -- Encounters and their labs. Visits for one patient are 10–15 days apart, starting Aug 1–Aug 25,
# -- so the last visit lands by Sep 25.
encounters, labs = [], []
for i, (name, visits) in enumerate(PATIENTS):
    mrn, ref = str(8000001 + i), f"Patient/{201 + i}"
    start = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 24), hours=rng.choice([7, 9, 11, 14, 18, 22]))
    for unit in visits:
        dx_codes, classes = UNITS[unit]
        enc_id = f"ENC-{200001 + len(encounters)}"
        encounters.append({
            "resourceType": "Encounter", "id": enc_id,
            "subject": {"reference": ref, "display": name},
            "identifier": [{"value": mrn}],
            "class": {"display": rng.choice(classes)},
            "period": {"start": iso(start)},
            "location": [{"location": {"display": unit}}],
            "reasonCode": [{"coding": [{"code": rng.choice(dx_codes)}]}],
        })
        for code in rng.sample(sorted(LABS), rng.randint(2, 4)):
            value, flag = lab_value(code)
            labs.append({"patient_mrn": mrn, "encounter_id": enc_id, "test_code": code, "result_value": value,
                         "result_flag": flag, "observed_at": iso(start + timedelta(hours=rng.randint(1, 20)))})
        start += timedelta(days=rng.randint(10, 15))

for n, e in enumerate(encounters):
    put(f"{BASE}/ehr_fhir/enc_s2_{n:03d}.json", e)
for n, l in enumerate(labs):
    put(f"{BASE}/lab_results/lab_s2_{n:03d}.json", l)

# -- Bed snapshots, Sep 20–25 at 08:00, for every unit.
beds = 0
for day in range(20, 26):
    for unit, total in BED_TOTALS.items():
        put(f"{BASE}/scheduling/bed_s2_{unit}_{day}.json",
            {"unit": unit, "beds_total": total, "beds_available": rng.randint(0, max(1, total // 4)),
             "as_of": f"2026-09-{day}T08:00:00Z"})
        beds += 1

print(f"{len(encounters)} encounters ({len(PATIENTS)} patients), {len(labs)} labs, {beds} bed snapshots")
print("classes:", {c: sum(e['class']['display'] == c for e in encounters) for c in ['inpatient', 'emergency', 'outpatient']})
print("units:", {u: sum(e['location'][0]['location']['display'] == u for e in encounters) for u in UNITS})
print("dates:", min(e['period']['start'] for e in encounters), "to", max(e['period']['start'] for e in encounters))
print(f"Landing data ready in {BASE}. Next: run the ingestion pipeline with a full refresh.")
