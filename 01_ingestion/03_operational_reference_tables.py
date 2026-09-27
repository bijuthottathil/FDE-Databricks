# Databricks notebook source
# MAGIC %md
# MAGIC # Operational reference tables — non-PHI
# MAGIC
# MAGIC `facilities_info` and `policy_docs` back the `cafeteria_hours` and `general_policy` intents in
# MAGIC `03_routing/01_query_classifier_router.py`. They hold no patient data, are tagged `general`,
# MAGIC and are safe for the external-LLM path. Sample content — replace with the real hospital's.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE hospital_lakehouse.operational.facilities_info (
      facility_id STRING,
      name STRING,
      location STRING,
      opens STRING,
      closes STRING,
      notes STRING
    ) COMMENT 'Hours and locations for public facilities (cafeteria, pharmacy, chapel). Non-sensitive.'
""")
spark.sql("""
    INSERT INTO hospital_lakehouse.operational.facilities_info VALUES
      ('FAC-001', 'Main Cafeteria', 'Level 1, East Wing', '06:30', '19:30', 'Hot meals 11:00-14:00; grab-and-go all day.'),
      ('FAC-002', 'Coffee Kiosk', 'Main Lobby', '06:00', '16:00', 'Closed on public holidays.'),
      ('FAC-003', 'Outpatient Pharmacy', 'Level 1, West Wing', '08:00', '18:00', 'Closed Sundays.'),
      ('FAC-004', 'Chapel', 'Level 2, Central', '00:00', '23:59', 'Open to all, 24 hours.'),
      ('FAC-005', 'Gift Shop', 'Main Lobby', '09:00', '18:00', 'Flowers, cards and snacks. Latex balloons are not allowed in patient areas.'),
      ('FAC-006', 'Patient Library', 'Level 3, West Wing', '10:00', '17:00', 'Books and tablets can be borrowed and delivered to rooms on request.'),
      ('FAC-007', 'Blood Draw (Phlebotomy) Lab', 'Level 1, Outpatient Center', '07:00', '17:00', 'Walk-ins welcome; bring your lab order. Fasting draws before 10:00.'),
      ('FAC-008', 'Imaging Reception', 'Level B1, Central', '07:00', '21:00', 'X-ray walk-ins until 20:00. MRI and CT by appointment only.'),
      ('FAC-009', 'Visitor Information Desk', 'Main Entrance', '07:00', '21:00', 'Visitor badges, wheelchairs and directions.')
""")

spark.sql("""
    CREATE OR REPLACE TABLE hospital_lakehouse.operational.policy_docs (
      doc_id STRING,
      title STRING,
      content STRING,
      updated_at TIMESTAMP
    ) COMMENT 'Public-facing hospital policies (visitors, parking, etc). Non-sensitive.'
""")
spark.sql("""
    INSERT INTO hospital_lakehouse.operational.policy_docs VALUES
      ('POL-001', 'Visitor Policy', 'General visiting hours are 10:00-20:00 daily. Two visitors per patient at a time. ICU visits are limited to immediate family and may be restricted by the care team.', current_timestamp()),
      ('POL-002', 'Parking', 'Visitor parking is in Garage B. The first 30 minutes are free; validation is available at the front desk.', current_timestamp()),
      ('POL-003', 'Mask Policy', 'Masks are required in patient care areas during respiratory virus season and are available at all entrances.', current_timestamp()),
      ('POL-004', 'Maternity Visiting', 'One support person may stay overnight in Maternity. Siblings of the newborn may visit 14:00-19:00. Other visitors 10:00-20:00.', current_timestamp()),
      ('POL-005', 'Pediatrics Visiting', 'Parents and guardians may visit Pediatrics at any time; one may stay overnight. Other visitors 10:00-20:00 and must be over 12.', current_timestamp()),
      ('POL-006', 'Wi-Fi', 'Free guest Wi-Fi is available on the network HospitalGuest. No password is required; accept the terms on the login page.', current_timestamp()),
      ('POL-007', 'Smoke-Free Campus', 'Smoking and vaping are not permitted anywhere on the hospital campus, including parking garages.', current_timestamp()),
      ('POL-008', 'Medical Records Requests', 'Patients can request copies of their records through the patient portal or at Health Information Services, Level 1, weekdays 08:00-16:30.', current_timestamp()),
      ('POL-009', 'Interpreter Services', 'Free interpreters, including sign language, are available 24 hours a day. Ask any staff member to arrange one.', current_timestamp()),
      ('POL-010', 'Lost and Found', 'Lost items are held at the Security Office, Level B1, for 30 days. Call the main number and ask for Security.', current_timestamp()),
      ('POL-011', 'Discharge Time', 'Discharges usually happen by 11:00. Please arrange a ride in advance; the discharge lounge on Level 1 is open 09:00-17:00.', current_timestamp())
""")

# -- Which care units each clinician may see PHI for. Read by the `phi_access_filter` row filter in
# -- 00_governance. Created without REPLACE so assignments added by hand survive a re-run.
# -- The @hospital.example rows are sample clinicians that show the table's shape, including one
# -- nurse covering two units; replace them with real users to test with more accounts.
spark.sql("""
    CREATE TABLE IF NOT EXISTS hospital_lakehouse.operational.staff_assignments (
      staff_email STRING,
      assigned_unit STRING
    ) COMMENT 'Clinician to care-unit assignments used by the PHI row filter. Non-PHI.'
""")
spark.sql("""
    MERGE INTO hospital_lakehouse.operational.staff_assignments t
    USING (SELECT * FROM VALUES
      ('dr.cardio@hospital.example',  'Cardiology'),
      ('dr.neuro@hospital.example',   'Neurology'),
      ('rn.float@hospital.example',   'Maternity'),
      ('rn.float@hospital.example',   'Pediatrics'),
      ('dr.ed@hospital.example',      'ED')
      AS v(staff_email, assigned_unit)) s
    ON t.staff_email = s.staff_email AND t.assigned_unit = s.assigned_unit
    WHEN NOT MATCHED THEN INSERT *
""")

for t in ["facilities_info", "policy_docs", "staff_assignments"]:
    spark.sql(f"ALTER TABLE hospital_lakehouse.operational.{t} SET TAGS ('classification' = 'general')")

_counts = {t: spark.table(f"hospital_lakehouse.operational.{t}").count()
           for t in ["facilities_info", "policy_docs", "staff_assignments"]}
print("Reference tables ready: " + ", ".join(f"{t} ({n} rows)" for t, n in _counts.items()) + ", all tagged 'general'.")
