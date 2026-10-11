# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 01 — Generate & Ingest Synthetic Data
# MAGIC
# MAGIC Generates deterministic synthetic healthcare data (seed 42) and writes five **Bronze** tables to `payer_catalog.hedis_stars_clinical_demo`.
# MAGIC
# MAGIC | Table | Description | Target Rows |
# MAGIC | --- | --- | --- |
# MAGIC | `bronze_members` | Member demographics, risk, utilization | 10,000 |
# MAGIC | `bronze_providers` | Provider/group attributes and quality scores | 200 |
# MAGIC | `bronze_plans` | Plan definitions (HMO, PPO, etc.) | 12 |
# MAGIC | `bronze_care_gaps` | Member-level HEDIS-aligned care-gap records | ~16,400 |
# MAGIC | `bronze_stars_performance` | Plan/provider Stars-aligned monthly performance | ~16,200 |

# COMMAND ----------

# DBTITLE 1,Setup: Catalog, Schema, and Configuration
# Configuration
CATALOG = "payer_catalog"
SCHEMA = "hedis_stars_clinical_demo"
SEED = 42

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using {CATALOG}.{SCHEMA} | Seed: {SEED}")

# COMMAND ----------

# DBTITLE 1,Generate Bronze Members (10,000 rows)
import random
import datetime
from pyspark.sql import Row
from pyspark.sql.types import *

random.seed(SEED)

STATES = ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI","NJ","VA","WA","AZ","MA"]
REGIONS = ["Northeast","Southeast","Midwest","West","Southwest"]
GENDERS = ["M","F"]
ETHNICITIES = ["White","Black","Hispanic","Asian","Other"]
RISK_CATEGORIES = ["Low","Moderate","High","Very High"]
LANGUAGES = ["English","Spanish","Mandarin","Vietnamese","Other"]

NUM_MEMBERS = 10000
NUM_PROVIDERS = 200
NUM_PLANS = 12

plan_ids = [f"PLAN_{str(i+1).zfill(3)}" for i in range(NUM_PLANS)]
provider_ids = [f"PROV_{str(i+1).zfill(4)}" for i in range(NUM_PROVIDERS)]

members = []
for i in range(NUM_MEMBERS):
    member_id = f"MBR_{str(i+1).zfill(6)}"
    age = random.randint(50, 85)
    gender = random.choice(GENDERS)
    state = random.choice(STATES)
    region = random.choice(REGIONS)
    ethnicity = random.choice(ETHNICITIES)
    language = random.choice(LANGUAGES)
    risk_score = round(random.uniform(0.5, 4.5), 2)
    risk_category = RISK_CATEGORIES[min(int(risk_score) - 0, 3)]
    chronic_condition_count = random.randint(0, 6)
    pcp_visit_count = random.randint(0, 12)
    er_visit_count = random.randint(0, 5)
    inpatient_days = random.randint(0, 15)
    rx_adherence_pct = round(random.uniform(0.3, 1.0), 2)
    plan_id = random.choice(plan_ids)
    pcp_provider_id = random.choice(provider_ids)
    enrollment_months = random.randint(6, 60)
    dual_eligible = random.choice([0, 1])
    social_risk_indicator = random.choice([0, 1])
    care_management_enrolled = random.choice([0, 0, 0, 1])  # 25% enrolled
    outreach_attempt_count = random.randint(0, 8)
    days_since_last_outreach = random.randint(0, 365)
    preferred_contact_method = random.choice(["Phone","Mail","Email","Portal"])
    
    members.append(Row(
        member_id=member_id, age=age, gender=gender, state=state, region=region,
        ethnicity=ethnicity, language=language, risk_score=risk_score,
        risk_category=risk_category, chronic_condition_count=chronic_condition_count,
        pcp_visit_count=pcp_visit_count, er_visit_count=er_visit_count,
        inpatient_days=inpatient_days, rx_adherence_pct=rx_adherence_pct,
        plan_id=plan_id, pcp_provider_id=pcp_provider_id,
        enrollment_months=enrollment_months, dual_eligible=dual_eligible,
        social_risk_indicator=social_risk_indicator,
        care_management_enrolled=care_management_enrolled,
        outreach_attempt_count=outreach_attempt_count,
        days_since_last_outreach=days_since_last_outreach,
        preferred_contact_method=preferred_contact_method
    ))

df_members = spark.createDataFrame(members)
df_members.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_members")
print(f"bronze_members: {df_members.count()} rows")
display(df_members.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Bronze Providers (200 rows)
SPECIALTIES = ["Family Medicine","Internal Medicine","Pediatrics","OB/GYN","Cardiology",
               "Endocrinology","Oncology","Gastroenterology","Pulmonology","Nephrology"]
PROVIDER_GROUPS = ["Lakeview Health","Summit Medical","Coastal Care","Prairie Health","Mountain View Medical",
                   "Riverside Clinic","Valley Health Partners","Metro Primary Care","Sunrise Health","Heritage Medical"]

random.seed(SEED + 1)
providers = []
for i in range(NUM_PROVIDERS):
    prov_id = f"PROV_{str(i+1).zfill(4)}"
    providers.append(Row(
        provider_id=prov_id,
        provider_name=f"Dr. Provider_{i+1}",
        specialty=random.choice(SPECIALTIES),
        provider_group=random.choice(PROVIDER_GROUPS),
        state=random.choice(STATES),
        region=random.choice(REGIONS),
        panel_size=random.randint(200, 2500),
        quality_score=round(random.uniform(2.0, 5.0), 2),
        accepting_new_patients=random.choice([0, 1]),
        telehealth_capable=random.choice([0, 1])
    ))

df_providers = spark.createDataFrame(providers)
df_providers.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_providers")
print(f"bronze_providers: {df_providers.count()} rows")
display(df_providers.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Bronze Plans (12 rows)
PLAN_TYPES = ["HMO","PPO","EPO","POS"]
PLAN_NAMES = ["Select Care","Premier Health","Basic Value","Gold Advantage",
              "Silver Standard","Platinum Plus","Bronze Essential","Diamond Elite",
              "Community First","Wellness Choice","Family Shield","Senior Complete"]

random.seed(SEED + 2)
plans = []
for i in range(NUM_PLANS):
    plans.append(Row(
        plan_id=f"PLAN_{str(i+1).zfill(3)}",
        plan_name=PLAN_NAMES[i],
        plan_type=random.choice(PLAN_TYPES),
        state=random.choice(STATES[:5]),
        region=random.choice(REGIONS),
        enrolled_members=random.randint(500, 3000),
        stars_overall=round(random.uniform(2.5, 5.0), 1),
        stars_quality=round(random.uniform(2.0, 5.0), 1),
        stars_experience=round(random.uniform(2.0, 5.0), 1),
        premium_tier=random.choice(["Low","Medium","High"])
    ))

df_plans = spark.createDataFrame(plans)
df_plans.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_plans")
print(f"bronze_plans: {df_plans.count()} rows")
display(df_plans)

# COMMAND ----------

# DBTITLE 1,Generate Bronze Care Gaps (~16,400 rows)
MEASURES = ["Colorectal Cancer Screening","Breast Cancer Screening","Diabetes HbA1c Control",
            "Blood Pressure Control","Medication Adherence - Diabetes",
            "Medication Adherence - Hypertension","Annual Wellness Visit","Flu Vaccination"]
GAP_STATUSES = ["Open","Closed"]
BARRIERS = ["Transportation","Language","Cost/Copay","Provider Availability",
            "Member Refusal","No Barrier Identified","Health Literacy","Social Determinants"]

random.seed(SEED + 3)
care_gaps = []
member_ids = [f"MBR_{str(i+1).zfill(6)}" for i in range(NUM_MEMBERS)]

for m_id in member_ids:
    n_measures = random.randint(1, 3)
    selected_measures = random.sample(MEASURES, n_measures)
    for measure in selected_measures:
        gap_open = random.random() < 0.55  # ~55% open gaps
        gap_status = "Open" if gap_open else "Closed"
        gap_id = f"GAP_{len(care_gaps)+1:06d}"
        identification_date = datetime.date(2026, random.randint(1, 6), random.randint(1, 28))
        closure_date = None if gap_open else identification_date + datetime.timedelta(days=random.randint(14, 120))
        barrier = random.choice(BARRIERS) if gap_open else "No Barrier Identified"
        days_open = random.randint(30, 300) if gap_open else 0
        outreach_count = random.randint(0, 5)
        
        care_gaps.append(Row(
            gap_id=gap_id, member_id=m_id, measure_name=measure,
            gap_status=gap_status, identification_date=str(identification_date),
            closure_date=str(closure_date) if closure_date else None,
            barrier_category=barrier, days_open=days_open,
            outreach_attempts=outreach_count,
            last_outreach_date=str(identification_date + datetime.timedelta(days=random.randint(1, 60))),
            measurement_year=2026
        ))

df_care_gaps = spark.createDataFrame(care_gaps)
df_care_gaps.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_care_gaps")
print(f"bronze_care_gaps: {df_care_gaps.count()} rows")
display(df_care_gaps.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Bronze Stars Performance (~16,200 rows)
random.seed(SEED + 4)
stars_rows = []
MONTHS = list(range(1, 10))  # Jan-Sep 2026

for plan_id in plan_ids:
    for prov_id in random.sample(provider_ids, min(150, NUM_PROVIDERS)):
        for month in MONTHS:
            stars_rows.append(Row(
                plan_id=plan_id,
                provider_id=prov_id,
                measurement_month=f"2026-{month:02d}",
                measure_name=random.choice(MEASURES),
                eligible_members=random.randint(20, 500),
                compliant_members=random.randint(5, 400),
                compliance_rate=round(random.uniform(0.30, 0.95), 3),
                stars_rating=round(random.uniform(2.0, 5.0), 1),
                improvement_flag=random.choice([0, 0, 0, 1])
            ))

df_stars = spark.createDataFrame(stars_rows)
df_stars.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_stars_performance")
print(f"bronze_stars_performance: {df_stars.count()} rows")
display(df_stars.limit(5))

# COMMAND ----------

# DBTITLE 1,Validation: Verify All Bronze Tables
# Validate all Bronze tables exist and have expected row counts
bronze_tables = [
    ("bronze_members", 10000),
    ("bronze_providers", 200),
    ("bronze_plans", 12),
    ("bronze_care_gaps", None),  # variable
    ("bronze_stars_performance", None)  # variable
]

print("=" * 60)
print("BRONZE TABLE VALIDATION")
print("=" * 60)
all_ok = True
for table_name, expected in bronze_tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table_name}").count()
    status = "✅" if (expected is None or count == expected) else "❌"
    if status == "❌":
        all_ok = False
    expected_str = f"(expected {expected:,})" if expected else ""
    print(f"{status} {table_name}: {count:,} rows {expected_str}")

print("=" * 60)
print("ALL BRONZE TABLES VALIDATED SUCCESSFULLY ✅" if all_ok else "VALIDATION ISSUES DETECTED ❌")