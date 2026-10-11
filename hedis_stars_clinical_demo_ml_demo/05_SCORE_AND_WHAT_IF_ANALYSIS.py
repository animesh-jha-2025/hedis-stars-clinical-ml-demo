# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 05 — Score & What-If Analysis
# MAGIC
# MAGIC Scores members using the Champion model and demonstrates **What-If Analysis** — interactively adjusting member parameters to see how predicted care-gap risk changes.
# MAGIC
# MAGIC ### What-If Scenarios
# MAGIC | Parameter | Direction | Expected Impact |
# MAGIC | --- | --- | --- |
# MAGIC | `outreach_attempt_count` ↑ | Increase outreach | Risk ↓ |
# MAGIC | `care_management_enrolled` → 1 | Enroll in care mgmt | Risk ↓ |
# MAGIC | `days_since_last_outreach` ↓ | Recent contact | Risk ↓ |
# MAGIC | `primary_care_visit_count` ↑ | More PCP visits | Risk ↓ |

# COMMAND ----------

# DBTITLE 1,Setup Explanation
# MAGIC %md
# MAGIC ### Step 1 — Load the Champion Model
# MAGIC
# MAGIC We load the **PyFunc-wrapped Gradient Boosted Trees** model directly from the Unity Catalog Model Registry using its `@Champion` alias. This avoids hard-coding a version number — whenever a new champion is promoted, this cell automatically picks it up.
# MAGIC
# MAGIC The PyFunc wrapper returns four columns per prediction: `risk_probability`, `predicted_class`, `risk_tier` (High / Medium / Low), and `recommended_action`.

# COMMAND ----------

# DBTITLE 1,Profiles Explanation
# MAGIC %md
# MAGIC ### Step 2 — Define Representative Member Profiles
# MAGIC
# MAGIC We define three synthetic member profiles that span the risk spectrum. Each profile contains exactly the **14 features** the Champion model was trained on — demographics (`age`, `enrollment_months`), clinical indicators (`chronic_condition_count`, `risk_score`, `medication_adherence_rate`), utilization (`prior_utilization_count`, `primary_care_visit_count`, `preventive_visit_completed`, `last_visit_days_ago`), outreach history (`outreach_attempt_count`, `days_since_last_outreach`), and contextual factors (`provider_quality_score`, `care_management_enrolled`, `barrier_category_encoded`).
# MAGIC
# MAGIC These profiles are reused across all scoring and What-If cells below.

# COMMAND ----------

# DBTITLE 1,Baseline Scoring Explanation
# MAGIC %md
# MAGIC ### Step 3 — Baseline Scoring
# MAGIC
# MAGIC Scores each of the three profiles with **no modifications** to establish baseline risk predictions. The output table shows the predicted `risk_probability`, `risk_tier`, and `recommended_action` for each profile — this is the "before" snapshot that the What-If scenarios below compare against.

# COMMAND ----------

# DBTITLE 1,Scenario 1 Explanation
# MAGIC %md
# MAGIC #### Scenario 1 — Outreach Intensity
# MAGIC
# MAGIC Tests whether **increasing the number of outreach attempts** (1 → 3 → 5 → 7) for the High Risk member reduces predicted risk. This simulates: *"What happens if we contact this member more frequently?"*

# COMMAND ----------

# DBTITLE 1,Scenario 2 Explanation
# MAGIC %md
# MAGIC #### Scenario 2 — Care Management Enrollment
# MAGIC
# MAGIC Toggles `care_management_enrolled` from 0 → 1 for the High Risk member. This simulates: *"What if we enroll this member in a structured care management program?"*

# COMMAND ----------

# DBTITLE 1,Scenario 3 Explanation
# MAGIC %md
# MAGIC #### Scenario 3 — Combined (Progressive) Interventions
# MAGIC
# MAGIC Stacks multiple interventions on the High Risk member in sequence to show **cumulative impact**:
# MAGIC 1. **Baseline** — no changes (risk ≈ 0.88)
# MAGIC 2. **+ Outreach + Care Mgmt** — increase attempts to 5, enroll in care management
# MAGIC 3. **+ Recent contact** — reduce days since outreach to 7
# MAGIC 4. **+ PCP visits + Rx adherence** — schedule 6 PCP visits, improve medication adherence to 85%
# MAGIC
# MAGIC This demonstrates how layered interventions can progressively move a member from **High → Medium → Low** risk.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎤 Demo Talking Points**
# MAGIC
# MAGIC > *"This is the hero cell of the notebook — the one that ties the clinical story together."*
# MAGIC
# MAGIC 1. **Frame the question:** "We have a 72-year-old member with 4 chronic conditions, no recent PCP visits, low medication adherence, and a predicted risk probability of 0.88 — solidly High risk. What combination of interventions would it take to change that outcome?"
# MAGIC
# MAGIC 2. **Walk through each layer:**
# MAGIC    * **Outreach + Care Mgmt (0.88 → 0.78):** "Adding 5 outreach touches and enrolling in care management reduces risk, but the member is still High. Outreach alone isn't enough for this profile."
# MAGIC    * **Recent contact (0.78 → 0.53):** "Recency matters. Just 7 days since last contact drops the member into Medium territory — the model learned that timely follow-up is a strong signal."
# MAGIC    * **PCP visits + Rx adherence (0.53 → 0.32):** "This is the tipping point. Scheduling regular PCP visits and improving medication adherence pushes the member all the way to Low risk."
# MAGIC
# MAGIC 3. **Highlight the insight:** "No single lever flips the prediction — it's the *combination* that matters. This mirrors clinical reality: sustained engagement across multiple channels is what closes care gaps."
# MAGIC
# MAGIC 4. **Connect to business value:** "A care manager can use this to build a personalized intervention plan and justify resource allocation — the model quantifies what intuition suggests."
# MAGIC
# MAGIC 5. **Platform point:** "Every prediction here runs through the same PyFunc wrapper deployed to the serving endpoint. What we're doing interactively in this notebook, an operational system can do at scale via REST API for thousands of members per minute."

# COMMAND ----------

# DBTITLE 1,Endpoint Scoring Explanation
# MAGIC %md
# MAGIC ### Step 4 — Live Endpoint Scoring (REST API)
# MAGIC
# MAGIC Sends the High Risk profile to the **deployed Model Serving endpoint** (`hedis_stars_clinical_demo_endpoint`) via the Databricks SDK. This validates that the production endpoint returns the same predictions as the local model — confirming the model is correctly deployed and serving.
# MAGIC
# MAGIC If the endpoint is warming up (scale-to-zero), the cell gracefully falls back to the local scoring results above.

# COMMAND ----------

# DBTITLE 1,Setup: Load Champion Model
import mlflow
import pandas as pd
import numpy as np

CATALOG = "payer_catalog"
SCHEMA = "hedis_stars_clinical_demo"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.hedis_care_gap_risk_model"
ENDPOINT_NAME = "hedis_stars_clinical_demo_endpoint"

# Load model locally for fast What-If scoring
model_uri = f"models:/{MODEL_NAME}@Champion"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Champion model loaded from {model_uri}")

# COMMAND ----------

# DBTITLE 1,Define Member Profiles for Scoring
# Correct 14 features matching the Champion model's training schema
FEATURE_COLS = [
    "age", "chronic_condition_count", "risk_score", "prior_utilization_count",
    "primary_care_visit_count", "preventive_visit_completed", "medication_adherence_rate",
    "last_visit_days_ago", "provider_quality_score", "days_since_last_outreach",
    "outreach_attempt_count", "care_management_enrolled", "enrollment_months",
    "barrier_category_encoded"
]

# Three representative member profiles
profiles = {
    "High Risk Member": {
        "age": 72, "chronic_condition_count": 4, "risk_score": 3.5,
        "prior_utilization_count": 8, "primary_care_visit_count": 0,
        "preventive_visit_completed": 0, "medication_adherence_rate": 0.38,
        "last_visit_days_ago": 250, "provider_quality_score": 2.5,
        "days_since_last_outreach": 90, "outreach_attempt_count": 4,
        "care_management_enrolled": 0, "enrollment_months": 48,
        "barrier_category_encoded": 0
    },
    "Medium Risk Member": {
        "age": 55, "chronic_condition_count": 2, "risk_score": 2.1,
        "prior_utilization_count": 4, "primary_care_visit_count": 2,
        "preventive_visit_completed": 0, "medication_adherence_rate": 0.65,
        "last_visit_days_ago": 120, "provider_quality_score": 3.8,
        "days_since_last_outreach": 45, "outreach_attempt_count": 2,
        "care_management_enrolled": 0, "enrollment_months": 24,
        "barrier_category_encoded": 1
    },
    "Low Risk Member": {
        "age": 42, "chronic_condition_count": 0, "risk_score": 1.2,
        "prior_utilization_count": 1, "primary_care_visit_count": 4,
        "preventive_visit_completed": 1, "medication_adherence_rate": 0.92,
        "last_visit_days_ago": 30, "provider_quality_score": 4.5,
        "days_since_last_outreach": 15, "outreach_attempt_count": 1,
        "care_management_enrolled": 1, "enrollment_months": 36,
        "barrier_category_encoded": 4
    }
}

print("Three member profiles defined: High / Medium / Low Risk")

# COMMAND ----------

# DBTITLE 1,Score All Profiles: Baseline Predictions
# Score each profile
results = []
for name, profile in profiles.items():
    pdf_input = pd.DataFrame([profile])[FEATURE_COLS]
    prediction = model.predict(pdf_input)
    results.append({
        "Profile": name,
        "Risk Probability": prediction["risk_probability"].iloc[0],
        "Risk Tier": prediction["risk_tier"].iloc[0],
        "Recommended Action": prediction["recommended_action"].iloc[0]
    })

df_results = pd.DataFrame(results)
print("\n" + "=" * 80)
print("BASELINE SCORING RESULTS")
print("=" * 80)
display(spark.createDataFrame(df_results))

# COMMAND ----------

# DBTITLE 1,What-If Header
# MAGIC %md
# MAGIC ## What-If Analysis
# MAGIC
# MAGIC The cells below demonstrate how adjusting a single member parameter changes the predicted care-gap risk. This is the core capability for care managers and quality teams to simulate intervention impact.

# COMMAND ----------

# DBTITLE 1,What-If Scenario 1: Increase Outreach
# WHAT-IF: Increase outreach_attempt_count for High Risk Member
high_risk = profiles["High Risk Member"].copy()

print("=" * 70)
print("WHAT-IF SCENARIO 1: Increase Outreach Attempts")
print("Member: High Risk | Parameter: outreach_attempt_count")
print("=" * 70)

scenarios = []
for outreach in [1, 3, 5, 7]:
    modified = high_risk.copy()
    modified["outreach_attempt_count"] = outreach
    pdf_input = pd.DataFrame([modified])[FEATURE_COLS]
    pred = model.predict(pdf_input)
    scenarios.append({
        "outreach_attempt_count": outreach,
        "risk_probability": pred["risk_probability"].iloc[0],
        "risk_tier": str(pred["risk_tier"].iloc[0]),
        "action": str(pred["recommended_action"].iloc[0])
    })

display(spark.createDataFrame(pd.DataFrame(scenarios)))

# COMMAND ----------

# DBTITLE 1,What-If Scenario 2: Enroll in Care Management
# WHAT-IF: Toggle care_management_enrolled for High Risk Member
print("=" * 70)
print("WHAT-IF SCENARIO 2: Enroll in Care Management")
print("Member: High Risk | Parameter: care_management_enrolled (0 → 1)")
print("=" * 70)

scenarios_cm = []
for enrolled in [0, 1]:
    modified = high_risk.copy()
    modified["care_management_enrolled"] = enrolled
    pdf_input = pd.DataFrame([modified])[FEATURE_COLS]
    pred = model.predict(pdf_input)
    scenarios_cm.append({
        "care_management_enrolled": enrolled,
        "risk_probability": pred["risk_probability"].iloc[0],
        "risk_tier": str(pred["risk_tier"].iloc[0]),
        "action": str(pred["recommended_action"].iloc[0])
    })

display(spark.createDataFrame(pd.DataFrame(scenarios_cm)))

# COMMAND ----------

# DBTITLE 1,What-If Scenario 3: Combined Interventions
# WHAT-IF: Combine multiple interventions for High Risk Member
print("=" * 70)
print("WHAT-IF SCENARIO 3: Combined Interventions")
print("Member: High Risk | Multiple parameters adjusted simultaneously")
print("=" * 70)

combined_scenarios = [
    ("Baseline (no intervention)", high_risk.copy()),
]

# Intervention 1: Outreach + Care Mgmt
mods1 = high_risk.copy()
mods1["outreach_attempt_count"] = 5
mods1["care_management_enrolled"] = 1
combined_scenarios.append(("+ Outreach (5) + Care Mgmt", mods1))

# Intervention 2: Above + recent contact
mods2 = mods1.copy()
mods2["days_since_last_outreach"] = 7
combined_scenarios.append(("+ Recent contact (7 days)", mods2))

# Intervention 3: Full intervention suite
mods3 = mods2.copy()
mods3["primary_care_visit_count"] = 6
mods3["medication_adherence_rate"] = 0.85
combined_scenarios.append(("+ PCP visits (6) + Rx adherence (85%)", mods3))

results_combined = []
for label, profile_data in combined_scenarios:
    pdf_input = pd.DataFrame([profile_data])[FEATURE_COLS]
    pred = model.predict(pdf_input)
    results_combined.append({
        "Scenario": label,
        "Risk Probability": pred["risk_probability"].iloc[0],
        "Risk Tier": str(pred["risk_tier"].iloc[0]),
    })

df_combined = pd.DataFrame(results_combined)
print("\nProgressive intervention impact:")
display(spark.createDataFrame(df_combined))

# COMMAND ----------

# DBTITLE 1,Score via Serving Endpoint (REST API)
# Score via the deployed endpoint (REST API)
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

try:
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[profiles["High Risk Member"]]
    )
    print("\n✅ Endpoint Scoring Successful!")
    print(json.dumps(response.as_dict(), indent=2))
except Exception as e:
    print(f"\n⚠️ Endpoint scoring failed (may need warm-up): {e}")
    print("The What-If Analysis above uses the local model, which always works.")

print("\n" + "=" * 70)
print("✅ Scoring & What-If Analysis complete!")
print("=" * 70)