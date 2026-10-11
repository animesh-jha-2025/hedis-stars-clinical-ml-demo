# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 04 — Deploy Model Serving Endpoint
# MAGIC
# MAGIC Deploys the **Champion GBT** model from Unity Catalog Model Registry to a Databricks Model Serving endpoint for real-time scoring.
# MAGIC
# MAGIC * **Model**: `payer_catalog.hedis_stars_clinical_demo.hedis_care_gap_risk_model` (Champion alias)
# MAGIC * **Endpoint**: `hedis_stars_clinical_demo_endpoint`
# MAGIC * **Workload size**: Small (serverless)
# MAGIC * **Scale-to-zero**: Enabled

# COMMAND ----------

# DBTITLE 1,Setup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput
)
import time

w = WorkspaceClient()

CATALOG = "payer_catalog"
SCHEMA = "hedis_stars_clinical_demo"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.hedis_care_gap_risk_model"
ENDPOINT_NAME = "hedis_stars_clinical_demo_endpoint"

print(f"Model: {MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Create or Update Endpoint
served_entities = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version="2",  # Champion version
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists (state: {existing.state.ready}). Updating config...")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=served_entities
    )
except Exception:
    print(f"Creating new endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities)
    )

print("Endpoint creation/update initiated.")

# COMMAND ----------

# DBTITLE 1,Wait for Endpoint to be Ready
# Poll endpoint status until READY
for i in range(60):
    ep = w.serving_endpoints.get(ENDPOINT_NAME)
    state = ep.state.ready
    config_state = ep.state.config_update
    print(f"[{i+1}] Endpoint ready: {state} | Config update: {config_state}")
    if str(state) == "READY" and str(config_state) in ["NOT_UPDATING", "None", None]:
        print(f"\n✅ Endpoint '{ENDPOINT_NAME}' is READY!")
        break
    time.sleep(30)
else:
    print(f"\n⚠️ Endpoint may still be provisioning. Check the Serving UI.")

# COMMAND ----------

# DBTITLE 1,Test Endpoint with Sample Request
import json

# Sample high-risk member payload
test_payload = {
    "dataframe_records": [{
        "age": 72, "risk_score": 3.8, "chronic_condition_count": 4,
        "pcp_visit_count": 1, "er_visit_count": 3, "inpatient_days": 8,
        "rx_adherence_pct": 0.45, "enrollment_months": 36,
        "dual_eligible": 1, "social_risk_indicator": 1,
        "care_management_enrolled": 0, "outreach_attempt_count": 1,
        "days_since_last_outreach": 200, "total_gaps": 3,
        "open_gaps": 2, "closed_gaps": 1, "avg_days_open": 180.0,
        "max_days_open": 250, "avg_outreach_attempts": 1.5,
        "gap_closure_rate": 0.333, "provider_quality_score": 2.8,
        "telehealth_capable": 0
    }]
}

try:
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=test_payload["dataframe_records"]
    )
    print("\n✅ Endpoint Test Successful!")
    print(json.dumps(response.as_dict(), indent=2))
except Exception as e:
    print(f"\n⚠️ Test request failed (endpoint may still be warming up): {e}")
    print("Try again in a few minutes.")