# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 06 — Genie Gap Analysis Setup
# MAGIC
# MAGIC Configures a **Genie space** for natural-language exploration of care-gap patterns across the Gold tables.
# MAGIC
# MAGIC ### Genie Space Details
# MAGIC * **Name**: HEDIS Stars Clinical Analytics — Gap & What-If Analysis
# MAGIC * **Catalog**: `payer_catalog`
# MAGIC * **Schema**: `hedis_stars_clinical_demo`
# MAGIC * **Attached Tables**: All 6 Gold tables
# MAGIC
# MAGIC ### Sample Questions for Gap Analysis
# MAGIC * *"Which provider groups have the lowest care-gap closure rate?"*
# MAGIC * *"What barriers are most common among members with open care gaps?"*
# MAGIC * *"Which regions have the most persistent gaps (open > 180 days)?"*
# MAGIC * *"Show me the top 10 providers by open gap count"*
# MAGIC * *"What is the average gap closure rate by plan type?"*

# COMMAND ----------

# DBTITLE 1,Verify Gold Tables for Genie
CATALOG = "payer_catalog"
SCHEMA = "hedis_stars_clinical_demo"

gold_tables = [
    "member_quality_features",
    "care_gap_summary",
    "provider_quality_summary",
    "plan_stars_summary",
    "high_risk_member_worklist",
    "gap_analysis_detail"
]

print("=" * 60)
print("GOLD TABLES AVAILABLE FOR GENIE")
print("=" * 60)
for t in gold_tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()
    print(f"✅ {CATALOG}.{SCHEMA}.{t}: {count:,} rows")
print("=" * 60)
print("All tables ready for Genie space attachment.")

# COMMAND ----------

# DBTITLE 1,Genie Space Configuration Reference
# MAGIC %md
# MAGIC ## Genie Space Configuration
# MAGIC
# MAGIC The Genie space was created with the following configuration:
# MAGIC
# MAGIC **Space ID**: `01f1c10141d11e9a89caa066ddb1cba4`
# MAGIC
# MAGIC ### Attached Tables
# MAGIC | Table | Purpose in Genie |
# MAGIC | --- | --- |
# MAGIC | `gap_analysis_detail` | Primary table for gap analysis — breaks down by provider, barrier, demographics |
# MAGIC | `care_gap_summary` | Aggregated metrics by measure, plan, region, provider group |
# MAGIC | `provider_quality_summary` | Provider-level quality scores and closure rates |
# MAGIC | `member_quality_features` | Member-level features for risk exploration |
# MAGIC | `high_risk_member_worklist` | Prioritized outreach worklist |
# MAGIC | `plan_stars_summary` | Plan-level Stars performance trends |
# MAGIC
# MAGIC ### How to Use
# MAGIC 1. Open the Genie space from the Genie tab in the workspace
# MAGIC 2. Ask questions in natural language
# MAGIC 3. Genie translates your question to SQL against the Gold tables
# MAGIC 4. Results are returned as tables and can be visualized
# MAGIC
# MAGIC ### Gap Analysis Focus Areas
# MAGIC * **Provider Performance**: Which provider groups have the lowest/highest gap closure rates?
# MAGIC * **Barrier Analysis**: What barriers most commonly prevent gap closure?
# MAGIC * **Geographic Patterns**: Which regions or states have the most open gaps?
# MAGIC * **Measure Analysis**: Which HEDIS measures have the highest open-gap rates?
# MAGIC * **Demographics**: How do gap patterns vary by age band, dual eligibility, social risk?
# MAGIC * **Outreach Effectiveness**: Is higher outreach count associated with better closure rates?
# MAGIC * **Persistent Gaps**: Which members have gaps open > 180 days?

# COMMAND ----------

# DBTITLE 1,Sample Genie Queries (SQL equivalents)
# These are the SQL equivalents of what Genie generates from natural-language questions
# Showing them here for reference and validation

print("\n" + "=" * 70)
print("SAMPLE GAP ANALYSIS QUERIES (Genie translates NL to these)")
print("=" * 70)

# Q1: Provider groups with lowest closure rates
print("\nQ1: Which provider groups have the lowest gap closure rate?")
display(
    spark.sql(f"""
        SELECT provider_group,
               COUNT(*) as total_gaps,
               SUM(gap_is_open) as open_gaps,
               ROUND(1 - SUM(gap_is_open) / COUNT(*), 3) as closure_rate
        FROM {CATALOG}.{SCHEMA}.gap_analysis_detail
        WHERE provider_group IS NOT NULL
        GROUP BY provider_group
        ORDER BY closure_rate ASC
        LIMIT 10
    """)
)

# COMMAND ----------

# DBTITLE 1,Sample Query 2: Barrier Analysis
# Q2: Most common barriers among open gaps
print("\nQ2: What barriers are most common among members with open care gaps?")
display(
    spark.sql(f"""
        SELECT barrier_category,
               COUNT(*) as total_gaps,
               SUM(gap_is_open) as open_gaps,
               ROUND(AVG(days_open), 1) as avg_days_open
        FROM {CATALOG}.{SCHEMA}.gap_analysis_detail
        WHERE gap_is_open = 1
        GROUP BY barrier_category
        ORDER BY open_gaps DESC
    """)
)

# COMMAND ----------

# DBTITLE 1,Sample Query 3: Regional Persistent Gaps
# Q3: Regions with the most persistent gaps
print("\nQ3: Which regions have the most persistent gaps (open > 180 days)?")
display(
    spark.sql(f"""
        SELECT region,
               COUNT(*) as persistent_gaps,
               ROUND(AVG(days_open), 1) as avg_days_open,
               COUNT(DISTINCT member_id) as affected_members
        FROM {CATALOG}.{SCHEMA}.gap_analysis_detail
        WHERE persistent_gap = 1
        GROUP BY region
        ORDER BY persistent_gaps DESC
    """)
)

print("\n" + "=" * 70)
print("✅ Genie Gap Analysis setup and validation complete!")
print("Open the Genie space to ask these questions in natural language.")
print("=" * 70)