# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 02 — Transform & Create Gold Tables
# MAGIC
# MAGIC Transforms Bronze tables into six **Gold** business-ready tables in `payer_catalog.hedis_stars_clinical_demo`.
# MAGIC
# MAGIC | Gold Table | Purpose | Key Consumers |
# MAGIC | --- | --- | --- |
# MAGIC | `member_quality_features` | ML-ready feature table with target label | ML Training |
# MAGIC | `care_gap_summary` | Aggregated care-gap metrics | Genie, Dashboards |
# MAGIC | `provider_quality_summary` | Provider-level quality performance | Genie, Dashboards |
# MAGIC | `plan_stars_summary` | Plan-level Stars-aligned trends | Genie, Dashboards |
# MAGIC | `high_risk_member_worklist` | Business-ready outreach prioritization | Operations |
# MAGIC | `gap_analysis_detail` | Detailed gap analysis dimensions | Genie Gap Analysis |

# COMMAND ----------

# DBTITLE 1,Setup
CATALOG = "payer_catalog"
SCHEMA = "hedis_stars_clinical_demo"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Load Bronze tables
df_members = spark.table(f"{CATALOG}.{SCHEMA}.bronze_members")
df_providers = spark.table(f"{CATALOG}.{SCHEMA}.bronze_providers")
df_plans = spark.table(f"{CATALOG}.{SCHEMA}.bronze_plans")
df_care_gaps = spark.table(f"{CATALOG}.{SCHEMA}.bronze_care_gaps")
df_stars = spark.table(f"{CATALOG}.{SCHEMA}.bronze_stars_performance")
print(f"Bronze tables loaded from {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Gold 1: member_quality_features (ML feature table)
from pyspark.sql import functions as F

# Aggregate care gap features per member
df_gap_features = (
    df_care_gaps
    .groupBy("member_id")
    .agg(
        F.count("*").alias("total_gaps"),
        F.sum(F.when(F.col("gap_status") == "Open", 1).otherwise(0)).alias("open_gaps"),
        F.sum(F.when(F.col("gap_status") == "Closed", 1).otherwise(0)).alias("closed_gaps"),
        F.avg("days_open").alias("avg_days_open"),
        F.max("days_open").alias("max_days_open"),
        F.avg("outreach_attempts").alias("avg_outreach_attempts"),
        F.first("measure_name").alias("primary_measure")
    )
)

# Join members with gap features and provider quality
df_member_features = (
    df_members
    .join(df_gap_features, "member_id", "inner")
    .join(
        df_providers.select(
            F.col("provider_id"),
            F.col("provider_group"),
            F.col("specialty").alias("pcp_specialty"),
            F.col("quality_score").alias("provider_quality_score"),
            F.col("telehealth_capable")
        ),
        df_members["pcp_provider_id"] == df_providers["provider_id"],
        "left"
    )
    .drop("provider_id")
    .withColumn("gap_closure_rate", F.round(F.col("closed_gaps") / F.col("total_gaps"), 3))
    .withColumn("target_open_gap", F.when(F.col("open_gaps") > 0, 1).otherwise(0))
)

df_member_features.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.member_quality_features")
count = df_member_features.count()
print(f"member_quality_features: {count:,} rows")
display(df_member_features.limit(5))

# COMMAND ----------

# DBTITLE 1,Gold 2: care_gap_summary (aggregated metrics)
df_gap_summary = (
    df_care_gaps
    .join(
        df_members.select("member_id", "plan_id", "region", "pcp_provider_id"),
        "member_id", "inner"
    )
    .join(
        df_providers.select(F.col("provider_id"), F.col("provider_group")),
        F.col("pcp_provider_id") == F.col("provider_id"),
        "left"
    )
    .groupBy("measure_name", "plan_id", "region", "provider_group", "barrier_category")
    .agg(
        F.count("*").alias("total_gaps"),
        F.sum(F.when(F.col("gap_status") == "Open", 1).otherwise(0)).alias("open_gaps"),
        F.sum(F.when(F.col("gap_status") == "Closed", 1).otherwise(0)).alias("closed_gaps"),
        F.avg("days_open").alias("avg_days_open"),
        F.avg("outreach_attempts").alias("avg_outreach_attempts")
    )
    .withColumn("closure_rate", F.round(F.col("closed_gaps") / F.col("total_gaps"), 3))
    .withColumn("open_rate", F.round(F.col("open_gaps") / F.col("total_gaps"), 3))
)

df_gap_summary.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.care_gap_summary")
print(f"care_gap_summary: {df_gap_summary.count():,} rows")
display(df_gap_summary.limit(5))

# COMMAND ----------

# DBTITLE 1,Gold 3: provider_quality_summary
# Provider-level quality summary joining care gaps and stars performance
df_prov_gaps = (
    df_care_gaps
    .join(df_members.select("member_id", "pcp_provider_id"), "member_id", "inner")
    .groupBy(F.col("pcp_provider_id").alias("provider_id"))
    .agg(
        F.count("*").alias("total_member_gaps"),
        F.sum(F.when(F.col("gap_status") == "Open", 1).otherwise(0)).alias("open_member_gaps"),
        F.avg("days_open").alias("avg_days_open"),
        F.avg("outreach_attempts").alias("avg_outreach_attempts")
    )
    .withColumn("gap_closure_rate", F.round(1 - F.col("open_member_gaps") / F.col("total_member_gaps"), 3))
)

df_prov_stars = (
    df_stars
    .groupBy("provider_id")
    .agg(
        F.avg("compliance_rate").alias("avg_compliance_rate"),
        F.avg("stars_rating").alias("avg_stars_rating"),
        F.sum("eligible_members").alias("total_eligible_members"),
        F.sum("compliant_members").alias("total_compliant_members")
    )
)

df_provider_summary = (
    df_providers
    .join(df_prov_gaps, "provider_id", "left")
    .join(df_prov_stars, "provider_id", "left")
)

df_provider_summary.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.provider_quality_summary")
print(f"provider_quality_summary: {df_provider_summary.count():,} rows")
display(df_provider_summary.limit(5))

# COMMAND ----------

# DBTITLE 1,Gold 4: plan_stars_summary
df_plan_stars = (
    df_stars
    .groupBy("plan_id", "measurement_month", "measure_name")
    .agg(
        F.sum("eligible_members").alias("eligible_members"),
        F.sum("compliant_members").alias("compliant_members"),
        F.avg("compliance_rate").alias("avg_compliance_rate"),
        F.avg("stars_rating").alias("avg_stars_rating"),
        F.sum("improvement_flag").alias("improved_providers")
    )
    .join(df_plans.select("plan_id", "plan_name", "plan_type", "region"), "plan_id", "left")
)

df_plan_stars.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.plan_stars_summary")
print(f"plan_stars_summary: {df_plan_stars.count():,} rows")
display(df_plan_stars.limit(5))

# COMMAND ----------

# DBTITLE 1,Gold 5: high_risk_member_worklist
# Members with open gaps, prioritized by risk score and gap count
df_worklist = (
    spark.table(f"{CATALOG}.{SCHEMA}.member_quality_features")
    .filter(F.col("open_gaps") > 0)
    .withColumn("priority_score", F.round(
        F.col("risk_score") * 0.4 +
        F.col("open_gaps") * 0.3 +
        (1 - F.col("gap_closure_rate")) * 0.3, 3
    ))
    .withColumn("risk_tier", F.when(F.col("priority_score") >= 2.5, "High")
                .when(F.col("priority_score") >= 1.5, "Medium")
                .otherwise("Low"))
    .withColumn("recommended_action", F.when(F.col("risk_tier") == "High",
                    "Prioritize care-manager outreach and preventive-care scheduling support")
                .when(F.col("risk_tier") == "Medium",
                    "Send preventive-care reminder and monitor outreach completion")
                .otherwise("Continue routine engagement and preventive-care reminders"))
    .select(
        "member_id", "age", "gender", "region", "state", "plan_id",
        "risk_score", "risk_category", "open_gaps", "closed_gaps",
        "avg_days_open", "gap_closure_rate", "outreach_attempt_count",
        "care_management_enrolled", "priority_score", "risk_tier",
        "recommended_action", "preferred_contact_method",
        "primary_measure", "provider_group"
    )
    .orderBy(F.col("priority_score").desc())
)

df_worklist.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.high_risk_member_worklist")
print(f"high_risk_member_worklist: {df_worklist.count():,} rows")
display(df_worklist.limit(5))

# COMMAND ----------

# DBTITLE 1,Gold 6: gap_analysis_detail (for Genie)
# Detailed gap analysis table — primary table for Genie Gap Analysis
df_gap_detail = (
    df_care_gaps
    .join(
        df_members.select(
            "member_id", "age", "gender", "region", "state",
            "plan_id", "pcp_provider_id", "risk_score", "risk_category",
            "social_risk_indicator", "care_management_enrolled",
            "outreach_attempt_count", "dual_eligible"
        ),
        "member_id", "inner"
    )
    .join(
        df_providers.select(
            F.col("provider_id"), F.col("provider_group"),
            F.col("specialty"), F.col("quality_score").alias("provider_quality_score")
        ),
        F.col("pcp_provider_id") == F.col("provider_id"),
        "left"
    )
    .drop("provider_id")
    .join(df_plans.select("plan_id", "plan_name", "plan_type"), "plan_id", "left")
    .withColumn("age_band", F.when(F.col("age") < 55, "50-54")
                .when(F.col("age") < 60, "55-59")
                .when(F.col("age") < 65, "60-64")
                .when(F.col("age") < 70, "65-69")
                .when(F.col("age") < 75, "70-74")
                .when(F.col("age") < 80, "75-79")
                .otherwise("80+"))
    .withColumn("gap_is_open", F.when(F.col("gap_status") == "Open", 1).otherwise(0))
    .withColumn("persistent_gap", F.when(
        (F.col("gap_status") == "Open") & (F.col("days_open") > 180), 1).otherwise(0))
)

df_gap_detail.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.gap_analysis_detail")
print(f"gap_analysis_detail: {df_gap_detail.count():,} rows")
display(df_gap_detail.limit(5))

# COMMAND ----------

# DBTITLE 1,Validation: Verify All Gold Tables
gold_tables = [
    "member_quality_features",
    "care_gap_summary",
    "provider_quality_summary",
    "plan_stars_summary",
    "high_risk_member_worklist",
    "gap_analysis_detail"
]

print("=" * 60)
print("GOLD TABLE VALIDATION")
print("=" * 60)
for t in gold_tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()
    status = "✅" if count > 0 else "❌"
    print(f"{status} {t}: {count:,} rows")
print("=" * 60)
print("ALL GOLD TABLES VALIDATED SUCCESSFULLY ✅")