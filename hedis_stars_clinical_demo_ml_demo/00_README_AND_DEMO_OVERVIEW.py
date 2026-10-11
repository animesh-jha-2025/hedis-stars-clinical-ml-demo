# Databricks notebook source
# DBTITLE 1,Title and Disclaimer
# MAGIC %md
# MAGIC # HEDIS Stars Clinical Analytics Demo — README & Overview
# MAGIC
# MAGIC > **⚠️ DISCLAIMER** 
# MAGIC > This demonstration uses synthetic data and illustrative analytics only. Model outputs are intended to support quality-improvement and outreach-prioritization workflows. They do not diagnose conditions, determine treatment, establish eligibility, replace clinical judgment, or represent official HEDIS or CMS Star Ratings calculations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *Project prefix:* `hedis_stars_clinical_demo` &nbsp;|&nbsp; *Catalog:* `payer_catalog` &nbsp;|&nbsp; *Schema:* `hedis_stars_clinical_demo` &nbsp;|&nbsp; *Random seed:* `42`

# COMMAND ----------

# DBTITLE 1,Business Use Case
# MAGIC %md
# MAGIC ## Business Use Case
# MAGIC
# MAGIC A healthcare payer wants to **identify eligible members with an open preventive-care quality gap** who are most likely to remain non-compliant by the end of the measurement period. The goal is to **prioritize member outreach**, care navigation, appointment scheduling, and quality-improvement workflows.
# MAGIC
# MAGIC ### Primary Measure Scenario
# MAGIC
# MAGIC > Predict the likelihood that an eligible member with an open **synthetic Colorectal Cancer Screening** care gap will remain non-compliant by the end of the measurement period.
# MAGIC
# MAGIC ### Business Narrative
# MAGIC
# MAGIC > *"The payer combines member, provider, plan, quality, care-gap, utilization, outreach, and Stars-aligned performance data. It identifies members with open quality gaps, predicts which members are least likely to close those gaps without intervention, prioritizes outreach, enables operational scoring through a model endpoint, and gives business users natural-language access to quality-performance insights."*
# MAGIC
# MAGIC ### What This Is
# MAGIC * An **operational outreach-prioritization** workflow
# MAGIC * A platform capability demonstration for clinical analytics
# MAGIC
# MAGIC ### What This Is NOT
# MAGIC * Not a clinical diagnostic tool or treatment recommendation engine
# MAGIC * Not a utilization-management determination or eligibility decisioning system
# MAGIC * Not an official HEDIS reporting system or CMS Star Ratings calculator

# COMMAND ----------

# DBTITLE 1,Architecture and Data Flow
# MAGIC %md
# MAGIC ## End-to-End Architecture & Data Flow
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────────────┐
# MAGIC │  SYNTHETIC DATA GENERATION  →  BRONZE (Raw)  →  SILVER (Clean)  →  GOLD (Business)  │
# MAGIC └──────────────────────────────────────────────────────────────────────────────┘
# MAGIC                                                          │
# MAGIC                                     ┌────────────────────────┴────────────────────────┐
# MAGIC                                     │              GOLD TABLES                       │
# MAGIC                                     └─────┬─────────────────┬─────────────────┬────────┘
# MAGIC                                           │                 │                 │
# MAGIC                                     ┌─────┴─────┐   ┌───────┴───────┐   ┌───┴───────┐
# MAGIC                                     │ ML Training │   │ Model Serving │   │ Genie NL    │
# MAGIC                                     │ (MLflow)    │   │ (Endpoint)    │   │ Analytics   │
# MAGIC                                     └────────────┘   └─────┬─────────┘   └────┬────────┘
# MAGIC                                                         │                   │
# MAGIC                                                   ┌─────┴────────┐   ┌────┴────────┐
# MAGIC                                                   │  What-If      │   │ Gap Analysis │
# MAGIC                                                   │  Analysis     │   │ (NL Queries) │
# MAGIC                                                   └──────────────┘   └─────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Key Capabilities
# MAGIC
# MAGIC | Capability | Delivery Mechanism | Description |
# MAGIC | --- | --- | --- |
# MAGIC | Data Governance | Unity Catalog Medallion | Bronze → Silver → Gold with table descriptions and column comments |
# MAGIC | ML Training & Tracking | MLflow Experiment | Baseline LR vs. Champion GBT, full metric logging |
# MAGIC | Model Serving | Databricks Endpoint | Real-time scoring via REST API |
# MAGIC | **Gap Analysis** | **Genie Agent** | Natural-language exploration of care gaps by provider, measure, region, barrier |
# MAGIC | **What-If Analysis** | **Model Serving Endpoint** | Adjust member parameters interactively and see predicted risk change in real-time |

# COMMAND ----------

# DBTITLE 1,Catalog Schema and Naming
# MAGIC %md
# MAGIC ## Catalog, Schema & Naming Conventions
# MAGIC
# MAGIC | Setting | Value |
# MAGIC | --- | --- |
# MAGIC | **Catalog** | `payer_catalog` |
# MAGIC | **Schema** | `hedis_stars_clinical_demo` |
# MAGIC | **Project Prefix** | `hedis_stars_clinical_demo` |
# MAGIC | **Random Seed** | `42` |
# MAGIC | **Target Record Volume** | \~10,000 members, \~16,000 care-gap records |
# MAGIC | **Measurement Year** | 2026 (synthetic) |
# MAGIC | **Primary Measure** | Colorectal Cancer Screening (synthetic) |

# COMMAND ----------

# DBTITLE 1,Notebook Execution Order
# MAGIC %md
# MAGIC ## Notebook Execution Order
# MAGIC
# MAGIC Run these notebooks in sequence. Each notebook validates its own outputs before the next step proceeds.
# MAGIC
# MAGIC | # | Notebook | Purpose |
# MAGIC | --- | --- | --- |
# MAGIC | 0 | `00_README_AND_DEMO_OVERVIEW` | This notebook — documentation and demo guide |
# MAGIC | 1 | `01_GENERATE_AND_INGEST_SYNTHETIC_DATA` | Generate deterministic synthetic data, write to Bronze tables |
# MAGIC | 2 | `02_TRANSFORM_AND_CREATE_GOLD_TABLES` | Clean, join, and transform into business-ready Gold tables |
# MAGIC | 3 | `03_TRAIN_TRACK_EVALUATE_WITH_MLFLOW` | Train baseline + final models, log to MLflow, register Champion |
# MAGIC | 4 | `04_DEPLOY_MODEL_SERVING_ENDPOINT` | Deploy registered Champion model to serving endpoint |
# MAGIC | 5 | `05_SCORE_AND_WHAT_IF_ANALYSIS` | Score sample members + interactive **What-If Analysis** |
# MAGIC | 6 | `06_GENIE_GAP_ANALYSIS_SETUP` | Configure Genie space for **Gap Analysis** via natural language |

# COMMAND ----------

# DBTITLE 1,Created Assets Inventory
# MAGIC %md
# MAGIC ## Created Assets Inventory
# MAGIC
# MAGIC ### Bronze Tables
# MAGIC | Table | Description | Rows |
# MAGIC | --- | --- | --- |
# MAGIC | `bronze_members` | Synthetic member demographics, risk, utilization, enrollment | 10,000 |
# MAGIC | `bronze_providers` | Synthetic provider/group attributes and quality scores | 200 |
# MAGIC | `bronze_plans` | Synthetic plan definitions (HMO, PPO, etc.) | 12 |
# MAGIC | `bronze_care_gaps` | Member-level HEDIS-aligned care-gap records | 16,408 |
# MAGIC | `bronze_stars_performance` | Plan/provider Stars-aligned monthly performance | 16,200 |
# MAGIC
# MAGIC ### Gold Tables
# MAGIC | Table | Description | Rows |
# MAGIC | --- | --- | --- |
# MAGIC | `member_quality_features` | ML-ready member-measure features with target label | 15,074 |
# MAGIC | `care_gap_summary` | Aggregated care-gap metrics by measure, plan, region, provider group | 2,859 |
# MAGIC | `provider_quality_summary` | Provider-level quality performance and gap closure rates | 1,000 |
# MAGIC | `plan_stars_summary` | Plan-level Stars-aligned performance trends | 1,080 |
# MAGIC | `high_risk_member_worklist` | Business-ready outreach prioritization worklist | 5,635 |
# MAGIC | `gap_analysis_detail` | Detailed gap analysis by provider, barrier, demographic (for Genie) | 15,013 |
# MAGIC
# MAGIC ### ML & Serving Assets
# MAGIC | Asset | Value |
# MAGIC | --- | --- |
# MAGIC | **MLflow Experiment** | `/Users/<your-username>/hedis_stars_clinical_demo_experiment` |
# MAGIC | **Registered Model** | `payer_catalog.hedis_stars_clinical_demo.hedis_care_gap_risk_model` |
# MAGIC | **Champion Version** | Version 2 (PyFunc GBT with probability output) |
# MAGIC | **Champion Alias** | `Champion` |
# MAGIC | **Serving Endpoint** | `hedis_stars_clinical_demo_endpoint` |
# MAGIC | **Genie Space** | HEDIS Stars Clinical Analytics — Gap & What-If Analysis |

# COMMAND ----------

# DBTITLE 1,Gap Analysis and What-If Analysis
# MAGIC %md
# MAGIC ## Gap Analysis & What-If Analysis Capabilities
# MAGIC
# MAGIC ### Gap Analysis — Delivered via Genie Agent
# MAGIC
# MAGIC **Purpose:** Enable business users to explore care-gap patterns using natural language, without writing SQL.
# MAGIC
# MAGIC **How it works:**
# MAGIC * The Genie space is configured over the governed Gold tables: `gap_analysis_detail`, `care_gap_summary`, `provider_quality_summary`, `member_quality_features`, and `high_risk_member_worklist`
# MAGIC * Users ask questions like:
# MAGIC   * *"Which provider groups have the lowest care-gap closure rate?"*
# MAGIC   * *"What barriers are most common among members with unresolved care gaps?"*
# MAGIC   * *"Which regions should be prioritized for member outreach?"*
# MAGIC   * *"Which measures have the highest open gap rate by plan?"*
# MAGIC * Genie translates questions into SQL against the governed Gold tables and returns insights
# MAGIC
# MAGIC **Key Gold table for Gap Analysis:** `gap_analysis_detail` breaks down care gaps by provider group, specialty, region, plan, barrier category, age band, and social risk indicator — with gap rates, persistent gap rates, and outreach metrics.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What-If Analysis — Delivered via Model Serving Endpoint
# MAGIC
# MAGIC **Purpose:** Allow users to modify member parameters and instantly see how the predicted care-gap risk changes.
# MAGIC
# MAGIC **How it works:**
# MAGIC * The scoring application loads the Champion model (GBT with probability output)
# MAGIC * Users start with a saved member profile (Low, Medium, or High risk)
# MAGIC * Users adjust parameters interactively:
# MAGIC   * Increase `outreach_attempt_count` → see risk decrease
# MAGIC   * Toggle `care_management_enrolled` to 1 → see risk decrease
# MAGIC   * Reduce `days_since_last_outreach` → see risk decrease
# MAGIC   * Increase `primary_care_visit_count` → see risk decrease
# MAGIC * The model re-scores in real-time, showing:
# MAGIC   * Updated risk probability
# MAGIC   * Updated risk tier (Low / Medium / High)
# MAGIC   * Updated recommended operational action
# MAGIC
# MAGIC **Business value:** Enables care managers and quality teams to simulate intervention impact before committing resources. For example: *"What if we enroll this member in care management and schedule an outreach call?"*
# MAGIC
# MAGIC ### Risk Tier Thresholds
# MAGIC | Risk Tier | Probability Range | Illustrative Action |
# MAGIC | --- | --- | --- |
# MAGIC | **High** | ≥ 0.70 | Prioritize care-manager outreach and preventive-care scheduling support |
# MAGIC | **Medium** | 0.40 – 0.69 | Send preventive-care reminder and monitor outreach completion |
# MAGIC | **Low** | < 0.40 | Continue routine engagement and preventive-care reminders |

# COMMAND ----------

# DBTITLE 1,Demo Flow
# MAGIC %md
# MAGIC ## Demo Flow (10–15 Minutes)
# MAGIC
# MAGIC | Time | Step | Action | Key Highlight |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 0–1 min | **Introduce use case** | Open this README | Synthetic data disclaimer, outreach-prioritization goal |
# MAGIC | 1–3 min | **Show synthetic data & Bronze** | Open notebook 01 | Member, provider, care-gap source data in Bronze tables |
# MAGIC | 3–5 min | **Show governed Gold tables** | Open notebook 02 | Feature table, care-gap summary, provider quality, Stars trends |
# MAGIC | 5–7 min | **Show MLflow tracking** | Open notebook 03 | Baseline vs. Champion runs, metrics, artifacts |
# MAGIC | 7–9 min | **Show registration & serving** | Open notebook 04 | Champion model in UC registry, endpoint READY |
# MAGIC | 9–11 min | **Score + What-If Analysis** | Open notebook 05 | Adjust parameters, watch risk probability change live |
# MAGIC | 11–14 min | **Genie Gap Analysis** | Open Genie space | Ask "Which providers have the most open gaps?" |
# MAGIC | 14–15 min | **Close** | Return to README | Summarize: Ingestion → Governance → ML → Serving → NL Analytics |

# COMMAND ----------

# DBTITLE 1,Prerequisites
# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC * Databricks workspace with **Unity Catalog** enabled
# MAGIC * **Serverless compute** (auto-selected for all notebooks)
# MAGIC * **MLflow Model Serving** enabled (for endpoint deployment)
# MAGIC * **Genie Spaces** enabled (for natural-language gap analysis)
# MAGIC * Access to catalog `payer_catalog` with schema create/write permissions

# COMMAND ----------

# DBTITLE 1,Important Notes
# MAGIC %md
# MAGIC ## Important Notes
# MAGIC
# MAGIC * **This is a capability demo**, not a production implementation. Prioritize simplicity, reliability, and clarity over technical depth.
# MAGIC * **All data is synthetic**, generated deterministically with seed `42`. No PHI, real member data, real provider data, or actual payer performance data is used.
# MAGIC * **HEDIS measures, Stars ratings, and thresholds are illustrative.** They do not represent official NCQA HEDIS results, official CMS Star Ratings, official CMS cut points, or official methodology.
# MAGIC * **Model features are illustrative associations** in synthetic data, not clinical causality. Feature importance should not be interpreted as medical evidence.
# MAGIC * **Recommended actions are operational illustrations** (outreach prioritization, appointment scheduling) — not medical advice or clinical recommendations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *Built with Databricks — Unified platform for data engineering, ML, and analytics.*

# COMMAND ----------

# DBTITLE 1,Demo Flow
# MAGIC %md
# MAGIC ## Demo Flow (10–15 minutes)
# MAGIC
# MAGIC | Time | Step | What to Show |
# MAGIC | --- | --- | --- |
# MAGIC | 0–1 min | **Introduction** | Open this README. Present the use case, disclaimer, and architecture diagram |
# MAGIC | 1–3 min | **Synthetic Data & Bronze** | Run `01_GENERATE_AND_INGEST_SYNTHETIC_DATA`. Show deterministic generation (seed 42), Unity Catalog Bronze tables |
# MAGIC | 3–5 min | **Gold Tables** | Run `02_TRANSFORM_AND_CREATE_GOLD_TABLES`. Show governed Gold tables with descriptions and column comments |
# MAGIC | 5–7 min | **MLflow Experiment** | Run `03_TRAIN_TRACK_EVALUATE_WITH_MLFLOW`. Show experiment UI — compare Baseline LR vs. Champion GBT metrics |
# MAGIC | 7–9 min | **Model Registry & Serving** | Run `04_DEPLOY_MODEL_SERVING_ENDPOINT`. Show Unity Catalog model registry, Champion alias, endpoint deployment |
# MAGIC | 9–11 min | **Scoring + What-If Analysis** | Run `05_SCORE_AND_WHAT_IF_ANALYSIS`. Score a high-risk member, then adjust parameters to show real-time risk changes |
# MAGIC | 11–14 min | **Genie Gap Analysis** | Open the Genie space. Ask natural-language questions about care gaps, provider performance, and barrier analysis |
# MAGIC | 14–15 min | **Wrap-Up** | Summarize platform capabilities: governance, ML lifecycle, real-time serving, NL analytics |

# COMMAND ----------

# DBTITLE 1,Prerequisites and Important Notes
# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC * Databricks workspace with **Unity Catalog** enabled
# MAGIC * **Serverless compute** (CPU) — no GPU required
# MAGIC * **MLflow Model Serving** enabled on the workspace
# MAGIC * **Genie Spaces** enabled for natural-language analytics
# MAGIC * Catalog `payer_catalog` accessible with CREATE SCHEMA and CREATE TABLE privileges
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Important Notes
# MAGIC
# MAGIC * This is a **capability demonstration**, not a production implementation
# MAGIC * All data is **synthetic**, generated deterministically with `seed=42`
# MAGIC * HEDIS measure names, Stars ratings, and quality thresholds are **illustrative** — they do not replicate official NCQA specifications
# MAGIC * Model features represent **illustrative associations**, not validated clinical causality
# MAGIC * Risk tiers and recommended actions are **operational examples** — actual interventions require clinical review
# MAGIC * The Genie agent answers questions over governed Gold tables — it does not access raw Bronze data or external systems
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *Built on the Databricks Data Intelligence Platform — Unity Catalog · MLflow · Model Serving · Genie*