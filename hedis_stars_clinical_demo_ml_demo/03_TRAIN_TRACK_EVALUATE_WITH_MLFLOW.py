# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 03 — Train, Track & Evaluate with MLflow
# MAGIC
# MAGIC Trains a **Baseline Logistic Regression** and a **Champion Gradient Boosted Tree** to predict which members with open care gaps are most likely to remain non-compliant. Both models are logged to MLflow, and the Champion is registered to Unity Catalog.
# MAGIC
# MAGIC | Model | Type | Purpose |
# MAGIC | --- | --- | --- |
# MAGIC | Baseline LR | Logistic Regression | Interpretable baseline |
# MAGIC | Champion GBT | Gradient Boosted Trees | Best-performing model → registered as Champion |

# COMMAND ----------

# DBTITLE 1,Setup: Config and Load Features
import mlflow
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

CATALOG = "payer_catalog"
SCHEMA = "hedis_stars_clinical_demo"
CURRENT_USER = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_NAME = f"/Users/{CURRENT_USER}/hedis_stars_clinical_demo_experiment"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.hedis_care_gap_risk_model"

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")
print(f"Model registry: {MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Prepare Training Data
# Load the ML feature table
df_features = spark.table(f"{CATALOG}.{SCHEMA}.member_quality_features")

# Select numeric features for modeling
FEATURE_COLS = [
    "age", "risk_score", "chronic_condition_count", "pcp_visit_count",
    "er_visit_count", "inpatient_days", "rx_adherence_pct",
    "enrollment_months", "dual_eligible", "social_risk_indicator",
    "care_management_enrolled", "outreach_attempt_count",
    "days_since_last_outreach", "total_gaps", "open_gaps",
    "closed_gaps", "avg_days_open", "max_days_open",
    "avg_outreach_attempts", "gap_closure_rate",
    "provider_quality_score", "telehealth_capable"
]
TARGET_COL = "target_open_gap"

# Convert to pandas
pdf = df_features.select(FEATURE_COLS + [TARGET_COL]).toPandas()
pdf = pdf.fillna(0)

X = pdf[FEATURE_COLS]
y = pdf[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")
print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
print(f"Target distribution (test):  {y_test.value_counts().to_dict()}")

# COMMAND ----------

# DBTITLE 1,Train Baseline: Logistic Regression
with mlflow.start_run(run_name="Baseline_LogisticRegression") as run:
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    metrics_lr = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, y_prob_lr),
    }

    mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000, "class_weight": "balanced"})
    mlflow.log_metrics(metrics_lr)
    mlflow.sklearn.log_model(lr, "model", input_example=X_test.iloc[:1])

    print("Baseline Logistic Regression Metrics:")
    for k, v in metrics_lr.items():
        print(f"  {k}: {v:.4f}")
    lr_run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 1,Train Champion: Gradient Boosted Trees
with mlflow.start_run(run_name="Champion_GBT") as run:
    gbt = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gbt.fit(X_train, y_train)
    y_pred_gbt = gbt.predict(X_test)
    y_prob_gbt = gbt.predict_proba(X_test)[:, 1]

    metrics_gbt = {
        "accuracy": accuracy_score(y_test, y_pred_gbt),
        "precision": precision_score(y_test, y_pred_gbt),
        "recall": recall_score(y_test, y_pred_gbt),
        "f1": f1_score(y_test, y_pred_gbt),
        "roc_auc": roc_auc_score(y_test, y_prob_gbt),
    }

    mlflow.log_params({
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 200, "max_depth": 5,
        "learning_rate": 0.1, "subsample": 0.8
    })
    mlflow.log_metrics(metrics_gbt)

    # Log feature importances
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": gbt.feature_importances_
    }).sort_values("importance", ascending=False)
    mlflow.log_table(feat_imp, artifact_file="feature_importances.json")

    print("Champion GBT Metrics:")
    for k, v in metrics_gbt.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTop 5 Features:")
    print(feat_imp.head().to_string(index=False))
    gbt_run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 1,Compare Models
comparison = pd.DataFrame({
    "Metric": list(metrics_lr.keys()),
    "Baseline_LR": list(metrics_lr.values()),
    "Champion_GBT": list(metrics_gbt.values()),
}).round(4)
comparison["Winner"] = comparison.apply(
    lambda row: "GBT ✅" if row["Champion_GBT"] >= row["Baseline_LR"] else "LR", axis=1
)
print("MODEL COMPARISON")
print("=" * 50)
display(spark.createDataFrame(comparison))

# COMMAND ----------

# DBTITLE 1,Register Champion Model with PyFunc Wrapper
# PyFunc wrapper that returns probabilities + risk tiers
class CareGapRiskModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            probs = self.model.predict_proba(model_input)[:, 1]
        else:
            probs = self.model.predict_proba(pd.DataFrame(model_input))[:, 1]
        
        results = pd.DataFrame({
            "risk_probability": np.round(probs, 4),
            "risk_tier": pd.cut(probs, bins=[0, 0.4, 0.7, 1.0],
                                labels=["Low", "Medium", "High"], include_lowest=True),
            "recommended_action": pd.cut(probs, bins=[0, 0.4, 0.7, 1.0],
                                         labels=[
                                             "Continue routine engagement and preventive-care reminders",
                                             "Send preventive-care reminder and monitor outreach completion",
                                             "Prioritize care-manager outreach and preventive-care scheduling support"
                                         ], include_lowest=True)
        })
        return results

with mlflow.start_run(run_name="Champion_GBT_PyFunc_Registration") as run:
    wrapped_model = CareGapRiskModel(gbt)
    signature = infer_signature(X_test.iloc[:5], wrapped_model.predict(None, X_test.iloc[:5]))

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped_model,
        signature=signature,
        input_example=X_test.iloc[:1],
        registered_model_name=MODEL_NAME
    )
    mlflow.log_metrics(metrics_gbt)
    print(f"Champion model registered to: {MODEL_NAME}")
    reg_run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 1,Set Champion Alias
from mlflow import MlflowClient

client = MlflowClient()

# Get latest version
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(v.version for v in versions)

client.set_registered_model_alias(MODEL_NAME, "Champion", str(latest_version))
print(f"Set alias 'Champion' on {MODEL_NAME} version {latest_version}")
print(f"\n✅ Model training, evaluation, and registration complete!")