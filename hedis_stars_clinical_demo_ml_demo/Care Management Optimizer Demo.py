# Databricks notebook source
# DBTITLE 1,Care Management Optimizer — End-to-End Demo
# MAGIC %md
# MAGIC # Care Management Optimizer — End-to-End Demo
# MAGIC
# MAGIC Selects members for care-management enrollment to **maximize total estimated benefit** subject to budget, nurse-hour, enrollment, and per-provider capacity constraints.
# MAGIC
# MAGIC | Technology | Purpose |
# MAGIC |---|---|
# MAGIC | `scipy.optimize.milp` (HiGHS) | Mixed-integer linear programming |
# MAGIC | MLflow `PythonModel` | Package & version the decision logic |
# MAGIC | Unity Catalog | Model registry |
# MAGIC | Databricks Model Serving | REST API endpoint |
# MAGIC
# MAGIC > **Synthetic data only.** Benefit units are illustrative — not guaranteed savings or clinical outcomes. This is resource planning, not automated coverage denial.

# COMMAND ----------

# DBTITLE 1,Section 1 — Configuration and New Schema
# ---------- Section 1: Configuration & New Schema ----------
from datetime import datetime, timezone

TS = "20261007_135926"  # UTC timestamp suffix (generated during workspace inspection)

CATALOG   = "payer_demo"
SCHEMA    = f"solver_demo_{TS}"
FQ_SCHEMA = f"{CATALOG}.{SCHEMA}"

MEMBERS_TABLE   = f"{FQ_SCHEMA}.synthetic_members"
PROVIDER_TABLE  = f"{FQ_SCHEMA}.synthetic_provider_capacity"
MODEL_NAME      = f"{FQ_SCHEMA}.care_management_optimizer"
ENDPOINT_NAME   = f"solver-demo-{TS}"

print(f"Catalog  : {CATALOG}")
print(f"Schema   : {FQ_SCHEMA}")
print(f"Members  : {MEMBERS_TABLE}")
print(f"Providers: {PROVIDER_TABLE}")
print(f"Model    : {MODEL_NAME}")
print(f"Endpoint : {ENDPOINT_NAME}")

# Create schema (idempotent)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}` COMMENT 'Synthetic payer optimization demo. No real member data.'")
print(f"\nSchema {FQ_SCHEMA} ready.")
SCHEMA_OK = True

# COMMAND ----------

# DBTITLE 1,Dependencies note
# MAGIC %md
# MAGIC ## §2 — Dependencies
# MAGIC All packages ship pre-installed on serverless compute. We record versions here so MLflow can pin them at serving time.

# COMMAND ----------

# DBTITLE 1,Section 2 — Dependency Check
# ---------- Section 2: Minimal Dependencies ----------
# All required packages are pre-installed on serverless. Record versions.
import scipy, pandas, numpy, mlflow, databricks.sdk, json, time, math, importlib

PKG_VERSIONS = {
    "scipy":         scipy.__version__,
    "pandas":        pandas.__version__,
    "numpy":         numpy.__version__,
    "mlflow":        mlflow.__version__,
}
# databricks-sdk version via metadata
import importlib.metadata
PKG_VERSIONS["databricks-sdk"] = importlib.metadata.version("databricks-sdk")

print("Installed versions (pinned for MLflow packaging):")
for k, v in PKG_VERSIONS.items():
    print(f"  {k:20s} {v}")

from scipy.optimize import milp, LinearConstraint, Bounds
print("\nscipy.optimize.milp: OK")

# COMMAND ----------

# DBTITLE 1,Data and Optimizer note
# MAGIC %md
# MAGIC ## §3–5 — Synthetic Data, Optimizer & Verification
# MAGIC Six synthetic members across two providers. The MILP maximizes total estimated benefit subject to budget, nurse-hour, enrollment, and per-provider capacity constraints. A greedy baseline and exhaustive enumeration validate correctness.

# COMMAND ----------

# DBTITLE 1,Section 3 — Synthetic Data
# ---------- Section 3: Synthetic Data ----------
import pandas as pd

members_data = [
    {"member_id": "M001", "provider_id": "P001", "estimated_benefit": 10, "program_cost": 8, "nurse_hours": 4},
    {"member_id": "M002", "provider_id": "P001", "estimated_benefit":  9, "program_cost": 5, "nurse_hours": 3},
    {"member_id": "M003", "provider_id": "P001", "estimated_benefit":  8, "program_cost": 5, "nurse_hours": 3},
    {"member_id": "M004", "provider_id": "P002", "estimated_benefit":  6, "program_cost": 4, "nurse_hours": 2},
    {"member_id": "M005", "provider_id": "P002", "estimated_benefit":  5, "program_cost": 3, "nurse_hours": 2},
    {"member_id": "M006", "provider_id": "P002", "estimated_benefit":  4, "program_cost": 2, "nurse_hours": 1},
]
df_members = pd.DataFrame(members_data)

provider_data = [
    {"provider_id": "P001", "capacity": 1},
    {"provider_id": "P002", "capacity": 2},
]
df_providers = pd.DataFrame(provider_data)

print("Members:")
print(df_members.to_string(index=False))
print("\nProvider Capacity:")
print(df_providers.to_string(index=False))

# Save to Unity Catalog
if SCHEMA_OK:
    spark.createDataFrame(df_members).write.mode("overwrite").saveAsTable(MEMBERS_TABLE)
    spark.createDataFrame(df_providers).write.mode("overwrite").saveAsTable(PROVIDER_TABLE)
    print(f"\nSaved: {MEMBERS_TABLE}")
    print(f"Saved: {PROVIDER_TABLE}")

# Default scenario
DEFAULT_SCENARIO = {
    "members": members_data,
    "budget": 10,
    "nurse_capacity": 6,
    "max_members": 3,
    "provider_capacity": {"P001": 1, "P002": 2},
}
print(f"\nDefault scenario budget={DEFAULT_SCENARIO['budget']}, nurse_cap={DEFAULT_SCENARIO['nurse_capacity']}, max_members={DEFAULT_SCENARIO['max_members']}")

# COMMAND ----------

# DBTITLE 1,Section 4 — Optimizer (optimize_allocation)
# ---------- Section 4: Optimizer ----------
import numpy as np
import time as _time
import math as _math
from scipy.optimize import milp, LinearConstraint, Bounds

def optimize_allocation(scenario: dict) -> dict:
    """Solve care-management member selection via MILP (HiGHS)."""
    # ---- validation ----
    def _fail(msg):
        return {"status": "invalid_input", "selected_member_ids": None,
                "selected_count": None, "total_benefit": None,
                "total_cost": None, "total_nurse_hours": None,
                "solve_time_seconds": None, "message": msg,
                "provider_utilization": None}

    members = scenario.get("members")
    if not members or not isinstance(members, list):
        return _fail("members must be a non-empty list")
    if len(members) > 100:
        return _fail("At most 100 members per scenario")

    budget = scenario.get("budget")
    nurse_cap = scenario.get("nurse_capacity")
    max_mem = scenario.get("max_members")
    prov_cap = scenario.get("provider_capacity")

    if budget is None or not _math.isfinite(budget) or budget < 0:
        return _fail("budget must be finite and nonneg")
    if nurse_cap is None or not _math.isfinite(nurse_cap) or nurse_cap < 0:
        return _fail("nurse_capacity must be finite and nonneg")
    if max_mem is None or not isinstance(max_mem, int) or max_mem < 0:
        return _fail("max_members must be a nonneg integer")
    if not isinstance(prov_cap, dict) or len(prov_cap) == 0:
        return _fail("provider_capacity must be a non-empty dict")
    for pid, cap in prov_cap.items():
        if not isinstance(cap, int) or cap < 0:
            return _fail(f"provider_capacity[{pid}] must be a nonneg integer")

    ids, provs, benefits, costs, hours = [], [], [], [], []
    seen_ids = set()
    for m in members:
        mid = m.get("member_id")
        pid = m.get("provider_id")
        eb  = m.get("estimated_benefit")
        pc  = m.get("program_cost")
        nh  = m.get("nurse_hours")
        if not mid or mid in seen_ids:
            return _fail(f"Duplicate or missing member_id: {mid}")
        seen_ids.add(mid)
        if not pid:
            return _fail(f"Missing provider_id for {mid}")
        if pid not in prov_cap:
            return _fail(f"No capacity entry for provider {pid}")
        if eb is None or not _math.isfinite(eb) or eb < 0:
            return _fail(f"Invalid estimated_benefit for {mid}")
        if pc is None or not _math.isfinite(pc) or pc < 0:
            return _fail(f"Invalid program_cost for {mid}")
        if nh is None or not _math.isfinite(nh) or nh <= 0:
            return _fail(f"Invalid nurse_hours for {mid} (must be > 0)")
        ids.append(mid); provs.append(pid)
        benefits.append(float(eb)); costs.append(float(pc)); hours.append(float(nh))

    n = len(ids)
    benefits_arr = np.array(benefits)
    costs_arr    = np.array(costs)
    hours_arr    = np.array(hours)

    # ---- build MILP ----
    c = -benefits_arr  # negate for minimization

    # Resource constraints: cost <= budget, hours <= nurse_cap, count <= max_mem
    A_rows = [costs_arr, hours_arr, np.ones(n)]
    ub_vals = [float(budget), float(nurse_cap), float(max_mem)]

    # Provider constraints
    unique_provs = sorted(set(provs))
    for pid in unique_provs:
        row = np.array([1.0 if provs[i] == pid else 0.0 for i in range(n)])
        A_rows.append(row)
        ub_vals.append(float(prov_cap[pid]))

    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lb=-np.inf, ub=ub_vals)
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)

    # ---- solve ----
    t0 = _time.perf_counter()
    try:
        from scipy.optimize import milp as _milp
        result = _milp(c, integrality=integrality, bounds=bounds,
                       constraints=constraints,
                       options={"time_limit": 5.0})
    except Exception as ex:
        return {"status": "solver_error", "selected_member_ids": None,
                "selected_count": None, "total_benefit": None,
                "total_cost": None, "total_nurse_hours": None,
                "solve_time_seconds": round(_time.perf_counter() - t0, 4),
                "message": str(ex), "provider_utilization": None}
    solve_sec = round(_time.perf_counter() - t0, 4)

    # ---- interpret result ----
    if result.x is None:
        return {"status": "infeasible", "selected_member_ids": None,
                "selected_count": None, "total_benefit": None,
                "total_cost": None, "total_nurse_hours": None,
                "solve_time_seconds": solve_sec,
                "message": result.message, "provider_utilization": None}

    is_optimal = (result.status == 0)
    status_label = "optimal" if is_optimal else "feasible_time_limited"

    # integrality check
    x_raw = result.x
    tol = 1e-6
    x_bin = []
    for v in x_raw:
        if abs(v - round(v)) > tol:
            return {"status": "integrality_failure", "selected_member_ids": None,
                    "selected_count": None, "total_benefit": None,
                    "total_cost": None, "total_nurse_hours": None,
                    "solve_time_seconds": solve_sec,
                    "message": "Solution failed integrality check",
                    "provider_utilization": None}
        x_bin.append(int(round(v)))

    selected = [ids[i] for i in range(n) if x_bin[i] == 1]
    sel_idx  = [i for i in range(n) if x_bin[i] == 1]

    total_benefit = float(sum(benefits_arr[i] for i in sel_idx))
    total_cost    = float(sum(costs_arr[i] for i in sel_idx))
    total_hours   = float(sum(hours_arr[i] for i in sel_idx))

    # ---- post-solve constraint validation ----
    if total_cost > budget + tol:
        return _fail(f"Post-solve: cost {total_cost} exceeds budget {budget}")
    if total_hours > nurse_cap + tol:
        return _fail(f"Post-solve: hours {total_hours} exceeds nurse_capacity {nurse_cap}")
    if len(selected) > max_mem:
        return _fail(f"Post-solve: {len(selected)} members exceeds max_members {max_mem}")

    prov_util = {}
    for pid in unique_provs:
        cnt = sum(1 for i in sel_idx if provs[i] == pid)
        cap = prov_cap[pid]
        if cnt > cap:
            return _fail(f"Post-solve: provider {pid} has {cnt} > capacity {cap}")
        prov_util[pid] = {"selected_count": cnt, "capacity": cap,
                         "remaining_capacity": cap - cnt}

    return {
        "status": status_label,
        "selected_member_ids": selected,
        "selected_count": len(selected),
        "total_benefit": total_benefit,
        "total_cost": total_cost,
        "total_nurse_hours": total_hours,
        "solve_time_seconds": solve_sec,
        "message": result.message,
        "provider_utilization": prov_util,
    }

# Quick smoke test
res = optimize_allocation(DEFAULT_SCENARIO)
print(f"Smoke test: status={res['status']}, selected={res['selected_member_ids']}, benefit={res['total_benefit']}")

# COMMAND ----------

# DBTITLE 1,Section 5 — Baseline and Quick Verification
# ---------- Section 5: Baseline & Verification ----------
from itertools import product as _product
import copy

def greedy_allocation(scenario: dict) -> dict:
    """Greedy baseline: sort by descending benefit, add if all constraints allow."""
    members = sorted(scenario["members"], key=lambda m: m["estimated_benefit"], reverse=True)
    budget = scenario["budget"]
    nurse_cap = scenario["nurse_capacity"]
    max_mem = scenario["max_members"]
    prov_cap = dict(scenario["provider_capacity"])  # mutable copy

    selected, cost_used, hours_used = [], 0.0, 0.0
    prov_used = {p: 0 for p in prov_cap}
    for m in members:
        mc, mh, pid = m["program_cost"], m["nurse_hours"], m["provider_id"]
        if (cost_used + mc <= budget and hours_used + mh <= nurse_cap
                and len(selected) < max_mem and prov_used.get(pid, 0) < prov_cap.get(pid, 0)):
            selected.append(m["member_id"])
            cost_used += mc; hours_used += mh; prov_used[pid] += 1
    prov_util = {pid: {"selected_count": prov_used[pid], "capacity": prov_cap[pid],
                       "remaining_capacity": prov_cap[pid] - prov_used[pid]} for pid in prov_cap}
    return {"status": "greedy", "selected_member_ids": selected,
            "selected_count": len(selected),
            "total_benefit": sum(m["estimated_benefit"] for m in scenario["members"] if m["member_id"] in selected),
            "total_cost": cost_used, "total_nurse_hours": hours_used,
            "solve_time_seconds": None, "message": "greedy",
            "provider_utilization": prov_util}

def exhaustive_optimum(scenario: dict) -> tuple:
    """Brute-force all 2^n allocations. Returns (best_benefit, best_ids_set)."""
    members = scenario["members"]
    n = len(members)
    budget = scenario["budget"]
    nurse_cap = scenario["nurse_capacity"]
    max_mem = scenario["max_members"]
    prov_cap = scenario["provider_capacity"]
    best_benefit, best_ids = -1, set()
    for bits in range(1 << n):
        sel = [i for i in range(n) if bits & (1 << i)]
        if len(sel) > max_mem:
            continue
        tc = sum(members[i]["program_cost"] for i in sel)
        th = sum(members[i]["nurse_hours"] for i in sel)
        if tc > budget or th > nurse_cap:
            continue
        # provider capacity
        prov_cnt = {}
        ok = True
        for i in sel:
            pid = members[i]["provider_id"]
            prov_cnt[pid] = prov_cnt.get(pid, 0) + 1
            if prov_cnt[pid] > prov_cap.get(pid, 0):
                ok = False; break
        if not ok:
            continue
        tb = sum(members[i]["estimated_benefit"] for i in sel)
        if tb > best_benefit:
            best_benefit = tb
            best_ids = {members[i]["member_id"] for i in sel}
    return best_benefit, best_ids

# --- Three scenarios ---
scenarios = {
    "A: Original": DEFAULT_SCENARIO,
    "B: More P001 cap": {**DEFAULT_SCENARIO, "provider_capacity": {"P001": 2, "P002": 2}},
    "C: More budget":  {**DEFAULT_SCENARIO, "budget": 14},
}

rows = []
for label, sc in scenarios.items():
    g = greedy_allocation(sc)
    o = optimize_allocation(sc)
    eb, _ = exhaustive_optimum(sc)
    # Verify optimizer matches exhaustive when optimal
    if o["status"] == "optimal":
        assert abs(o["total_benefit"] - eb) < 1e-6, f"{label}: optimizer {o['total_benefit']} != exhaustive {eb}"
    for method, r in [("Greedy", g), ("Optimizer", o)]:
        pu = r["provider_utilization"] or {}
        rows.append({
            "Scenario": label, "Method": method, "Status": r["status"],
            "Selected": ", ".join(r["selected_member_ids"]) if r["selected_member_ids"] else "(none)",
            "Benefit": r["total_benefit"], "Cost": r["total_cost"],
            "Hours": r["total_nurse_hours"],
            "P001 sel": pu.get("P001", {}).get("selected_count", "-"),
            "P002 sel": pu.get("P002", {}).get("selected_count", "-"),
            "Solve(s)": r["solve_time_seconds"] if r["solve_time_seconds"] else "-",
        })

df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False))

# --- Edge-case checks ---
# Zero-capacity provider
sc_zero = copy.deepcopy(DEFAULT_SCENARIO)
sc_zero["provider_capacity"]["P001"] = 0
res_zero = optimize_allocation(sc_zero)
assert all(m not in (res_zero["selected_member_ids"] or []) for m in ["M001","M002","M003"]), "Zero-cap P001 should have no selections"
print(f"\nZero-cap P001: selected={res_zero['selected_member_ids']} ✓")

# Missing provider capacity
sc_missing = copy.deepcopy(DEFAULT_SCENARIO)
del sc_missing["provider_capacity"]["P002"]
res_missing = optimize_allocation(sc_missing)
assert res_missing["status"] == "invalid_input", "Missing provider should be invalid"
print(f"Missing P002 cap: status={res_missing['status']}, msg={res_missing['message']} ✓")

# COMMAND ----------

# DBTITLE 1,MLflow packaging note
# MAGIC %md
# MAGIC ## §6–7 — MLflow Packaging & UC Registration
# MAGIC The optimizer is wrapped in an `mlflow.pyfunc.PythonModel` with bundled code dependencies. Each input row is a complete JSON scenario; the model is self-contained with no Spark or workspace dependencies at serving time.

# COMMAND ----------

# DBTITLE 1,Section 6 — MLflow PythonModel
# ---------- Section 6: MLflow PythonModel ----------
import os, tempfile, textwrap

# Write optimizer module to a temp directory for code bundling
CODE_DIR = tempfile.mkdtemp(prefix="optimizer_code_")
MODULE_PATH = os.path.join(CODE_DIR, "optimizer_core.py")

module_code = textwrap.dedent('''
import numpy as np
import math
import time as _time
from scipy.optimize import milp, LinearConstraint, Bounds

def optimize_allocation(scenario: dict) -> dict:
    """Solve care-management member selection via MILP (HiGHS)."""
    def _fail(msg):
        return {"status": "invalid_input", "selected_member_ids": None,
                "selected_count": None, "total_benefit": None,
                "total_cost": None, "total_nurse_hours": None,
                "solve_time_seconds": None, "message": msg,
                "provider_utilization": None}

    members = scenario.get("members")
    if not members or not isinstance(members, list):
        return _fail("members must be a non-empty list")
    if len(members) > 100:
        return _fail("At most 100 members per scenario")

    budget = scenario.get("budget")
    nurse_cap = scenario.get("nurse_capacity")
    max_mem = scenario.get("max_members")
    prov_cap = scenario.get("provider_capacity")

    if budget is None or not math.isfinite(budget) or budget < 0:
        return _fail("budget must be finite and nonneg")
    if nurse_cap is None or not math.isfinite(nurse_cap) or nurse_cap < 0:
        return _fail("nurse_capacity must be finite and nonneg")
    if max_mem is None or not isinstance(max_mem, int) or max_mem < 0:
        return _fail("max_members must be a nonneg integer")
    if not isinstance(prov_cap, dict) or len(prov_cap) == 0:
        return _fail("provider_capacity must be a non-empty dict")
    for pid, cap in prov_cap.items():
        if not isinstance(cap, int) or cap < 0:
            return _fail(f"provider_capacity[{pid}] must be a nonneg integer")

    ids, provs, benefits, costs, hours = [], [], [], [], []
    seen_ids = set()
    for m in members:
        mid = m.get("member_id")
        pid = m.get("provider_id")
        eb  = m.get("estimated_benefit")
        pc  = m.get("program_cost")
        nh  = m.get("nurse_hours")
        if not mid or mid in seen_ids:
            return _fail(f"Duplicate or missing member_id: {mid}")
        seen_ids.add(mid)
        if not pid:
            return _fail(f"Missing provider_id for {mid}")
        if pid not in prov_cap:
            return _fail(f"No capacity entry for provider {pid}")
        if eb is None or not math.isfinite(eb) or eb < 0:
            return _fail(f"Invalid estimated_benefit for {mid}")
        if pc is None or not math.isfinite(pc) or pc < 0:
            return _fail(f"Invalid program_cost for {mid}")
        if nh is None or not math.isfinite(nh) or nh <= 0:
            return _fail(f"Invalid nurse_hours for {mid} (must be > 0)")
        ids.append(mid); provs.append(pid)
        benefits.append(float(eb)); costs.append(float(pc)); hours.append(float(nh))

    n = len(ids)
    benefits_arr = np.array(benefits)
    costs_arr    = np.array(costs)
    hours_arr    = np.array(hours)

    c_obj = -benefits_arr
    A_rows = [costs_arr, hours_arr, np.ones(n)]
    ub_vals = [float(budget), float(nurse_cap), float(max_mem)]
    unique_provs = sorted(set(provs))
    for pid in unique_provs:
        row = np.array([1.0 if provs[i] == pid else 0.0 for i in range(n)])
        A_rows.append(row)
        ub_vals.append(float(prov_cap[pid]))

    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lb=-np.inf, ub=ub_vals)
    integrality = np.ones(n)
    bounds_obj = Bounds(lb=0, ub=1)

    t0 = _time.perf_counter()
    try:
        result = milp(c_obj, integrality=integrality, bounds=bounds_obj,
                      constraints=constraints, options={"time_limit": 5.0})
    except Exception as ex:
        return {"status": "solver_error", "selected_member_ids": None,
                "selected_count": None, "total_benefit": None,
                "total_cost": None, "total_nurse_hours": None,
                "solve_time_seconds": round(_time.perf_counter() - t0, 4),
                "message": str(ex), "provider_utilization": None}
    solve_sec = round(_time.perf_counter() - t0, 4)

    if result.x is None:
        return {"status": "infeasible", "selected_member_ids": None,
                "selected_count": None, "total_benefit": None,
                "total_cost": None, "total_nurse_hours": None,
                "solve_time_seconds": solve_sec,
                "message": result.message, "provider_utilization": None}

    is_optimal = (result.status == 0)
    status_label = "optimal" if is_optimal else "feasible_time_limited"
    x_raw = result.x
    tol = 1e-6
    x_bin = []
    for v in x_raw:
        if abs(v - round(v)) > tol:
            return {"status": "integrality_failure", "selected_member_ids": None,
                    "selected_count": None, "total_benefit": None,
                    "total_cost": None, "total_nurse_hours": None,
                    "solve_time_seconds": solve_sec,
                    "message": "Solution failed integrality check",
                    "provider_utilization": None}
        x_bin.append(int(round(v)))

    selected = [ids[i] for i in range(n) if x_bin[i] == 1]
    sel_idx  = [i for i in range(n) if x_bin[i] == 1]
    total_benefit = float(sum(benefits_arr[i] for i in sel_idx))
    total_cost    = float(sum(costs_arr[i] for i in sel_idx))
    total_hours   = float(sum(hours_arr[i] for i in sel_idx))

    if total_cost > budget + tol:
        return _fail(f"Post-solve: cost {total_cost} exceeds budget {budget}")
    if total_hours > nurse_cap + tol:
        return _fail(f"Post-solve: hours {total_hours} exceeds nurse_capacity {nurse_cap}")
    if len(selected) > max_mem:
        return _fail(f"Post-solve: {len(selected)} members exceeds max_members {max_mem}")

    prov_util = {}
    for pid in unique_provs:
        cnt = sum(1 for i in sel_idx if provs[i] == pid)
        cap = prov_cap[pid]
        if cnt > cap:
            return _fail(f"Post-solve: provider {pid} has {cnt} > capacity {cap}")
        prov_util[pid] = {"selected_count": cnt, "capacity": cap,
                         "remaining_capacity": cap - cnt}

    return {
        "status": status_label,
        "selected_member_ids": selected,
        "selected_count": len(selected),
        "total_benefit": total_benefit,
        "total_cost": total_cost,
        "total_nurse_hours": total_hours,
        "solve_time_seconds": solve_sec,
        "message": result.message,
        "provider_utilization": prov_util,
    }
''')

with open(MODULE_PATH, "w") as f:
    f.write(module_code)
print(f"Module written to {MODULE_PATH}")

# --- PythonModel wrapper ---
import mlflow

class CareManagementOptimizer(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: each input row is a JSON-encoded optimization scenario."""

    def load_context(self, context):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "optimizer_core",
            os.path.join(context.artifacts["code_dir"], "optimizer_core.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["optimizer_core"] = mod
        spec.loader.exec_module(mod)
        self._optimize = mod.optimize_allocation

    def predict(self, context, model_input, params=None):
        import json, pandas as pd
        results = []
        for _, row in model_input.iterrows():
            scenario = json.loads(row["scenario_json"])
            res = self._optimize(scenario)
            results.append(json.dumps(res))
        return pd.DataFrame({"result_json": results})

print("CareManagementOptimizer PythonModel defined.")

# COMMAND ----------

# DBTITLE 1,Section 7 — Log, Register, and Reload
# ---------- Section 7: Log, Register & Reload ----------
import mlflow, json, pandas as pd
from mlflow.models.signature import infer_signature

# Set UC model registry
mlflow.set_registry_uri("databricks-uc")

# Set experiment
user = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{user}/care_mgmt_optimizer_demo_{TS}"
mlflow.set_experiment(experiment_path)
print(f"MLflow experiment: {experiment_path}")

# Build input/output examples
scenario_json_str = json.dumps(DEFAULT_SCENARIO)
input_example = pd.DataFrame({"scenario_json": [scenario_json_str]})

# Generate actual output for signature
local_result = optimize_allocation(DEFAULT_SCENARIO)
output_example = pd.DataFrame({"result_json": [json.dumps(local_result)]})
signature = infer_signature(input_example, output_example)

# Pip requirements pinned to installed versions
pip_reqs = [
    f"scipy=={PKG_VERSIONS['scipy']}",
    f"pandas=={PKG_VERSIONS['pandas']}",
    f"numpy=={PKG_VERSIONS['numpy']}",
    f"mlflow=={PKG_VERSIONS['mlflow']}",
]
print(f"Pip requirements: {pip_reqs}")

# Log and register
REGISTRATION_OK = False
MODEL_VERSION = None
RUN_ID = None

with mlflow.start_run(run_name="care_mgmt_optimizer") as run:
    RUN_ID = run.info.run_id
    print(f"Run ID: {RUN_ID}")

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CareManagementOptimizer(),
        artifacts={"code_dir": CODE_DIR},
        input_example=input_example,
        signature=signature,
        pip_requirements=pip_reqs,
        registered_model_name=MODEL_NAME if SCHEMA_OK else None,
    )
    print(f"Model logged: {model_info.model_uri}")

# Find exact registered version
if SCHEMA_OK:
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(registry_uri="databricks-uc")
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for v in versions:
            if v.run_id == RUN_ID:
                MODEL_VERSION = int(v.version)
                break
        if MODEL_VERSION:
            REGISTRATION_OK = True
            print(f"Registered: {MODEL_NAME} version {MODEL_VERSION}")
        else:
            print("WARNING: Could not find version matching run ID")
    except Exception as e:
        print(f"Registration lookup failed: {e}")

# Reload and verify
if REGISTRATION_OK:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    reloaded = mlflow.pyfunc.load_model(model_uri)
    print(f"\nReloaded {model_uri}")

    # Single-scenario test
    out1 = reloaded.predict(input_example)
    r1 = json.loads(out1["result_json"].iloc[0])
    assert r1["status"] == "optimal", f"Expected optimal, got {r1['status']}"
    assert abs(r1["total_benefit"] - local_result["total_benefit"]) < 1e-6
    print(f"Single scenario: benefit={r1['total_benefit']}, selected={r1['selected_member_ids']} ✓")

    # Two-row independence test
    sc2 = {**DEFAULT_SCENARIO, "budget": 14}
    input_2row = pd.DataFrame({"scenario_json": [scenario_json_str, json.dumps(sc2)]})
    out2 = reloaded.predict(input_2row)
    r2a = json.loads(out2["result_json"].iloc[0])
    r2b = json.loads(out2["result_json"].iloc[1])
    print(f"Row 0: benefit={r2a['total_benefit']}, selected={r2a['selected_member_ids']}")
    print(f"Row 1: benefit={r2b['total_benefit']}, selected={r2b['selected_member_ids']}")
    assert r2a["total_benefit"] != r2b["total_benefit"] or r2a["selected_member_ids"] != r2b["selected_member_ids"], "Two-row results should differ"
    print("Two-row independence: ✓")
else:
    print("Skipping reload — registration did not succeed.")

# COMMAND ----------

# DBTITLE 1,Serving and invocation note
# MAGIC %md
# MAGIC ## §8–9 — Model Serving & Live Invocation
# MAGIC A scale-to-zero CPU endpoint serves the registered model version. The invocation cell polls for readiness, then sends the default six-member scenario and displays the parsed result with end-to-end latency.

# COMMAND ----------

# DBTITLE 1,Section 8 — New Serving Endpoint
# ---------- Section 8: Serving Endpoint ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput,
)
import time

ENDPOINT_OK = False
ENDPOINT_STATE = "skipped"

if REGISTRATION_OK:
    w = WorkspaceClient()

    # Check if endpoint name already exists; if so, generate a new unique name
    ep_name = ENDPOINT_NAME
    try:
        existing = w.serving_endpoints.get(ep_name)
        # Name collision — append extra suffix
        ep_name = f"{ENDPOINT_NAME}-{int(time.time()) % 10000}"
        print(f"Name collision, using: {ep_name}")
    except Exception:
        pass  # Name is free

    ENDPOINT_NAME = ep_name
    print(f"Creating endpoint: {ENDPOINT_NAME}")
    print(f"  Model: {MODEL_NAME} v{MODEL_VERSION}")

    try:
        endpoint = w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=[
                    ServedEntityInput(
                        entity_name=MODEL_NAME,
                        entity_version=str(MODEL_VERSION),
                        workload_size="Small",
                        scale_to_zero_enabled=True,
                    )
                ]
            ),
        )
        print(f"Endpoint creation submitted.")
    except Exception as e:
        print(f"Endpoint creation failed: {e}")
        ENDPOINT_STATE = f"failed: {e}"

    # Bounded readiness check (60 seconds)
    if "failed" not in ENDPOINT_STATE:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                ep = w.serving_endpoints.get(ENDPOINT_NAME)
                state = ep.state
                ready = getattr(state, "ready", None)
                config_update = getattr(state, "config_update", None)
                print(f"  State: ready={ready}, config_update={config_update}")
                if "READY" in str(ready) and "NOT_READY" not in str(ready):
                    ENDPOINT_OK = True
                    ENDPOINT_STATE = "ready"
                    print(f"Endpoint {ENDPOINT_NAME} is READY.")
                    break
                if "UPDATE_FAILED" in str(config_update):
                    ENDPOINT_STATE = "update_failed"
                    print("Endpoint update failed.")
                    break
            except Exception as e:
                print(f"  Poll error: {e}")
            time.sleep(10)
        else:
            ENDPOINT_STATE = "provisioning"
            print(f"Endpoint still provisioning after 60s. Check later.")
else:
    print("Skipping endpoint — model registration did not succeed.")

print(f"\nEndpoint state: {ENDPOINT_STATE}")

# COMMAND ----------

# DBTITLE 1,Section 9 — Endpoint Invocation
# ---------- Section 9: Endpoint Invocation ----------
# This cell is runnable once the endpoint reaches READY state.

import json, time
from databricks.sdk import WorkspaceClient

def invoke_endpoint(endpoint_name, scenario):
    """Invoke the serving endpoint and return parsed result + latency."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import DataframeSplitInput
    w = WorkspaceClient()
    split_input = DataframeSplitInput(
        columns=["scenario_json"],
        data=[[json.dumps(scenario)]]
    )
    t0 = time.perf_counter()
    response = w.serving_endpoints.query(name=endpoint_name, dataframe_split=split_input)
    latency = round(time.perf_counter() - t0, 3)
    return response, latency

# Poll until endpoint is ready (up to 5 minutes)
w = WorkspaceClient()
ep_ready = False
for _poll in range(20):
    try:
        ep = w.serving_endpoints.get(ENDPOINT_NAME)
        ready_str = str(getattr(ep.state, "ready", None))
        ep_ready = "READY" in ready_str and "NOT_READY" not in ready_str
        if ep_ready:
            ENDPOINT_STATE = "ready"
            ENDPOINT_OK = True
            print(f"✅ Endpoint ready after {_poll * 15}s")
            break
        print(f"  [{_poll+1}/20] {ready_str} — waiting 15s...")
        time.sleep(15)
    except Exception as e:
        print(f"  Poll error: {e}")
        time.sleep(15)
if not ep_ready:
    print(f"Endpoint not ready after 5 min. Re-run this cell later.")

if ep_ready:
    print(f"Invoking {ENDPOINT_NAME}...")
    resp, latency = invoke_endpoint(ENDPOINT_NAME, DEFAULT_SCENARIO)

    # Parse response
    predictions = resp.predictions
    if isinstance(predictions, list) and len(predictions) > 0:
        row = predictions[0]
        result_str = row.get("result_json", row) if isinstance(row, dict) else str(row)
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
    else:
        result = predictions

    print(f"\nEndpoint latency: {latency}s")
    print(f"Solver runtime:   {result.get('solve_time_seconds', '?')}s")
    print(f"Status:           {result.get('status')}")
    print(f"Selected:         {result.get('selected_member_ids')}")
    print(f"Total benefit:    {result.get('total_benefit')}")
    print(f"Total cost:       {result.get('total_cost')}")
    print(f"Nurse hours:      {result.get('total_nurse_hours')}")
    print(f"Provider util:    {json.dumps(result.get('provider_utilization'), indent=2)}")
else:
    print(f"Endpoint {ENDPOINT_NAME} is not yet ready (state: {ENDPOINT_STATE}).")
    print("Re-run this cell once provisioning completes.")
    print("No edits needed — it will auto-detect readiness.")

# COMMAND ----------

# DBTITLE 1,Section 10 — Final Status and Demo Explanation
# ---------- Section 10: Final Status & Demo Explanation ----------
steps = [
    ("Notebook",               "completed", "Care Management Optimizer Demo"),
    ("Catalog",                "completed", CATALOG),
    ("Schema",                 "completed" if SCHEMA_OK else "failed", FQ_SCHEMA),
    ("Synthetic members",      "completed" if SCHEMA_OK else "skipped", MEMBERS_TABLE),
    ("Synthetic providers",    "completed" if SCHEMA_OK else "skipped", PROVIDER_TABLE),
    ("Local optimization",     "completed", "3 scenarios verified against exhaustive enumeration"),
    ("MLflow run",             "completed" if RUN_ID else "failed", RUN_ID or "N/A"),
    ("Registered model",       "completed" if REGISTRATION_OK else "failed",
                               f"{MODEL_NAME} v{MODEL_VERSION}" if REGISTRATION_OK else "N/A"),
    ("Serving endpoint",       ENDPOINT_STATE, ENDPOINT_NAME),
]

print("=" * 70)
print("DEMO STATUS")
print("=" * 70)
for step, status, detail in steps:
    icon = {"completed": "✅", "ready": "✅", "provisioning": "⏳",
            "failed": "❌", "skipped": "⏭️", "update_failed": "❌"}.get(status, "❓")
    print(f"  {icon} {step:24s} [{status:14s}]  {detail}")
print("=" * 70)

print("""
📌 EXPLANATION
HiGHS finds the best feasible member allocation under budget, nurse-hour,
enrollment, and provider-capacity constraints. MLflow packages and versions
the decision logic. Databricks Model Serving exposes it through an API.

⚠️  Serving charges apply while the endpoint exists (even at scale-to-zero,
   cold-start latency is expected on first request after idle period).
""")

# COMMAND ----------

# DBTITLE 1,Visualization — Optimizer vs Greedy
# ---------- Visualization: Optimizer vs Greedy ----------
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"figure.facecolor": "white"})

# Data from Section 5 verification
scenario_labels = ["A: Original\n(budget=10)", "B: More P001 cap\n(P001 cap=2)", "C: More budget\n(budget=14)"]
greedy_benefits  = [14.0, 14.0, 16.0]
optimal_benefits = [18.0, 18.0, 19.0]
improvements = [o - g for o, g in zip(optimal_benefits, greedy_benefits)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

# --- Bar chart: Greedy vs Optimizer ---
ax1 = axes[0]
x = range(len(scenario_labels))
bar_w = 0.35
bars_g = ax1.bar([i - bar_w/2 for i in x], greedy_benefits, bar_w, label="Greedy", color="#B0BEC5", edgecolor="#78909C")
bars_o = ax1.bar([i + bar_w/2 for i in x], optimal_benefits, bar_w, label="MILP Optimizer", color="#1E88E5", edgecolor="#1565C0")

for bar, val in zip(bars_g, greedy_benefits):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.0f}", ha="center", fontsize=10, color="#546E7A")
for bar, val in zip(bars_o, optimal_benefits):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.0f}", ha="center", fontsize=10, fontweight="bold", color="#1565C0")

ax1.set_xticks(list(x))
ax1.set_xticklabels(scenario_labels, fontsize=9)
ax1.set_ylabel("Total Benefit (units)", fontsize=11)
ax1.set_title("Optimizer vs Greedy Baseline — Benefit Comparison", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_ylim(0, max(optimal_benefits) + 4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Improvement chart ---
ax2 = axes[1]
colors = ["#43A047" if imp > 0 else "#E0E0E0" for imp in improvements]
ax2.barh(scenario_labels, improvements, color=colors, edgecolor="#2E7D32", height=0.5)
for i, imp in enumerate(improvements):
    pct = (imp / greedy_benefits[i]) * 100
    ax2.text(imp + 0.1, i, f"+{imp:.0f} ({pct:.0f}%)", va="center", fontsize=10, fontweight="bold", color="#2E7D32")
ax2.set_xlabel("Benefit Improvement", fontsize=11)
ax2.set_title("MILP Uplift over Greedy", fontsize=13, fontweight="bold")
ax2.set_xlim(0, max(improvements) + 2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
print("Optimizer consistently outperforms greedy by 19-29% across all scenarios.")

# COMMAND ----------

# DBTITLE 1,Bonus analyses note
# MAGIC %md
# MAGIC ## Bonus — Deep-Dive Analyses
# MAGIC Why does the optimizer pick M002+M005+M006 over the greedy choice of M001+M006? The member efficiency chart and budget-vs-nurse sensitivity heatmap below explain the decision surface.

# COMMAND ----------

# DBTITLE 1,Member Efficiency Analysis
# ---------- Bonus: Member Efficiency & Selection Explanation ----------
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({"figure.facecolor": "white"})

# Build member-level metrics
members = DEFAULT_SCENARIO["members"]
optimal_ids = set(res["selected_member_ids"])  # from Section 4 smoke test
greedy_ids  = set(greedy_allocation(DEFAULT_SCENARIO)["selected_member_ids"])

mids      = [m["member_id"] for m in members]
benefits  = [m["estimated_benefit"] for m in members]
costs     = [m["program_cost"] for m in members]
hrs       = [m["nurse_hours"] for m in members]
providers = [m["provider_id"] for m in members]
efficiency = [b / c for b, c in zip(benefits, costs)]

# Print table
print(f"{'Member':>8} {'Provider':>10} {'Benefit':>8} {'Cost':>6} {'Hours':>6} {'B/C Ratio':>10} {'Optimizer':>10} {'Greedy':>8}")
print("-" * 78)
for i, mid in enumerate(mids):
    opt_flag = "✅" if mid in optimal_ids else "  "
    grd_flag = "✅" if mid in greedy_ids else "  "
    print(f"{mid:>8} {providers[i]:>10} {benefits[i]:>8} {costs[i]:>6} {hrs[i]:>6} {efficiency[i]:>10.2f} {opt_flag:>10} {grd_flag:>8}")

print(f"\n💡 Greedy picks M001 (highest benefit=10) but it costs 8 — using 80% of the budget on one member.")
print(f"   The optimizer skips M001 and picks M002+M005+M006 (benefit=9+5+4=18) for cost 5+3+2=10.")
print(f"   Same budget, +29% benefit. The key insight: benefit/cost ratio matters more than raw benefit.")

# --- Scatter: Benefit vs Cost with selection overlay ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Color by selection status
for i, mid in enumerate(mids):
    in_opt = mid in optimal_ids
    in_grd = mid in greedy_ids
    if in_opt and in_grd:
        color, marker, label = "#1E88E5", "*", "Both"
    elif in_opt:
        color, marker, label = "#43A047", "o", "Optimizer only"
    elif in_grd:
        color, marker, label = "#E53935", "s", "Greedy only"
    else:
        color, marker, label = "#BDBDBD", "x", "Neither"
    ax1.scatter(costs[i], benefits[i], c=color, marker=marker, s=200, edgecolors="black", linewidth=0.5, zorder=3)
    ax1.annotate(f" {mid}\n({providers[i]})", (costs[i], benefits[i]), fontsize=8, fontweight="bold")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#1E88E5", markersize=12, label="Both select"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#43A047", markersize=10, label="Optimizer only"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#E53935", markersize=10, label="Greedy only"),
    Line2D([0],[0], marker="x", color="w", markerfacecolor="#BDBDBD", markersize=10, markeredgecolor="#BDBDBD", label="Neither"),
]
ax1.legend(handles=legend_elements, fontsize=9, loc="upper left")
ax1.set_xlabel("Program Cost", fontsize=11)
ax1.set_ylabel("Estimated Benefit", fontsize=11)
ax1.set_title("Member Selection: Benefit vs Cost", fontsize=13, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)

# --- Bar chart: Benefit/Cost efficiency ---
colors = ["#43A047" if mid in optimal_ids else "#B0BEC5" for mid in mids]
ax2.bar(mids, efficiency, color=colors, edgecolor=["#2E7D32" if mid in optimal_ids else "#78909C" for mid in mids])
for i, (mid, eff) in enumerate(zip(mids, efficiency)):
    ax2.text(i, eff + 0.03, f"{eff:.2f}", ha="center", fontsize=9, fontweight="bold",
             color="#2E7D32" if mid in optimal_ids else "#546E7A")
ax2.set_ylabel("Benefit / Cost Ratio", fontsize=11)
ax2.set_title("Cost Efficiency by Member", fontsize=13, fontweight="bold")
ax2.axhline(y=np.mean(efficiency), color="#FF9800", linestyle="--", linewidth=1, label=f"Mean={np.mean(efficiency):.2f}")
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Sensitivity Analysis — Budget × Nurse Capacity Heatmap
# ---------- Bonus: Sensitivity Analysis ----------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
matplotlib.rcParams.update({"figure.facecolor": "white"})

# Sweep budget and nurse capacity
budgets = list(range(2, 22, 2))       # 2, 4, 6, ..., 20
nurse_caps = list(range(2, 14, 2))    # 2, 4, 6, ..., 12

benefit_grid = np.zeros((len(nurse_caps), len(budgets)))
count_grid   = np.zeros((len(nurse_caps), len(budgets)))

for i, nc in enumerate(nurse_caps):
    for j, b in enumerate(budgets):
        sc = copy.deepcopy(DEFAULT_SCENARIO)
        sc["budget"] = b
        sc["nurse_capacity"] = nc
        sc["max_members"] = 6  # unconstrain enrollment for this sweep
        r = optimize_allocation(sc)
        benefit_grid[i, j] = r["total_benefit"] if r["total_benefit"] else 0
        count_grid[i, j]   = r["selected_count"] if r["selected_count"] else 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# --- Benefit heatmap ---
im1 = ax1.imshow(benefit_grid, cmap="YlGnBu", aspect="auto", origin="lower")
ax1.set_xticks(range(len(budgets)))
ax1.set_xticklabels(budgets)
ax1.set_yticks(range(len(nurse_caps)))
ax1.set_yticklabels(nurse_caps)
ax1.set_xlabel("Budget", fontsize=11)
ax1.set_ylabel("Nurse Capacity (hours)", fontsize=11)
ax1.set_title("Optimal Total Benefit", fontsize=13, fontweight="bold")
for i in range(len(nurse_caps)):
    for j in range(len(budgets)):
        ax1.text(j, i, f"{benefit_grid[i,j]:.0f}", ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="white" if benefit_grid[i,j] > benefit_grid.max()*0.6 else "black")
fig.colorbar(im1, ax=ax1, shrink=0.8, label="Benefit")

# --- Members selected heatmap ---
im2 = ax2.imshow(count_grid, cmap="OrRd", aspect="auto", origin="lower")
ax2.set_xticks(range(len(budgets)))
ax2.set_xticklabels(budgets)
ax2.set_yticks(range(len(nurse_caps)))
ax2.set_yticklabels(nurse_caps)
ax2.set_xlabel("Budget", fontsize=11)
ax2.set_ylabel("Nurse Capacity (hours)", fontsize=11)
ax2.set_title("Members Selected", fontsize=13, fontweight="bold")
for i in range(len(nurse_caps)):
    for j in range(len(budgets)):
        ax2.text(j, i, f"{int(count_grid[i,j])}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if count_grid[i,j] > count_grid.max()*0.6 else "black")
fig.colorbar(im2, ax=ax2, shrink=0.8, label="Count")

plt.suptitle("Sensitivity: Budget × Nurse Capacity (provider caps unchanged)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Summary insights
print("Key insights:")
print(f"  Max achievable benefit: {benefit_grid.max():.0f} (all 6 members enrolled)")
print(f"  With original constraints (budget=10, nurses=6): benefit={benefit_grid[2, 4]:.0f}")
print(f"  Budget is binding when nurse hours are abundant (bottom-right plateau)")
print(f"  Nurse capacity is binding when budget is abundant (top-left plateau)")

# COMMAND ----------

# DBTITLE 1,Shadow Prices — Marginal Value of Each Constraint
# ---------- Bonus: Shadow Prices (Marginal Value Analysis) ----------
# "If we had 1 more unit of X, how much additional benefit could we capture?"
# Approximated by re-solving with each constraint relaxed incrementally.
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"figure.facecolor": "white"})

base = optimize_allocation(DEFAULT_SCENARIO)
base_benefit = base["total_benefit"]
print(f"Baseline: benefit={base_benefit}, cost={base['total_cost']}, hours={base['total_nurse_hours']}")
print(f"Constraints: budget={DEFAULT_SCENARIO['budget']}, nurse_cap={DEFAULT_SCENARIO['nurse_capacity']}, max_members={DEFAULT_SCENARIO['max_members']}")
print(f"Provider caps: {DEFAULT_SCENARIO['provider_capacity']}\n")

relaxations = [
    ("Budget +1",        {"budget": DEFAULT_SCENARIO["budget"] + 1}),
    ("Budget +2",        {"budget": DEFAULT_SCENARIO["budget"] + 2}),
    ("Budget +5",        {"budget": DEFAULT_SCENARIO["budget"] + 5}),
    ("Nurse hours +1",   {"nurse_capacity": DEFAULT_SCENARIO["nurse_capacity"] + 1}),
    ("Nurse hours +2",   {"nurse_capacity": DEFAULT_SCENARIO["nurse_capacity"] + 2}),
    ("Max members +1",   {"max_members": DEFAULT_SCENARIO["max_members"] + 1}),
    ("P001 cap +1",      {"provider_capacity": {"P001": 2, "P002": 2}}),
    ("P002 cap +1",      {"provider_capacity": {"P001": 1, "P002": 3}}),
]

print(f"{'Relaxation':<20} {'New Benefit':>12} {'Marginal Gain':>14} {'New Selection'}")
print("-" * 80)
labels, gains = [], []
for label, overrides in relaxations:
    sc = copy.deepcopy(DEFAULT_SCENARIO)
    sc.update(overrides)
    r = optimize_allocation(sc)
    gain = r["total_benefit"] - base_benefit
    sel = ", ".join(r["selected_member_ids"]) if r["selected_member_ids"] else "(none)"
    print(f"{label:<20} {r['total_benefit']:>12.1f} {'+' + str(gain) if gain >= 0 else str(gain):>14} {sel}")
    labels.append(label)
    gains.append(gain)

# --- Waterfall chart ---
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#43A047" if g > 0 else "#E0E0E0" for g in gains]
bars = ax.barh(labels[::-1], gains[::-1], color=colors[::-1],
               edgecolor=["#2E7D32" if g > 0 else "#9E9E9E" for g in gains][::-1], height=0.6)
for i, (lbl, g) in enumerate(zip(labels[::-1], gains[::-1])):
    if g > 0:
        ax.text(g + 0.1, i, f"+{g:.1f}", va="center", fontsize=10, fontweight="bold", color="#2E7D32")
    else:
        ax.text(0.1, i, f"{g:.1f}", va="center", fontsize=10, color="#757575")
ax.set_xlabel("Marginal Benefit Gain", fontsize=11)
ax.set_title("Shadow Prices: Benefit Gained per Constraint Relaxation", fontsize=13, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# Interpretation
binding = [l for l, g in zip(labels, gains) if g > 0]
non_binding = [l for l, g in zip(labels, gains) if g == 0]
print(f"\n\U0001f4ca Binding constraints (relaxing improves benefit): {', '.join(binding) if binding else 'none'}")
print(f"\U0001f7e2 Non-binding constraints (already slack): {', '.join(non_binding) if non_binding else 'none'}")
print(f"\n\U0001f4a1 Resource allocation recommendation:")
if binding:
    best_label, best_gain = max(zip(labels, gains), key=lambda x: x[1])
    print(f"   Highest ROI: '{best_label}' yields +{best_gain:.1f} benefit per unit relaxed.")
    print(f"   Invest here first before relaxing other constraints.")

# COMMAND ----------

# DBTITLE 1,Persist Results to Unity Catalog
# ---------- Bonus: Persist Analytical Results to UC ----------
# Save key outputs as queryable tables for dashboards and downstream SQL consumers.
import pandas as pd, copy, numpy as np
from datetime import datetime, timezone

run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# --- 1. Scenario comparison table ---
comp_rows = []
for label, sc in scenarios.items():
    g = greedy_allocation(sc)
    o = optimize_allocation(sc)
    for method, r in [("greedy", g), ("optimizer", o)]:
        comp_rows.append({
            "run_timestamp": run_ts,
            "scenario": label,
            "method": method,
            "status": r["status"],
            "selected_members": ",".join(r["selected_member_ids"]) if r["selected_member_ids"] else None,
            "selected_count": r["selected_count"],
            "total_benefit": r["total_benefit"],
            "total_cost": r["total_cost"],
            "total_nurse_hours": r["total_nurse_hours"],
            "solve_seconds": r["solve_time_seconds"],
        })
df_comp = pd.DataFrame(comp_rows)
COMP_TABLE = f"{FQ_SCHEMA}.scenario_comparison"
spark.createDataFrame(df_comp).write.mode("overwrite").saveAsTable(COMP_TABLE)
print(f"Saved: {COMP_TABLE} ({len(comp_rows)} rows)")

# --- 2. Sensitivity grid table ---
budgets_list = list(range(2, 22, 2))
nurse_list   = list(range(2, 14, 2))
sens_rows = []
for nc in nurse_list:
    for b in budgets_list:
        sc = copy.deepcopy(DEFAULT_SCENARIO)
        sc["budget"] = b
        sc["nurse_capacity"] = nc
        sc["max_members"] = 6
        r = optimize_allocation(sc)
        sens_rows.append({
            "run_timestamp": run_ts,
            "budget": b,
            "nurse_capacity": nc,
            "optimal_benefit": r["total_benefit"] if r["total_benefit"] else 0.0,
            "members_selected": r["selected_count"] if r["selected_count"] else 0,
            "selected_ids": ",".join(r["selected_member_ids"]) if r["selected_member_ids"] else None,
            "status": r["status"],
        })
df_sens = pd.DataFrame(sens_rows)
SENS_TABLE = f"{FQ_SCHEMA}.sensitivity_grid"
spark.createDataFrame(df_sens).write.mode("overwrite").saveAsTable(SENS_TABLE)
print(f"Saved: {SENS_TABLE} ({len(sens_rows)} rows)")

# --- 3. Shadow price table ---
sp_rows = []
for label, overrides in relaxations:
    sc = copy.deepcopy(DEFAULT_SCENARIO)
    sc.update(overrides)
    r = optimize_allocation(sc)
    sp_rows.append({
        "run_timestamp": run_ts,
        "relaxation": label,
        "new_benefit": r["total_benefit"],
        "marginal_gain": r["total_benefit"] - base_benefit,
        "new_selection": ",".join(r["selected_member_ids"]) if r["selected_member_ids"] else None,
        "is_binding": r["total_benefit"] > base_benefit,
    })
df_sp = pd.DataFrame(sp_rows)
SP_TABLE = f"{FQ_SCHEMA}.shadow_prices"
spark.createDataFrame(df_sp).write.mode("overwrite").saveAsTable(SP_TABLE)
print(f"Saved: {SP_TABLE} ({len(sp_rows)} rows)")

print(f"\n\u2705 All analytical results persisted to {FQ_SCHEMA}")
print(f"   Tables: scenario_comparison, sensitivity_grid, shadow_prices")
print(f"   Query with: SELECT * FROM {COMP_TABLE}")

# COMMAND ----------

# DBTITLE 1,Cleanup (optional — do NOT run automatically)
# ---------- Cleanup: Remove ONLY resources created by this demo ----------
# ⚠️  Uncomment and run manually when you are done with the demo.

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()

# # 1. Delete serving endpoints (both original and re-run)
# for ep_name in ["solver-demo-20261007_135926", "solver-demo-20261007_135926-8749"]:
#     try:
#         w.serving_endpoints.delete(ep_name)
#         print(f"Deleted endpoint: {ep_name}")
#     except Exception as e:
#         print(f"Endpoint delete ({ep_name}): {e}")

# # 2. Delete registered model (removes all versions)
# from mlflow.tracking import MlflowClient
# mlflow.set_registry_uri("databricks-uc")
# try:
#     mc = MlflowClient(registry_uri="databricks-uc")
#     mc.delete_registered_model(MODEL_NAME)
#     print(f"Deleted model: {MODEL_NAME}")
# except Exception as e:
#     print(f"Model delete: {e}")

# # 3. Drop schema (cascade drops all 5 tables: synthetic_members, synthetic_provider_capacity,
# #    scenario_comparison, sensitivity_grid, shadow_prices)
# try:
#     spark.sql(f"DROP SCHEMA IF EXISTS `{CATALOG}`.`{SCHEMA}` CASCADE")
#     print(f"Dropped schema: {FQ_SCHEMA}")
# except Exception as e:
#     print(f"Schema drop: {e}")

# print("Cleanup complete.")