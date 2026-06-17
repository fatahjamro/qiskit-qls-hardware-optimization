import json
import math
import sys
import os
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Add script directory to sys.path to resolve imports when running from root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from qls_circuits import (
    IDEAL_DISTRIBUTIONS, TWIN_PAIRS, CELL_NAMES
)

def calculate_probabilities(counts):
    total = sum(counts.values())
    if total == 0:
        return {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}
    return {state: counts.get(state, 0) / total for state in ["00", "01", "10", "11"]}

def calculate_tvd(p, q):
    return 0.5 * sum(abs(p.get(s, 0.0) - q.get(s, 0.0)) for s in ["00", "01", "10", "11"])

def analyze_record(rec):
    raw_counts = rec["raw_counts_16_cells"]
    # Sort counts by cell_index to make sure ordering is correct
    raw_counts_sorted = sorted(raw_counts, key=lambda x: x["cell_index"])
    
    probs_list = [calculate_probabilities(item["counts"]) for item in raw_counts_sorted]
    
    # Calculate cell TVDs
    cell_tvds = []
    for idx, name in enumerate(CELL_NAMES):
        p = probs_list[idx]
        q = IDEAL_DISTRIBUTIONS[name]
        cell_tvds.append(calculate_tvd(p, q))
        
    # Calculate twin TVDs
    twin_tvds = {}
    twin_values = []
    for state, (idx1, idx2) in TWIN_PAIRS.items():
        p1 = probs_list[idx1]
        p2 = probs_list[idx2]
        val = calculate_tvd(p1, p2)
        twin_tvds[state] = val
        twin_values.append(val)
        
    # Calculate control TVDs
    control_values = []
    for i in range(16):
        for j in range(i + 1, 16):
            if CELL_NAMES[i] != CELL_NAMES[j]:
                p_i = probs_list[i]
                p_j = probs_list[j]
                control_values.append(calculate_tvd(p_i, p_j))
                
    return {
        "avg_cell_tvd": sum(cell_tvds) / len(cell_tvds),
        "avg_twin_tvd": sum(twin_values) / len(twin_values),
        "avg_control_tvd": sum(control_values) / len(control_values),
        "twin_tvds": twin_tvds,
        "cell_tvds": cell_tvds
    }

def main():
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    results_path = os.path.join("data", "raw", "qls_qpu_benchmark_results.json")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run benchmark first!")
        sys.exit(1)
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    records = data.get("records", [])
    if not records:
        print("Error: No records found in the results file.")
        sys.exit(1)
        
    # Output sanity check
    print(f"Loaded {len(records)} records from {results_path}.")
    record_ids = [r["record_id"] for r in records]
    if len(record_ids) != len(set(record_ids)):
        print("Warning: Duplicate record_ids detected!")
        
    rows = []
    for rec in records:
        # TVD analysis
        analysis = analyze_record(rec)
        
        # Build features dict
        cal = rec["calibration_features"]
        trans = rec["transpilation_summary"]
        pair = rec["physical_pair"]
        
        row = {
            "record_id": rec["record_id"],
            "pair_type": rec["pair_type"],
            "pair_type_encoded": 1 if rec["pair_type"] == "adjacent" else 0,
            "u": pair[0],
            "v": pair[1],
            "group_id": f"{rec['pair_type']}_{min(pair)}_{max(pair)}",
            "direct_edge": 1 if rec["direct_edge"] else 0,
            "routing_distance": rec["routing_distance"],
            
            # Calibration features
            "T1_u": cal["T1_u"],
            "T1_v": cal["T1_v"],
            "T2_u": cal["T2_u"],
            "T2_v": cal["T2_v"],
            "readout_error_u": cal["readout_error_u"],
            "readout_error_v": cal["readout_error_v"],
            "direct_two_qubit_error": cal.get("direct_two_qubit_error") if cal.get("direct_two_qubit_error") is not None else np.nan,
            "mean_path_cz_error": cal.get("mean_path_cz_error") if cal.get("mean_path_cz_error") is not None else np.nan,
            "max_path_cz_error": cal.get("max_path_cz_error") if cal.get("max_path_cz_error") is not None else np.nan,
            "sum_path_cz_error": cal.get("sum_path_cz_error") if cal.get("sum_path_cz_error") is not None else np.nan,
            "heuristic_cost": cal.get("heuristic_cost") if cal.get("heuristic_cost") is not None else np.nan,
            
            # Transpilation features
            "mean_depth": trans["mean_depth"],
            "max_depth": trans["max_depth"],
            "mean_two_qubit_gate_count": trans["mean_two_qubit_gate_count"],
            "max_two_qubit_gate_count": trans["max_two_qubit_gate_count"],
            "mean_swap_count": trans["mean_swap_count"],
            "max_swap_count": trans["max_swap_count"],
            "total_two_qubit_gate_count": trans["total_two_qubit_gate_count"],
            "total_swap_count": trans["total_swap_count"],
            
            # Mitigation features
            "M_DD": rec["M_DD"],
            "M_twirling": rec["M_twirling"],
            "optimization_level": rec["optimization_level"],
            
            # Targets and labels
            "avg_twin_tvd": analysis["avg_twin_tvd"],
            "avg_cell_tvd": analysis["avg_cell_tvd"],
            "avg_control_tvd": analysis["avg_control_tvd"]
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("data", "processed", "cleaned_dataset.csv"), index=False)
    print("Cleaned dataset saved to data/processed/cleaned_dataset.csv")
    
    # Generate repeats summary grouping
    summary_df = df.groupby(["pair_type", "u", "v", "mitigation_setting" if "mitigation_setting" in df.columns else "M_DD"]).agg(
        mean_avg_twin_tvd=("avg_twin_tvd", "mean"),
        std_avg_twin_tvd=("avg_twin_tvd", "std"),
        mean_avg_cell_tvd=("avg_cell_tvd", "mean"),
        std_avg_cell_tvd=("avg_cell_tvd", "std"),
        mean_avg_control_tvd=("avg_control_tvd", "mean"),
        std_avg_control_tvd=("avg_control_tvd", "std"),
        n_repeats=("avg_twin_tvd", "count")
    ).reset_index()
    summary_df.to_csv(os.path.join("data", "processed", "tvd_summary_by_mapping.csv"), index=False)
    print("Mapping summary saved to data/processed/tvd_summary_by_mapping.csv")
    
    # ML Setup: Define features
    feature_cols = [
        "pair_type_encoded", "direct_edge", "routing_distance",
        "T1_u", "T1_v", "T2_u", "T2_v", "readout_error_u", "readout_error_v",
        "direct_two_qubit_error", "mean_path_cz_error", "max_path_cz_error", "sum_path_cz_error",
        "heuristic_cost", "mean_depth", "max_depth", "mean_two_qubit_gate_count", "max_two_qubit_gate_count",
        "mean_swap_count", "max_swap_count", "total_two_qubit_gate_count", "total_swap_count",
        "M_DD", "M_twirling", "optimization_level"
    ]
    
    # Verify features presence and missing targets
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: feature column {col} missing from dataset!")
            df[col] = 0
            
    X = df[feature_cols].copy()
    y = df["avg_twin_tvd"].copy()
    groups = df["group_id"].values
    
    logo = LeaveOneGroupOut()
    
    # Initialize OOF array predictions
    oof_ridge = np.zeros(len(df))
    oof_rf = np.zeros(len(df))
    oof_mean = np.zeros(len(df))
    
    fold_maes_ridge = []
    fold_maes_rf = []
    fold_maes_mean = []
    
    # Leave One Group Out loop
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Pipelines with Imputer
        ridge_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        rf_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
        ])
        
        # Train models
        ridge_pipe.fit(X_train, y_train)
        rf_pipe.fit(X_train, y_train)
        
        # Predict
        pred_ridge = ridge_pipe.predict(X_test)
        pred_rf = rf_pipe.predict(X_test)
        # Mean baseline baseline computed inside fold
        pred_mean = np.full(len(y_test), y_train.mean())
        
        oof_ridge[test_idx] = pred_ridge
        oof_rf[test_idx] = pred_rf
        oof_mean[test_idx] = pred_mean
        
        fold_maes_ridge.append(np.mean(np.abs(pred_ridge - y_test)))
        fold_maes_rf.append(np.mean(np.abs(pred_rf - y_test)))
        fold_maes_mean.append(np.mean(np.abs(pred_mean - y_test)))
        
    mae_ridge = np.mean(np.abs(oof_ridge - y))
    mae_rf = np.mean(np.abs(oof_rf - y))
    mae_mean = np.mean(np.abs(oof_mean - y))
    
    print("\nLeave-One-Group-Out Cross Validation Results (MAE):")
    print(f"  - Mean-Predictor Baseline: {mae_mean:.4f} ± {np.std(fold_maes_mean):.4f}")
    print(f"  - Ridge Regression:         {mae_ridge:.4f} ± {np.std(fold_maes_ridge):.4f}")
    print(f"  - Random Forest Regressor:  {mae_rf:.4f} ± {np.std(fold_maes_rf):.4f}")
    
    # Store predictions in DataFrame
    df["pred_tvd_ridge"] = oof_ridge
    df["pred_tvd_rf"] = oof_rf
    
    # Selector comparison
    # Aggregated predictions by mapping configuration group
    agg_df = df.groupby(["group_id", "M_DD", "M_twirling"]).agg(
        measured_tvd=("avg_twin_tvd", "mean"),
        measured_cell_tvd=("avg_cell_tvd", "mean"),
        measured_control_tvd=("avg_control_tvd", "mean"),
        pred_tvd_ridge=("pred_tvd_ridge", "mean"),
        pred_tvd_rf=("pred_tvd_rf", "mean"),
        heuristic_cost=("heuristic_cost", "first"),
        pair_type=("pair_type", "first")
    ).reset_index()
    
    # 1. Regression Selector (Ridge)
    best_ridge_row = agg_df.loc[agg_df["pred_tvd_ridge"].idxmin()]
    
    # 2. Heuristic Selector
    # Filter adjacent only for heuristic cost, and pick lowest cost
    adjacent_agg = agg_df[agg_df["pair_type"] == "adjacent"]
    best_heur_row = adjacent_agg.loc[adjacent_agg["heuristic_cost"].idxmin()]
    
    # 3. Random adjacent baseline
    rand_adj_measured = adjacent_agg["measured_tvd"].mean()
    
    # 4. Routed baseline
    routed_agg = agg_df[agg_df["pair_type"] == "routed"]
    routed_measured = routed_agg["measured_tvd"].mean()
    
    print("\nSelector Comparison (Measured Average Twin TVD on Hardware):")
    print(f"  - Regression-Guided (Ridge):  {best_ridge_row['measured_tvd']:.4f} (Mapping: {best_ridge_row['group_id']}, Mitigation: {'active' if best_ridge_row['M_DD'] else 'baseline'})")
    print(f"  - Calibration Cost Heuristic: {best_heur_row['measured_tvd']:.4f} (Mapping: {best_heur_row['group_id']}, Mitigation: {'active' if best_heur_row['M_DD'] else 'baseline'})")
    print(f"  - Random Adjacent Baseline:   {rand_adj_measured:.4f}")
    print(f"  - Routed-Stress Baseline:     {routed_measured:.4f}")
    
    # Train RF on full dataset to get feature importances
    rf_final = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
    ])
    rf_final.fit(X, y)
    importances = rf_final.named_steps['model'].feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    
    print("\nRandom Forest Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"  - {name:<30}: {imp:.4f}")
        
    # Save JSON summary
    results_json = {
        "cv_mae": {
            "mean_baseline": {"mean": mae_mean, "std": np.std(fold_maes_mean)},
            "ridge": {"mean": mae_ridge, "std": np.std(fold_maes_ridge)},
            "random_forest": {"mean": mae_rf, "std": np.std(fold_maes_rf)}
        },
        "selected_mappings": {
            "regression_ridge": {
                "group_id": best_ridge_row["group_id"],
                "mitigation": "active" if best_ridge_row["M_DD"] else "baseline",
                "measured_twin_tvd": best_ridge_row["measured_tvd"],
                "measured_cell_tvd": best_ridge_row["measured_cell_tvd"],
                "measured_control_tvd": best_ridge_row["measured_control_tvd"]
            },
            "heuristic_calibration": {
                "group_id": best_heur_row["group_id"],
                "mitigation": "active" if best_heur_row["M_DD"] else "baseline",
                "measured_twin_tvd": best_heur_row["measured_tvd"],
                "measured_cell_tvd": best_heur_row["measured_cell_tvd"],
                "measured_control_tvd": best_heur_row["measured_control_tvd"]
            }
        },
        "feature_importances": {name: float(imp) for name, imp in feat_imp}
    }
    with open(os.path.join("results", "regression_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)
    print("\nRegression results saved to results/regression_results.json")
    
    # Generate Plots if Matplotlib is available
    if HAS_MATPLOTLIB:
        # 1. Predicted vs Measured
        plt.figure(figsize=(8, 6))
        plt.scatter(df["avg_twin_tvd"], df["pred_tvd_ridge"], color="#3498db", alpha=0.6, label="Ridge predictions")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Ideal")
        plt.xlabel("Measured Average Twin TVD")
        plt.ylabel("Predicted Average Twin TVD (Out-of-Fold)")
        plt.title("QLS-TVD Predictor Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "predicted_vs_measured_twin_tvd.png"), dpi=300)
        plt.close()
        
        # 2. Selector Comparison Plot
        plt.figure(figsize=(8, 6))
        selectors = ["Regression-Guided", "Calibration Heuristic", "Random Adjacent", "Routed Baseline"]
        measured_vals = [
            best_ridge_row['measured_tvd'],
            best_heur_row['measured_tvd'],
            rand_adj_measured,
            routed_measured
        ]
        colors = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c"]
        plt.bar(selectors, measured_vals, color=colors, width=0.5)
        plt.axhline(y=0.0079, color="gray", linestyle="--", label="Shot Noise Limit (~0.008)")
        plt.ylabel("Measured Average Twin TVD")
        plt.title("Selector Mapping Quality Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "selector_comparison_twin_tvd.png"), dpi=300)
        plt.close()
        
        # 3. Feature Importance Plot
        plt.figure(figsize=(10, 6))
        top_feats = feat_imp[:12]
        names = [f[0] for f in top_feats]
        imps = [f[1] for f in top_feats]
        plt.barh(names[::-1], imps[::-1], color="#9b59b6")
        plt.xlabel("Random Forest Feature Importance")
        plt.title("Top QLS Twin-Symmetry Predictors")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "feature_importance.png"), dpi=300)
        plt.close()
        print("Plots saved successfully.")
        
if __name__ == "__main__":
    main()
