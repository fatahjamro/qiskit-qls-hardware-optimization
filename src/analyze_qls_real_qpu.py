import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Import helpers from analyze_qls_benchmark
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_qls_benchmark import (
    analyze_record
)

def main():
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    results_path = os.path.join("data", "raw", "qls_qpu_benchmark_results.json")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        sys.exit(1)
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    # Filter records to only include real QPU (ibm_fez)
    all_records = data.get("records", [])
    records = [r for r in all_records if r.get("backend") == "ibm_fez"]
    
    print(f"Loaded {len(records)} real QPU records from {results_path} (filtered from {len(all_records)} total).")
    if not records:
        print("Error: No real QPU records found.")
        sys.exit(1)
        
    rows = []
    for rec in records:
        analysis = analyze_record(rec)
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
    df.to_csv(os.path.join("data", "processed", "cleaned_dataset_real_qpu_only.csv"), index=False)
    print("Saved QPU-only cleaned dataset to data/processed/cleaned_dataset_real_qpu_only.csv")
    
    summary_df = df.groupby(["pair_type", "u", "v", "M_DD"]).agg(
        mean_avg_twin_tvd=("avg_twin_tvd", "mean"),
        std_avg_twin_tvd=("avg_twin_tvd", "std"),
        mean_avg_cell_tvd=("avg_cell_tvd", "mean"),
        std_avg_cell_tvd=("avg_cell_tvd", "std"),
        mean_avg_control_tvd=("avg_control_tvd", "mean"),
        std_avg_control_tvd=("avg_control_tvd", "std"),
        n_repeats=("avg_twin_tvd", "count")
    ).reset_index()
    summary_df.to_csv(os.path.join("data", "processed", "tvd_summary_real_qpu_only.csv"), index=False)
    print("Saved QPU-only summary to data/processed/tvd_summary_real_qpu_only.csv")
    
    feature_cols = [
        "pair_type_encoded", "direct_edge", "routing_distance",
        "T1_u", "T1_v", "T2_u", "T2_v", "readout_error_u", "readout_error_v",
        "direct_two_qubit_error", "mean_path_cz_error", "max_path_cz_error", "sum_path_cz_error",
        "heuristic_cost", "mean_depth", "max_depth", "mean_two_qubit_gate_count", "max_two_qubit_gate_count",
        "mean_swap_count", "max_swap_count", "total_two_qubit_gate_count", "total_swap_count",
        "M_DD", "M_twirling", "optimization_level"
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols].copy()
    y = df["avg_twin_tvd"].copy()
    groups = df["group_id"].values
    
    # Run Leave-One-Group-Out CV
    logo = LeaveOneGroupOut()
    
    oof_ridge = np.zeros(len(df))
    oof_rf = np.zeros(len(df))
    oof_mean = np.zeros(len(df))
    
    fold_maes_ridge = []
    fold_maes_rf = []
    fold_maes_mean = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        ridge_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        rf_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
        ])
        
        ridge_pipe.fit(X_train, y_train)
        rf_pipe.fit(X_train, y_train)
        
        pred_ridge = ridge_pipe.predict(X_test)
        pred_rf = rf_pipe.predict(X_test)
        
        oof_ridge[test_idx] = pred_ridge
        oof_rf[test_idx] = pred_rf
        
        mean_val = y_train.mean()
        pred_mean = np.full(len(y_test), mean_val)
        oof_mean[test_idx] = pred_mean
        
        fold_maes_ridge.append(np.mean(np.abs(pred_ridge - y_test)))
        fold_maes_rf.append(np.mean(np.abs(pred_rf - y_test)))
        fold_maes_mean.append(np.mean(np.abs(pred_mean - y_test)))
        
    df["pred_tvd_ridge"] = oof_ridge
    df["pred_tvd_rf"] = oof_rf
    
    mae_ridge = np.mean(fold_maes_ridge)
    mae_rf = np.mean(fold_maes_rf)
    mae_mean = np.mean(fold_maes_mean)
    
    print("\nQPU-Only Leave-One-Group-Out CV MAE:")
    print(f"  - Mean-Predictor Baseline: {mae_mean:.6f} +/- {np.std(fold_maes_mean):.6f}")
    print(f"  - Ridge Regression:        {mae_ridge:.6f} +/- {np.std(fold_maes_ridge):.6f}")
    print(f"  - Random Forest Regressor: {mae_rf:.6f} +/- {np.std(fold_maes_rf):.6f}")
    
    # Find best mappings on QPU
    best_ridge_idx = df["pred_tvd_ridge"].idxmin()
    best_ridge_row = df.loc[best_ridge_idx]
    
    # Calibration heuristic row (min heuristic cost among real QPU adjacent records)
    adj_df = df[df["pair_type"] == "adjacent"].copy()
    if not adj_df.empty:
        best_heur_row = adj_df.loc[adj_df["heuristic_cost"].idxmin()]
    else:
        best_heur_row = df.loc[df["heuristic_cost"].idxmin()]
        
    # Baseline comparison values
    rand_adj_df = df[(df["pair_type"] == "adjacent") & (df["M_DD"] == 0)]
    rand_adj_measured = rand_adj_df["avg_twin_tvd"].mean() if not rand_adj_df.empty else np.nan
    
    routed_df = df[df["pair_type"] == "routed"]
    routed_measured = routed_df["avg_twin_tvd"].mean() if not routed_df.empty else np.nan
    
    print("\nQPU-Only Selector Comparison (Measured Average Twin TVD on Hardware):")
    print(f"  - Regression-Guided (Ridge):  {best_ridge_row['avg_twin_tvd']:.4f} (Mapping: {best_ridge_row['group_id']}, Mitigation: {'active' if best_ridge_row['M_DD'] else 'baseline'})")
    print(f"  - Calibration Cost Heuristic: {best_heur_row['avg_twin_tvd']:.4f} (Mapping: {best_heur_row['group_id']}, Mitigation: {'active' if best_heur_row['M_DD'] else 'baseline'})")
    print(f"  - Random Adjacent Baseline:   {rand_adj_measured:.4f}")
    print(f"  - Routed-Stress Baseline:     {routed_measured:.4f}")
    
    rf_final = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
    ])
    rf_final.fit(X, y)
    importances = rf_final.named_steps['model'].feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    
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
                "measured_twin_tvd": float(best_ridge_row["avg_twin_tvd"]),
                "measured_cell_tvd": float(best_ridge_row["avg_cell_tvd"]),
                "measured_control_tvd": float(best_ridge_row["avg_control_tvd"])
            },
            "heuristic_calibration": {
                "group_id": best_heur_row["group_id"],
                "mitigation": "active" if best_heur_row["M_DD"] else "baseline",
                "measured_twin_tvd": float(best_heur_row["avg_twin_tvd"]),
                "measured_cell_tvd": float(best_heur_row["avg_cell_tvd"]),
                "measured_control_tvd": float(best_heur_row["avg_control_tvd"])
            }
        },
        "feature_importances": {name: float(imp) for name, imp in feat_imp}
    }
    
    with open(os.path.join("results", "regression_results_real_qpu_only.json"), "w") as f:
        json.dump(results_json, f, indent=4)
    print("Saved QPU-only regression results to results/regression_results_real_qpu_only.json")

if __name__ == "__main__":
    main()
