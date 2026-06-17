import sys
import os
import json
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
import run_qls_qpu_benchmark
import analyze_qls_benchmark
import analyze_qls_real_qpu

def test_full_pipeline_integration():
    # Define file paths
    raw_data_dir = os.path.join("data", "raw")
    processed_data_dir = os.path.join("data", "processed")
    results_dir = "results"
    figures_dir = "figures"
    
    results_json = os.path.join(raw_data_dir, "qls_qpu_benchmark_results.json")
    backup_results_json = results_json + ".backup"
    
    # 1. Back up existing results file if it exists
    has_backup = False
    if os.path.exists(results_json):
        if os.path.exists(backup_results_json):
            os.remove(backup_results_json)
        os.rename(results_json, backup_results_json)
        has_backup = True
        
    try:
        # 2. Run a micro simulator experiment (2 records max)
        run_qls_qpu_benchmark.run_experiment(
            backend_name="ibm_kingston_simulated",
            is_simulator=True,
            is_pilot=True,
            max_records=2,
            shots=100,
            repeats=1
        )
        
        # Verify raw JSON was created
        assert os.path.exists(results_json), "Raw results JSON should be created by benchmark runner"
        
        # 3. Inject a mock 'ibm_fez' record so analyze_qls_real_qpu has data to process
        with open(results_json, "r") as f:
            db = json.load(f)
            
        assert len(db.get("records", [])) > 0, "No records generated in integration test"
        
        # Change backend of both records to ibm_fez and force distinct groups
        db["records"][0]["backend"] = "ibm_fez"
        db["records"][0]["pair_type"] = "adjacent"
        db["records"][0]["physical_pair"] = [0, 1]
        db["records"][0]["direct_edge"] = True
        db["records"][0]["routing_distance"] = 1
        db["records"][0]["calibration_features"]["heuristic_cost"] = 0.01
        db["records"][0]["calibration_features"]["direct_two_qubit_error"] = 0.01
        
        if len(db["records"]) > 1:
            db["records"][1]["backend"] = "ibm_fez"
            db["records"][1]["pair_type"] = "routed"
            db["records"][1]["physical_pair"] = [7, 9]
            db["records"][1]["direct_edge"] = False
            db["records"][1]["routing_distance"] = 2
            db["records"][1]["calibration_features"]["heuristic_cost"] = 0.05
            
        with open(results_json, "w") as f:
            json.dump(db, f, indent=4)
            
        # 4. Run the full analysis script
        analyze_qls_benchmark.main()
        
        # 5. Run the QPU-only analysis script
        analyze_qls_real_qpu.main()
        
        # 6. Verify that datasets and result stats are successfully saved
        assert os.path.exists(os.path.join(processed_data_dir, "cleaned_dataset.csv"))
        assert os.path.exists(os.path.join(processed_data_dir, "tvd_summary_by_mapping.csv"))
        assert os.path.exists(os.path.join(processed_data_dir, "cleaned_dataset_real_qpu_only.csv"))
        assert os.path.exists(os.path.join(processed_data_dir, "tvd_summary_real_qpu_only.csv"))
        assert os.path.exists(os.path.join(results_dir, "regression_results.json"))
        assert os.path.exists(os.path.join(results_dir, "regression_results_real_qpu_only.json"))
        
    finally:
        # Cleanup test files
        test_files = [
            os.path.join(processed_data_dir, "cleaned_dataset.csv"),
            os.path.join(processed_data_dir, "tvd_summary_by_mapping.csv"),
            os.path.join(processed_data_dir, "cleaned_dataset_real_qpu_only.csv"),
            os.path.join(processed_data_dir, "tvd_summary_real_qpu_only.csv"),
            os.path.join(results_dir, "regression_results.json"),
            os.path.join(results_dir, "regression_results_real_qpu_only.json"),
            results_json
        ]
        for tf in test_files:
            if os.path.exists(tf):
                os.remove(tf)
                
        # Restore backup if it exists
        if has_backup:
            if os.path.exists(results_json):
                os.remove(results_json)
            os.rename(backup_results_json, results_json)
