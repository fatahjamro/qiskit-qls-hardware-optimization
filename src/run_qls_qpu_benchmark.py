import argparse
import json
import time
import sys
import random
import os
from qiskit import transpile
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Add script directory to sys.path to resolve imports when running from root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from qls_circuits import get_qls_circuits

def run_bit_ordering_unit_test():
    print("Running QLS bit-ordering unit test...")
    # Verify bit-ordering:
    # Qiskit little-endian: q0 is the rightmost bit, q1 is the leftmost bit in a 2-bit string (i.e. "q1q0").
    # If we apply X to q0, state is |01>. If we apply X to q1, state is |10>.
    from qiskit import QuantumCircuit
    qc_01 = QuantumCircuit(2)
    qc_01.x(0)
    qc_01.measure_all()
    
    qc_10 = QuantumCircuit(2)
    qc_10.x(1)
    qc_10.measure_all()
    
    sampler = StatevectorSampler()
    job = sampler.run([qc_01, qc_10])
    res = job.result()
    
    # Check key names dynamically
    pub_res_01 = res[0]
    meas_name_01 = list(pub_res_01.data.keys())[0] if hasattr(pub_res_01.data, 'keys') else 'meas'
    counts_01 = getattr(pub_res_01.data, meas_name_01).get_counts()
    
    pub_res_10 = res[1]
    meas_name_10 = list(pub_res_10.data.keys())[0] if hasattr(pub_res_10.data, 'keys') else 'meas'
    counts_10 = getattr(pub_res_10.data, meas_name_10).get_counts()
    
    state_01 = max(counts_01, key=counts_01.get)
    state_10 = max(counts_10, key=counts_10.get)
    
    if state_01 != "01" or state_10 != "10":
        print(f"ERROR: Bit-ordering unit test failed! Got |01> -> '{state_01}', |10> -> '{state_10}'")
        print("Expected Qiskit bit-ordering standard is not met.")
        sys.exit(1)
    print("Bit-ordering unit test passed successfully!")

def select_qubits(backend, is_pilot=True):
    target = backend.target
    props = backend.properties()
    
    cz_gate = target['cz']
    cnot_pairs = list(cz_gate.keys())
    
    unique_pairs = []
    for u, v in cnot_pairs:
        if u < v:
            unique_pairs.append((u, v))
            
    pair_costs = []
    for u, v in unique_pairs:
        try:
            cz_err = cz_gate[(u, v)].error
            readout_u = props.qubit_property(u)['readout_error'][0]
            readout_v = props.qubit_property(v)['readout_error'][0]
            
            t1_u = props.qubit_property(u)['T1'][0]
            t2_u = props.qubit_property(u)['T2'][0]
            t1_v = props.qubit_property(v)['T1'][0]
            t2_v = props.qubit_property(v)['T2'][0]
            
            coherence_cost = 1.0/t1_u + 1.0/t2_u + 1.0/t1_v + 1.0/t2_v
            cost = cz_err + 0.1 * (readout_u + readout_v) + 1e-6 * coherence_cost
            
            pair_costs.append({
                "pair": [u, v],
                "cost": cost,
                "cz_error": cz_err,
                "readout_error_u": readout_u,
                "readout_error_v": readout_v,
                "T1_u": t1_u,
                "T2_u": t2_u,
                "T1_v": t1_v,
                "T2_v": t2_v
            })
        except Exception:
            continue
            
    if not pair_costs:
        raise ValueError("Could not calculate costs for any qubit pairs.")
        
    pair_costs.sort(key=lambda x: x["cost"])
    
    n_adjacent = 6 if is_pilot else 10
    n_routed = 3 if is_pilot else 5
    
    # Select adjacent pairs
    selected_adjacent = []
    # 1. Best pairs
    best_count = 3 if is_pilot else 4
    selected_adjacent.extend(pair_costs[:best_count])
    
    # 2. Medium pairs
    medium_index = len(pair_costs) // 2
    medium_count = 1 if is_pilot else 3
    selected_adjacent.extend(pair_costs[medium_index : medium_index + medium_count])
    
    # 3. Spatially diverse/random connected pairs (excluding already chosen)
    chosen_pairs = {tuple(x["pair"]) for x in selected_adjacent}
    remaining_pairs = [x for x in pair_costs if tuple(x["pair"]) not in chosen_pairs]
    random.shuffle(remaining_pairs)
    diverse_count = 2 if is_pilot else 3
    selected_adjacent.extend(remaining_pairs[:diverse_count])
    
    # Select routed pairs (coupling distance 2 or 3)
    # Build simple graph from coupling map to compute distances
    from collections import deque
    adj_list = {}
    for u, v in unique_pairs:
        adj_list.setdefault(u, []).append(v)
        adj_list.setdefault(v, []).append(u)
        
    def get_shortest_path(start, end):
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    routed_candidates = []
    all_qubits = list(adj_list.keys())
    for i in range(len(all_qubits)):
        for j in range(i+1, len(all_qubits)):
            u, v = all_qubits[i], all_qubits[j]
            if v not in adj_list.get(u, []): # not directly connected
                path = get_shortest_path(u, v)
                dist = len(path) - 1
                if 2 <= dist <= 3:
                    # Construct mock/aggregate properties for features
                    try:
                        readout_u = props.qubit_property(u)['readout_error'][0]
                        readout_v = props.qubit_property(v)['readout_error'][0]
                        t1_u = props.qubit_property(u)['T1'][0]
                        t2_u = props.qubit_property(u)['T2'][0]
                        t1_v = props.qubit_property(v)['T1'][0]
                        t2_v = props.qubit_property(v)['T2'][0]
                        
                        routed_candidates.append({
                            "pair": [u, v],
                            "routing_distance": dist,
                            "shortest_path": path,
                            "intermediate_qubits": path[1:-1],
                            "readout_error_u": readout_u,
                            "readout_error_v": readout_v,
                            "T1_u": t1_u,
                            "T2_u": t2_u,
                            "T1_v": t1_v,
                            "T2_v": t2_v
                        })
                    except Exception:
                        continue
                        
    random.shuffle(routed_candidates)
    selected_routed = routed_candidates[:n_routed]
    
    return selected_adjacent, selected_routed

def get_transpiled_circuit_features(circuits, backend, pair, is_adjacent, routing_distance, shortest_path, is_simulator=False):
    # Transpile at opt level 1 to extract transpilation features
    if is_simulator:
        transpiled_circuits = transpile(circuits, basis_gates=['id','rz','sx','x','cz'], initial_layout=pair, optimization_level=1, seed_transpiler=42)
    else:
        transpiled_circuits = transpile(circuits, backend=backend, initial_layout=pair, optimization_level=1, seed_transpiler=42)
    
    per_circuit_transpilation = []
    depths = []
    two_q_counts = []
    swap_counts = []
    
    for idx, tc in enumerate(transpiled_circuits):
        ops = tc.count_ops()
        cz_count = ops.get('cz', 0)
        cx_count = ops.get('cx', 0)
        ecr_count = ops.get('ecr', 0)
        swap_count = ops.get('swap', 0)
        total_2q = cz_count + cx_count + ecr_count + swap_count * 3
        
        depths.append(tc.depth())
        two_q_counts.append(total_2q)
        swap_counts.append(swap_count)
        
        per_circuit_transpilation.append({
            "cell_index": idx,
            "qls_state": tc.name,
            "depth": tc.depth(),
            "two_qubit_gate_count": total_2q,
            "swap_count": swap_count,
            "cz_count": cz_count,
            "cx_count": cx_count,
            "ecr_count": ecr_count,
            "final_layout": list(pair)
        })
        
    summary = {
        "mean_depth": sum(depths) / len(depths),
        "max_depth": max(depths),
        "mean_two_qubit_gate_count": sum(two_q_counts) / len(two_q_counts),
        "max_two_qubit_gate_count": max(two_q_counts),
        "mean_swap_count": sum(swap_counts) / len(swap_counts),
        "max_swap_count": max(swap_counts),
        "total_two_qubit_gate_count": sum(two_q_counts),
        "total_swap_count": sum(swap_counts)
    }
    
    # Save a compact text summary of the first entangling QLS circuit (cell 4, Psi+) for verification
    compact_summary = str(transpiled_circuits[4])
    
    return summary, per_circuit_transpilation, compact_summary

def run_experiment(backend_name, is_simulator=False, is_pilot=True, max_records=None, shots=4000, repeats=3):
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Safety Safeguard Check
    if not is_simulator:
        prevent_qpu = os.environ.get("QLS_PREVENT_QPU", "").lower() in ("1", "true", "yes")
        is_ci = os.environ.get("CI", "").lower() == "true"
        if prevent_qpu or is_ci:
            print("\n=======================================================")
            print("SECURITY SAFEGUARD: IBM Quantum QPU submission is blocked!")
            print("QLS_PREVENT_QPU or CI environment variable is active.")
            print("=======================================================\n")
            raise RuntimeError("QPU submission blocked by safety environment variables.")
            
    run_bit_ordering_unit_test()
    
    # Load intermediate results for resume capability if they exist
    output_file = os.path.join("data", "raw", "qls_qpu_benchmark_results.json")
    existing_records = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "records" in data:
                    existing_records = data["records"]
                    print(f"Resuming experiment. Loaded {len(existing_records)} existing records.")
        except Exception as e:
            print(f"Warning: Failed to load existing records: {e}. Starting fresh.")

    existing_ids = {r["record_id"] for r in existing_records}
    
    # Set up Backend / Mock Backend properties
    if is_simulator:
        print("Setting up local simulator mode with mock ibm_fez calibration data...")
        # Create a mock class mimicking a backend to use identical code paths
        class MockBackend:
            class Target:
                def __init__(self):
                    # Mock 10 qubits with linear topology
                    cz_dict = {}
                    for i in range(9):
                        cz_dict[(i, i+1)] = type('Gate', (), {'error': random.uniform(0.005, 0.015)})
                        cz_dict[(i+1, i)] = type('Gate', (), {'error': random.uniform(0.005, 0.015)})
                    self.cz = {'cz': cz_dict}
                def __getitem__(self, key):
                    return self.cz[key]
            class Properties:
                def __init__(self):
                    self.last_update_date = str(time.time())
                def qubit_property(self, qubit, prop=None):
                    props = {
                        'readout_error': [random.uniform(0.01, 0.03)],
                        'T1': [random.uniform(100e-6, 250e-6)],
                        'T2': [random.uniform(80e-6, 180e-6)]
                    }
                    return props if prop is None else props[prop]
            def __init__(self):
                self.target = self.Target()
                self.properties = lambda: self.Properties()
        backend = MockBackend()
    else:
        print(f"Connecting to IBM backend: {backend_name}...")
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        
    props = backend.properties()
        
    adjacent_pairs, routed_pairs = select_qubits(backend, is_pilot=is_pilot)
    
    # Build complete configuration list
    all_configs = []
    
    # Adjacent configs
    for item in adjacent_pairs:
        for mitigation in ["baseline", "active"]:
            for rep in range(repeats):
                all_configs.append({
                    "pair_type": "adjacent",
                    "physical_pair": item["pair"],
                    "direct_edge": True,
                    "routing_distance": 1,
                    "shortest_path": item["pair"],
                    "intermediate_qubits": [],
                    "mitigation_setting": mitigation,
                    "repeat_index": rep,
                    "calibration": item
                })
                
    # Routed configs
    for item in routed_pairs:
        for mitigation in ["baseline", "active"]:
            for rep in range(repeats):
                all_configs.append({
                    "pair_type": "routed",
                    "physical_pair": item["pair"],
                    "direct_edge": False,
                    "routing_distance": item["routing_distance"],
                    "shortest_path": item["shortest_path"],
                    "intermediate_qubits": item["intermediate_qubits"],
                    "mitigation_setting": mitigation,
                    "repeat_index": rep,
                    "calibration": item
                })
                
    # Shuffle configurations to prevent drift and queue bias
    random.seed(42)
    random.shuffle(all_configs)
    
    if max_records is not None:
        all_configs = all_configs[:max_records]
        print(f"Truncated configurations list to max_records={max_records}")
        
    print(f"Total configurations to execute: {len(all_configs)}")
    
    records = list(existing_records)
    circuits = get_qls_circuits()
    
    for idx, cfg in enumerate(all_configs):
        pair = cfg["physical_pair"]
        mitigation = cfg["mitigation_setting"]
        rep = cfg["repeat_index"]
        pair_type = cfg["pair_type"]
        
        record_id = f"{pair_type}_{pair[0]}_{pair[1]}_{mitigation}_rep{rep}"
        
        if record_id in existing_ids:
            print(f"[{idx+1}/{len(all_configs)}] Skipping {record_id} (already completed).")
            continue
            
        print(f"[{idx+1}/{len(all_configs)}] Executing {record_id}...")
        
        # Get transpilation features
        summary, per_circuit_transpilation, compact_summary = get_transpiled_circuit_features(
            circuits, backend,
            pair, cfg["direct_edge"], cfg["routing_distance"], cfg["shortest_path"], is_simulator=is_simulator
        )
        
        # Calibration features mapping
        cal = cfg["calibration"]
        cal_features = {
            "T1_u": cal["T1_u"],
            "T1_v": cal["T1_v"],
            "T2_u": cal["T2_u"],
            "T2_v": cal["T2_v"],
            "readout_error_u": cal["readout_error_u"],
            "readout_error_v": cal["readout_error_v"],
            "direct_two_qubit_error": cal.get("cz_error", None), # None for routed
            "mean_path_cz_error": None,
            "max_path_cz_error": None,
            "sum_path_cz_error": None,
            "heuristic_cost": cal.get("cost", None)
        }
        
        # Fill path features for routed pairs
        if not cfg["direct_edge"] and not is_simulator:
            # Look up cz errors along shortest path
            cz_gate = backend.target['cz']
            path = cfg["shortest_path"]
            cz_errors = []
            for k in range(len(path)-1):
                p1, p2 = path[k], path[k+1]
                edge = (p1, p2) if (p1, p2) in cz_gate else (p2, p1)
                cz_errors.append(cz_gate[edge].error)
            cal_features["mean_path_cz_error"] = sum(cz_errors) / len(cz_errors)
            cal_features["max_path_cz_error"] = max(cz_errors)
            cal_features["sum_path_cz_error"] = sum(cz_errors)
        elif not cfg["direct_edge"] and is_simulator:
            # Mock values for simulator
            cal_features["mean_path_cz_error"] = random.uniform(0.01, 0.02)
            cal_features["max_path_cz_error"] = random.uniform(0.015, 0.025)
            cal_features["sum_path_cz_error"] = cal_features["mean_path_cz_error"] * cfg["routing_distance"]
            
        # Mock or run actual backend counts
        raw_counts = []
        job_id = "mock_job_id"
        sub_time = time.time()
        comp_time = time.time()
        
        # Active mitigation options validation
        m_dd = 1 if mitigation == "active" else 0
        m_twirling = 1 if mitigation == "active" else 0
        dd_seq = "XY4" if mitigation == "active" else "None"
        
        if is_simulator:
            sampler = StatevectorSampler()
            job = sampler.run(circuits)
            res = job.result()
            for c_idx, qc in enumerate(circuits):
                pub_res = res[c_idx]
                meas_name = list(pub_res.data.keys())[0] if hasattr(pub_res.data, 'keys') else 'meas'
                counts = getattr(pub_res.data, meas_name).get_counts()
                
                # Scale statevector counts to target shots and add minor mock noise
                total_shots = sum(counts.values())
                noisy_counts = {}
                # Add baseline depolarizing noise mock depending on SWAP overhead and active mitigation
                noise_rate = 0.02 * summary["total_swap_count"]
                if mitigation == "baseline":
                    noise_rate += 0.05
                else:
                    noise_rate += 0.02 # active mitigation decreases noise rate slightly
                noise_rate = min(noise_rate, 0.9)
                
                for state in ["00", "01", "10", "11"]:
                    ideal_p = counts.get(state, 0.0) / total_shots
                    noisy_p = ideal_p * (1.0 - noise_rate) + noise_rate / 4.0
                    noisy_counts[state] = int(round(noisy_p * shots))
                raw_counts.append({
                    "cell_index": c_idx,
                    "qls_state": qc.name,
                    "counts": noisy_counts
                })
        else:
            # Actual hardware job submission
            try:
                sampler = SamplerV2(mode=backend)
                sampler.options.default_shots = shots
                if mitigation == "active":
                    # Check active mitigation support in Runtime options
                    try:
                        sampler.options.dynamical_decoupling.enable = True
                        sampler.options.dynamical_decoupling.sequence_type = 'XY4'
                        sampler.options.twirling.enable_measure = True
                    except AttributeError as ae:
                        print(f"Warning: Active mitigation options not fully supported on this Qiskit Runtime version: {ae}")
                        
                # Transpile for this specific configuration's layout
                isa_circuits = transpile(circuits, backend=backend, initial_layout=pair, optimization_level=3 if mitigation=="active" else 1, seed_transpiler=42)
                
                print(f"Submitting job to QPU {backend_name}...")
                job = sampler.run(isa_circuits)
                job_id = job.job_id()
                print(f"Job submitted successfully. Job ID: {job_id}")
                
                # Wait for job result
                res = job.result()
                comp_time = time.time()
                
                for c_idx, qc in enumerate(circuits):
                    pub_res = res[c_idx]
                    meas_name = list(pub_res.data.keys())[0] if hasattr(pub_res.data, 'keys') else 'meas'
                    counts = getattr(pub_res.data, meas_name).get_counts()
                    raw_counts.append({
                        "cell_index": c_idx,
                        "qls_state": qc.name,
                        "counts": counts
                    })
            except Exception as ex:
                print(f"ERROR executing job for {record_id}: {ex}")
                # Save details to failed_jobs.json
                failed_entry = {
                    "record_id": record_id,
                    "job_id": job_id,
                    "error_message": str(ex),
                    "timestamp": time.time()
                }
                failed_file = os.path.join("results", "failed_jobs.json")
                failed_data = []
                if os.path.exists(failed_file):
                    try:
                        with open(failed_file, "r") as f:
                            failed_data = json.load(f)
                    except Exception:
                        pass
                failed_data.append(failed_entry)
                with open(failed_file, "w") as f:
                    json.dump(failed_data, f, indent=4)
                continue
                
        # Record successful observation
        record_entry = {
            "record_id": record_id,
            "backend": backend_name if not is_simulator else "ibm_kingston_simulated",
            "calibration_timestamp": str(props.last_update_date) if not is_simulator and hasattr(props, 'last_update_date') else str(time.time()),
            "submission_time": sub_time,
            "completion_time": comp_time,
            "job_id": job_id,
            "repeat_index": rep,
            "shots": shots,
            "pair_type": pair_type,
            "physical_pair": pair,
            "direct_edge": cfg["direct_edge"],
            "routing_distance": cfg["routing_distance"],
            "shortest_path": cfg["shortest_path"],
            "intermediate_qubits": cfg["intermediate_qubits"],
            "mitigation_setting": mitigation,
            "M_DD": m_dd,
            "M_twirling": m_twirling,
            "optimization_level": 3 if mitigation == "active" else 1,
            "dd_sequence": dd_seq,
            "measurement_twirling": True if mitigation == "active" else False,
            "calibration_features": cal_features,
            "transpilation_summary": summary,
            "per_circuit_transpilation": per_circuit_transpilation,
            "transpiled_circuit_sample": compact_summary,
            "raw_counts_16_cells": raw_counts
        }
        
        records.append(record_entry)
        
        # Save intermediate results incrementally
        output_data = {
            "schema_version": "1.0.0",
            "backend_name": backend_name if not is_simulator else "ibm_kingston_simulated",
            "records": records
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
            
    print(f"Data collection completed. Total records saved: {len(records)}")

def main():
    parser = argparse.ArgumentParser(description="QLS QPU Benchmarking Pipeline with ML Feature Collection")
    parser.add_argument("--simulator", action="store_true", help="Run in local simulator mode")
    parser.add_argument("--backend", type=str, default="ibm_fez", help="Name of the IBM QPU backend to run on")
    parser.add_argument("--pilot", action="store_true", default=True, help="Run pilot configuration (default: True)")
    parser.add_argument("--full", action="store_true", help="Run full configuration (10 adj, 5 routed)")
    parser.add_argument("--max-records", type=int, default=None, help="Limit maximum number of configurations to run")
    parser.add_argument("--shots", type=int, default=4000, help="Number of shots per circuit")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per configuration")
    parser.add_argument("--confirm", action="store_true", help="Confirm real QPU execution (bypasses interactive prompt)")
    args = parser.parse_args()
    
    is_pilot = not args.full
    
    if args.simulator:
        run_experiment("ibm_kingston_simulated", is_simulator=True, is_pilot=is_pilot, max_records=args.max_records, shots=args.shots, repeats=args.repeats)
    else:
        if not args.confirm:
            # Check environment safeguard first to fail fast without input prompts
            prevent_qpu = os.environ.get("QLS_PREVENT_QPU", "").lower() in ("1", "true", "yes")
            is_ci = os.environ.get("CI", "").lower() == "true"
            if prevent_qpu or is_ci:
                print("\n=======================================================")
                print("SECURITY SAFEGUARD: IBM Quantum QPU submission is blocked!")
                print("QLS_PREVENT_QPU or CI environment variable is active.")
                print("=======================================================\n")
                sys.exit(1)
            try:
                ans = input(f"WARNING: Submitting jobs to real QPU backend '{args.backend}'. Proceed? (y/N): ")
                if ans.lower() != 'y':
                    print("Cancelled.")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                sys.exit(1)
        run_experiment(args.backend, is_simulator=False, is_pilot=is_pilot, max_records=args.max_records, shots=args.shots, repeats=args.repeats)

if __name__ == "__main__":
    main()
