# Cleaned Dataset Schema

This document details the features contained in `data/processed/cleaned_dataset.csv` and `data/processed/cleaned_dataset_real_qpu_only.csv`.

---

## Feature Categories

### 1. Layout Identifiers
*   `record_id` (string): Unique identifier for the experiment run, formatted as `{pair_type}_{qubit_u}_{qubit_v}_{mitigation}_rep{repeat_index}`.
*   `pair_type` (string): Either `adjacent` (physically connected coupling map edge) or `routed` (non-adjacent qubits requiring SWAP gates).
*   `pair_type_encoded` (int): Binary encoding of `pair_type` (`1` = adjacent, `0` = routed).
*   `u` (int): Physical index of the first qubit in the pair.
*   `v` (int): Physical index of the second qubit in the pair.
*   `group_id` (string): Group grouping name, formatted as `{pair_type}_{min_qubit}_{max_qubit}`.
*   `direct_edge` (int): Binary indicator (`1` = direct connection exists, `0` = non-adjacent).
*   `routing_distance` (int): Topological distance (number of hops) between the mapped qubits on the coupling map.

### 2. Qubit Calibration Features
*   `T1_u`, `T1_v` (float): Coherence time $T_1$ in seconds for qubits $u$ and $v$ respectively.
*   `T2_u`, `T2_v` (float): Coherence time $T_2$ in seconds for qubits $u$ and $v$ respectively.
*   `readout_error_u`, `readout_error_v` (float): Readout assignment error rate for qubits $u$ and $v$ respectively.
*   `direct_two_qubit_error` (float): Error rate of the $CZ$ or $CNOT$ gate on the direct edge (if adjacent; otherwise `NaN`).
*   `mean_path_cz_error`, `max_path_cz_error`, `sum_path_cz_error` (float): Statistical metrics of $CZ$ errors along the shortest path routing the non-adjacent pair (for routed pairs; otherwise `NaN`).
*   `heuristic_cost` (float): A composite calibration error heuristic cost value calculated by the qubit selector.

### 3. Transpilation Features
*   `mean_depth`, `max_depth` (float): Average and maximum transpiled gate depth across the QLS circuit batch.
*   `mean_two_qubit_gate_count`, `max_two_qubit_gate_count` (float): Average and maximum count of two-qubit entangling gates.
*   `mean_swap_count`, `max_swap_count` (float): Average and maximum count of SWAP gates inserted by the transpiler router.
*   `total_two_qubit_gate_count` (int): Total entangling gate count summed across all 16 circuits.
*   `total_swap_count` (int): Total SWAP gate count summed across all 16 circuits.

### 4. Active Error Mitigation
*   `M_DD` (int): Binary flag indicating if XY4 Dynamical Decoupling sequences are enabled (`1` = active, `0` = baseline).
*   `M_twirling` (int): Binary flag indicating if Measurement Twirling is enabled (`1` = active, `0` = baseline).
*   `optimization_level` (int): Qiskit compiler optimization level (`1` or `3`).

---

## Target Metrics (Labels)

These fields represent the measured errors (Total Variation Distance) evaluated on QPU hardware:

*   `avg_twin_tvd` (float): **Primary Target**. Average TVD between twin-state pairs (quantifies noise symmetry preservation).
*   `avg_cell_tvd` (float): Average TVD between each cell's measured distribution and its theoretical ideal distribution.
*   `avg_control_tvd` (float): Average TVD compared against control distributions.
