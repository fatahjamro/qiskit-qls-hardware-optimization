# Qiskit QLS Hardware Optimization

> **Evaluating Hardware-Aware Circuit Optimization and Active Error Mitigation on Quantum Latin Squares.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4?logo=qiskit)](https://qiskit.org/)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Heron-052FAD?logo=ibm)](https://www.ibm.com/quantum)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1271867286.svg)](https://doi.org/10.5281/zenodo.20728901)

This repository contains the benchmarking suite, experimental QPU data, and analysis code for our study evaluating the spatial noise symmetries and compiler routing overhead of a $4 \times 4$ Bell-state Quantum Latin Square (QLS) on physical quantum processors.

Our experiment compares:
* **Optimal Qubit Mapping:** Dynamically mapping circuits to the physical qubits with the highest gate and readout fidelities.
* **Active Error Suppression:** Evaluating compiler optimizations combined with active XY4 dynamical decoupling sequences and measurement twirling.
* **Spatial Variation Control:** Quantifying the noise profile shift on a disjoint adjacent physical qubit pair.
* **Compiler Routing Overhead:** Stress-testing the noise topography by forcing non-adjacent qubit layouts that require the transpiler to insert SWAP gates.

---

## Experimental Results

The benchmarking suite was executed on the 156-qubit Heron backend **`ibm_fez`** with $N_{\text{shots}} = 4000$. The key Total Variation Distance (TVD) results are summarized below:

| Configuration | Avg Cell TVD | Avg Twin TVD | Avg Ctrl TVD |
| :--- | :---: | :---: | :---: |
| **Config A (Baseline - Optimal Qubits)** | 0.0247 | 0.0078 | 0.7637 |
| **Config B (Suppressed \& Optimized)** | 0.0253 | 0.0073 | 0.7624 |
| **Config C1 (Spatial Noise Control)** | 0.0290 | 0.0069 | 0.7602 |
| **Config C2 (Routing Stress Test)** | 0.0252 | 0.0148 | 0.7712 |

### Key Takeaways:
1. **Symmetry Preservation:** For adjacent qubit mappings (Configs A, B, and C1), the average twin-state TVD remains within the statistical shot-noise limit ($\sim 0.008$), proving excellent noise stability.
2. **Routing Penalty:** Forcing non-adjacent layouts (Config C2) requiring transpiler SWAPs **doubles** the average twin-state TVD to **$0.0148$** (exceeding the shot noise threshold), demonstrating that compiler-routed gates degrade spatial noise symmetries.
3. **Active Mitigation:** Active suppression sequences (Config B) successfully enforce a more uniform noise profile across entangled states while introducing negligible control pulse overhead.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
To process the compiled experimental QPU results and regenerate the publication figures without running new QPU jobs:
```bash
python3 src/analyze_qls_benchmark.py
python3 src/analyze_qls_real_qpu.py
```
This outputs the summary metric table to the console, processes datasets in `data/processed/`, saves regression outcomes in `results/`, and exports plots to `figures/`.

### 3. Run New QPU Benchmarks
To run the benchmarks yourself (requires an active IBM Quantum subscription with QPU access):
```bash
# Run a dry-run local simulator verification (safe, zero hardware credits used):
python3 src/run_qls_qpu_benchmark.py --simulator

# Run on a physical backend:
python3 src/run_qls_qpu_benchmark.py --backend <your_backend_name>
```

> [!TIP]
> **IBM Quantum QPU Safeguard:**
> To prevent accidental job submissions to QPU hardware in testing or CI environments, set the `QLS_PREVENT_QPU=1` environment variable. Any attempt to invoke a hardware run will trigger a security fail-fast exit.

---

## Repository Structure

```
qiskit-qls-hardware-optimization/
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment definition
├── CITATION.cff                   # Citation metadata
├── .gitignore                     # Git configuration
│
├── src/                           # Code scripts
│   ├── qls_circuits.py            # Central circuit generators
│   ├── run_qls_qpu_benchmark.py   # QPU run & batching script
│   ├── analyze_qls_benchmark.py   # Analysis and figure generation
│   └── analyze_qls_real_qpu.py    # QPU-only analysis
│
├── data/                          # Data files
│   ├── raw/
│   │   └── qls_qpu_benchmark_results.json # Raw counts returned from QPU
│   └── processed/
│       ├── cleaned_dataset.csv            # Processed QLS datasets
│       ├── cleaned_dataset_real_qpu_only.csv
│       ├── tvd_summary_by_mapping.csv     # Compiled TVD summaries
│       └── tvd_summary_real_qpu_only.csv
│
├── results/                       # Machine Learning regression outputs
│   ├── regression_results.json
│   └── regression_results_real_qpu_only.json
│
├── figures/                       # Publication-ready plots
│   ├── feature_importance.png
│   ├── predicted_vs_measured_twin_tvd.png
│   └── selector_comparison_twin_tvd.png
│
├── docs/                          # In-depth guides
│   ├── reproduction_instructions.md   # Step-by-step reproduction guide
│   └── dataset_schema.md              # Cleaned dataset feature schema
│
└── manuscript/                    # LaTeX manuscript draft
    ├── AQIS/
    ├── IEEEtran.cls
    ├── ai4qc_paper.tex
    ├── references.bib
    ├── ai4qc_paper.pdf
    └── qls_benchmark_comparison.png
```

> [!IMPORTANT]
> **LaTeX Manuscript Exclusion:** To comply with IEEE conference preprint and copyright policies, the local `manuscript/` folder (containing the draft source code and compiled PDF) is ignored by Git and will not be pushed to public repositories.


---

## Cite This Work

If you use this codebase or data in your research, please cite our paper:
```bibtex
@unpublished{fatah2026evaluating,
  title={Evaluating Hardware-Aware Circuit Optimization and Active Error Mitigation on Quantum Latin Squares},
  author={Fatah, Abdul and McLoughlin, Ian and Ghafoor, Saim},
  note={Preprint / Unpublished manuscript},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.
