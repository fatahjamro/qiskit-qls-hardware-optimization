# QLS-TVD Reproduction Instructions

This guide walks you through setting up and running the Quantum Latin Square (QLS) benchmarking pipeline and data analysis.

---

## 1. Setup & Environment

First, install the required packages using pip:
```bash
pip install -r requirements.txt
```

Alternatively, if using Conda, create the environment from the provided configuration:
```bash
conda env create -f environment.yml
conda activate qiskit-qls-opt
```

---

## 2. Running Safe Local Simulations

To test the code end-to-end without submitting any jobs to physical quantum hardware, run the experiment in simulator mode:
```bash
python src/run_qls_qpu_benchmark.py --simulator --pilot
```

You can limit the runtime by restricting the maximum number of executed configurations:
```bash
python src/run_qls_qpu_benchmark.py --simulator --pilot --max-records 4
```

This generates or updates the raw data file at:
`data/raw/qls_qpu_benchmark_results.json`

---

## 3. Security Safeguards

To prevent accidental physical job submissions (and avoid unexpected IBM Quantum credit usage), the code respects the `QLS_PREVENT_QPU` environment variable. 

If this variable is set to `1`, `true`, or if run in a `CI` runner, any attempt to run without the `--simulator` flag will crash immediately before executing any Qiskit Runtime calls:

```bash
# Set safeguard
export QLS_PREVENT_QPU=1

# Attempting a real hardware run will crash safely:
python src/run_qls_qpu_benchmark.py --backend ibm_fez
```

To allow real hardware execution, you must explicitly unset this safeguard:
```bash
unset QLS_PREVENT_QPU
```

---

## 4. Re-running the Data Analysis & ML Models

To run the Leave-One-Group-Out machine learning pipeline and regenerate all tabular data and charts:

```bash
# Run analysis on full simulator/QPU data
python src/analyze_qls_benchmark.py

# Run analysis on real QPU-only data
python src/analyze_qls_real_qpu.py
```

These scripts output:
*   Processed datasets to `data/processed/`
*   Predictive regression results to `results/`
*   Symmetry and comparison plots to `figures/`
