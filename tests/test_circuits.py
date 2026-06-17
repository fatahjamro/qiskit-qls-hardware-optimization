import sys
import os
from qiskit import QuantumCircuit

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from qls_circuits import get_qls_circuits, CELL_NAMES

def test_get_qls_circuits():
    circuits = get_qls_circuits()
    
    assert len(circuits) == 16, "Should return exactly 16 circuits for the 4x4 matrix"
    assert len(CELL_NAMES) == 16, "Cell names list must have 16 names"
    
    for idx, (qc, expected_name) in enumerate(zip(circuits, CELL_NAMES)):
        assert isinstance(qc, QuantumCircuit), f"Item at index {idx} must be a QuantumCircuit"
        assert qc.name == expected_name, f"Circuit at {idx} name mismatch: got '{qc.name}', expected '{expected_name}'"
        # Ensure it has measurements added
        assert qc.num_clbits > 0, f"Circuit at {idx} should contain classical bits for measurements"
