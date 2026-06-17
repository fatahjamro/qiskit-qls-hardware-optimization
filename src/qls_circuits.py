from qiskit import QuantumCircuit

# Ideal probability distributions for QLS states
IDEAL_DISTRIBUTIONS = {
    "00": {"00": 1.0, "01": 0.0, "10": 0.0, "11": 0.0},
    "01": {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0},
    "10": {"00": 0.0, "01": 0.0, "10": 1.0, "11": 0.0},
    "11": {"00": 0.0, "01": 0.0, "10": 0.0, "11": 1.0},
    "Psi+": {"00": 0.0, "01": 0.5, "10": 0.5, "11": 0.0},
    "Phi+": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
    "Phi-": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
    "Psi-": {"00": 0.0, "01": 0.5, "10": 0.5, "11": 0.0}
}

# Twin pairs in the QLS grid (indices 0 to 15, in row-major order)
TWIN_PAIRS = {
    "00": (0, 11),
    "01": (1, 10),
    "10": (2, 9),
    "11": (3, 8),
    "Psi+": (4, 15),
    "Phi+": (5, 14),
    "Phi-": (6, 13),
    "Psi-": (7, 12)
}

# Cell order of QLS matrix
CELL_NAMES = [
    "00", "01", "10", "11",
    "Psi+", "Phi+", "Phi-", "Psi-",
    "11", "10", "01", "00",
    "Psi-", "Phi-", "Phi+", "Psi+"
]

def make_psi_plus():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    return qc

def make_phi_plus():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def make_phi_minus():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def make_psi_minus():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)
    return qc

def get_qls_circuits():
    # 16 states of the QLS matrix (row-major order)
    state_constructors = [
        # Row 0
        lambda: QuantumCircuit(2), # |00>
        lambda: (qc := QuantumCircuit(2), qc.x(0), qc)[2], # |01>
        lambda: (qc := QuantumCircuit(2), qc.x(1), qc)[2], # |10>
        lambda: (qc := QuantumCircuit(2), qc.x(0), qc.x(1), qc)[3], # |11>
        # Row 1
        make_psi_plus,
        make_phi_plus,
        make_phi_minus,
        make_psi_minus,
        # Row 2
        lambda: (qc := QuantumCircuit(2), qc.x(0), qc.x(1), qc)[3], # |11>
        lambda: (qc := QuantumCircuit(2), qc.x(1), qc)[2], # |10>
        lambda: (qc := QuantumCircuit(2), qc.x(0), qc)[2], # |01>
        lambda: QuantumCircuit(2), # |00>
        # Row 3
        make_psi_minus,
        make_phi_minus,
        make_phi_plus,
        make_psi_plus
    ]
    
    circuits = []
    names = [
        "00", "01", "10", "11",
        "Psi+", "Phi+", "Phi-", "Psi-",
        "11", "10", "01", "00",
        "Psi-", "Phi-", "Phi+", "Psi+"
    ]
    
    for name, constructor in zip(names, state_constructors):
        qc = constructor()
        qc.name = name
        qc.measure_all()
        circuits.append(qc)
        
    return circuits
