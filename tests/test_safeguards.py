import sys
import os
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from run_qls_qpu_benchmark import run_experiment

def test_qpu_prevent_safeguard(monkeypatch):
    # Enable safety blocker
    monkeypatch.setenv("QLS_PREVENT_QPU", "1")
    monkeypatch.delenv("CI", raising=False)
    
    with pytest.raises(RuntimeError) as exc_info:
        run_experiment("ibm_fake_backend", is_simulator=False)
    
    assert "QPU submission blocked by safety environment variables" in str(exc_info.value)

def test_ci_prevent_safeguard(monkeypatch):
    # Enable CI environment block
    monkeypatch.delenv("QLS_PREVENT_QPU", raising=False)
    monkeypatch.setenv("CI", "true")
    
    with pytest.raises(RuntimeError) as exc_info:
        run_experiment("ibm_fake_backend", is_simulator=False)
        
    assert "QPU submission blocked by safety environment variables" in str(exc_info.value)
