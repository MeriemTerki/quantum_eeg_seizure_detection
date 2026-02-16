# utils/quantum_utils.py
"""
Quantum circuit utilities for QCGAN
"""

import pennylane as qml
import numpy as np
import torch
from typing import List

from config import Config


class QuantumCircuitBuilder:
    """Build quantum circuits for generator and discriminator"""
    
    def __init__(self, config: Config = Config):
        self.config = config
    
    def create_entanglement_layer(self, qubits: List[int], weights):
        """
        Create entanglement layer with Circuit Block (CB) topology
        
        Args:
            qubits: list of qubit indices
            weights: parameters for rotation gates (numpy array)
        """
        # Ensure weights is numpy
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights, dtype=np.float64)
        
        n_qubits = len(qubits)
        
        # Rotation gates
        for i, q in enumerate(qubits):
            qml.Rot(
                float(weights[i*3]), 
                float(weights[i*3+1]), 
                float(weights[i*3+2]), 
                wires=q
            )
        
        # Entanglement with CB topology
        # Layer 1: even pairs
        for i in range(0, n_qubits-1, 2):
            qml.CNOT(wires=[qubits[i], qubits[i+1]])
        
        # Layer 2: odd pairs
        for i in range(1, n_qubits-1, 2):
            qml.CNOT(wires=[qubits[i], qubits[i+1]])
    
    def encode_classical_data(self, data, qubits):
        """
        Encode classical data into quantum state using amplitude encoding
        
        Args:
            data: classical data to encode (numpy array or list)
            qubits: list of qubit indices
        """
        # Ensure data is numpy
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float64)
        
        # Normalize data to [0, π/2] for RY gates
        normalized = np.arctan(data) + np.pi/4
        
        for i, q in enumerate(qubits):
            if i < len(normalized):
                qml.RY(float(normalized[i]), wires=q)
    
    def encode_conditional_info(self, condition_qubits, condition_type):
        """
        Encode conditional information (seizure type)
        
        Args:
            condition_qubits: qubits for condition encoding
            condition_type: integer (0 or 1)
        """
        # Ensure it's an integer
        if isinstance(condition_type, torch.Tensor):
            condition_type = int(condition_type.item())
        else:
            condition_type = int(condition_type)
        
        # Binary encoding: 0 -> |0⟩, 1 -> |1⟩
        if condition_type == 1:
            qml.PauliX(wires=condition_qubits[0])


class QuantumNoise:
    """Generate quantum noise for generator input"""
    
    @staticmethod
    def sample_noise(batch_size: int, noise_dim: int, device='cpu'):
        """
        Sample random noise from latent space
        
        Args:
            batch_size: number of samples
            noise_dim: dimension of noise vector
            device: torch device
            
        Returns:
            noise: (batch_size, noise_dim)
        """
        # Sample from uniform distribution [0, 2π]
        noise = torch.rand(batch_size, noise_dim, device=device) * 2 * np.pi
        return noise


def parameter_shift_gradient(circuit, params, param_idx, shift=np.pi/2):
    """
    Calculate gradient using parameter shift rule
    
    Args:
        circuit: quantum circuit function
        params: current parameters (numpy array)
        param_idx: index of parameter to differentiate
        shift: shift amount (π/2 for standard gates)
        
    Returns:
        gradient: ∂⟨σZ⟩/∂θ
    """
    # Ensure params is numpy
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    params = np.array(params, dtype=np.float64)
    
    # Forward shift
    params_plus = params.copy()
    params_plus[param_idx] += shift
    val_plus = circuit(params_plus)
    
    # Backward shift
    params_minus = params.copy()
    params_minus[param_idx] -= shift
    val_minus = circuit(params_minus)
    
    # Gradient
    gradient = (val_plus - val_minus) / 2
    
    return gradient


def create_quantum_device(n_qubits: int, device_name: str = None):
    """
    Create PennyLane quantum device
    
    Args:
        n_qubits: number of qubits
        device_name: device name (default from config)
        
    Returns:
        device: PennyLane device
    """
    if device_name is None:
        device_name = Config.QUANTUM_DEVICE
    
    dev = qml.device(device_name, wires=n_qubits)
    return dev


if __name__ == "__main__":
    # Test quantum circuit builder
    print("Testing Quantum Circuit Builder")
    print("=" * 50)
    
    builder = QuantumCircuitBuilder()
    
    # Create simple test circuit
    n_qubits = 4
    dev = create_quantum_device(n_qubits)
    
    @qml.qnode(dev)
    def test_circuit(weights):
        builder.create_entanglement_layer(list(range(n_qubits)), weights)
        return qml.expval(qml.PauliZ(0))
    
    # Test with random weights
    weights = np.random.randn(n_qubits * 3)
    result = test_circuit(weights)
    
    print(f"Test circuit result: {result}")
    print("✓ Quantum circuit builder working!")