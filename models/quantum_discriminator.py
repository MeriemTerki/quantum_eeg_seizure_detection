# models/quantum_discriminator.py
"""
Quantum Discriminator for QCGAN
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

from config import Config
from utils.quantum_utils import create_quantum_device


class QuantumDiscriminator(nn.Module):
    """
    Quantum Discriminator to classify real vs fake EEG
    """
    
    def __init__(self, config: Config = Config):
        super().__init__()
        self.config = config
        
        # Classical pre-processing
        # Reduce dimensionality before quantum encoding
        input_dim = config.NUM_CHANNELS * config.SEGMENT_LENGTH
        
        self.pre_process = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, config.NUM_QUBITS)
        )
        
        # Quantum circuit setup
        total_qubits = config.NUM_QUBITS + config.NUM_CONDITION_QUBITS
        self.dev = create_quantum_device(total_qubits, config.QUANTUM_DEVICE)
        
        # Qubit allocation
        self.condition_qubits = list(range(config.NUM_CONDITION_QUBITS))
        self.data_qubits = list(range(
            config.NUM_CONDITION_QUBITS,
            total_qubits
        ))
        
        print(f"  Discriminator qubits:")
        print(f"    Total: {total_qubits}")
        print(f"    Condition: {self.condition_qubits}")
        print(f"    Data: {self.data_qubits}")
        
        # Initialize quantum parameters
        n_params = config.NUM_QUBITS * 3 * config.NUM_LAYERS
        self.register_buffer('_params_init', torch.randn(n_params) * 0.1)
        self.quantum_params = nn.Parameter(self._params_init.clone())
        
        # Classical post-processing
        self.post_process = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    
    def _circuit(self, data, condition, params):
        """
        Quantum circuit for discriminator
        
        Args:
            data: encoded EEG data (numpy array)
            condition: seizure type (int)
            params: circuit parameters (numpy array)
            
        Returns:
            expectation value
        """
        # Encode condition
        if condition == 1:
            qml.PauliX(wires=self.condition_qubits[0])
        
        # Encode data
        normalized = np.arctan(data) + np.pi/4
        for i, q in enumerate(self.data_qubits):
            if i < len(normalized):
                qml.RY(float(normalized[i]), wires=q)
        
        # Parameterized layers
        params_per_layer = self.config.NUM_QUBITS * 3
        
        for layer in range(self.config.NUM_LAYERS):
            layer_params = params[
                layer * params_per_layer : (layer + 1) * params_per_layer
            ]
            
            # Rotation gates
            for i, q in enumerate(self.data_qubits):
                qml.Rot(
                    float(layer_params[i*3]),
                    float(layer_params[i*3+1]),
                    float(layer_params[i*3+2]),
                    wires=q
                )
            
            # Entanglement (CB topology)
            for i in range(0, len(self.data_qubits)-1, 2):
                qml.CNOT(wires=[self.data_qubits[i], self.data_qubits[i+1]])
            
            for i in range(1, len(self.data_qubits)-1, 2):
                qml.CNOT(wires=[self.data_qubits[i], self.data_qubits[i+1]])
        
        # Measure first data qubit
        return qml.expval(qml.PauliZ(self.data_qubits[0]))
    
    def forward(self, eeg, condition):
        """
        Classify EEG as real or fake
        
        Args:
            eeg: (batch_size, n_channels, segment_length)
            condition: (batch_size,) - seizure type
            
        Returns:
            prob: (batch_size, 1) - probability of being real
        """
        batch_size = eeg.shape[0]
        device = eeg.device
        
        # Pre-process with classical network
        encoded = self.pre_process(eeg)  # (batch_size, num_qubits)
        
        # Ensure encoded is float32
        encoded = encoded.float()
        
        # Convert params to numpy
        params_np = self.quantum_params.detach().cpu().numpy()
        
        # Create QNode
        qnode = qml.QNode(self._circuit, self.dev, interface='autograd')
        
        # Process through quantum circuit
        outputs = []
        for i in range(batch_size):
            # Convert to numpy
            data_np = encoded[i].detach().cpu().numpy().astype(np.float64)
            condition_int = int(condition[i].item())
            
            # Get measurement
            measurement = qnode(data_np, condition_int, params_np)
            
            # Convert to tensor
            measurement_tensor = torch.tensor(
                [float(measurement)], 
                dtype=torch.float32,
                device=device
            )
            outputs.append(measurement_tensor)
        
        # Stack and reshape
        quantum_output = torch.stack(outputs).reshape(batch_size, 1)
        
        # Post-process
        prob = self.post_process(quantum_output)
        
        return prob


if __name__ == "__main__":
    # Test discriminator
    print("Testing Quantum Discriminator")
    print("=" * 50)
    
    Config.set_seed()
    
    discriminator = QuantumDiscriminator()
    
    # Test forward pass
    batch_size = 2
    eeg = torch.randn(batch_size, Config.NUM_CHANNELS, Config.SEGMENT_LENGTH)
    condition = torch.LongTensor([0, 1])
    
    print(f"Input EEG shape: {eeg.shape}")
    print(f"Condition: {condition}")
    
    output = discriminator(eeg, condition)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (probability): {output.squeeze()}")
    print("\n✓ Discriminator working!")