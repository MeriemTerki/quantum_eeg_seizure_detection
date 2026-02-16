# models/quantum_generator.py
"""
Quantum Generator using patch method
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List

from config import Config
from utils.quantum_utils import create_quantum_device


class QuantumSubGenerator(nn.Module):
    """Single quantum sub-generator for one patch"""
    
    def __init__(self, config: Config, sub_gen_idx: int):
        super().__init__()
        
        self.config = config
        self.sub_gen_idx = sub_gen_idx
        
        # Total qubits needed
        total_qubits = config.NUM_QUBITS + config.NUM_AUXILIARY_QUBITS + config.NUM_CONDITION_QUBITS
        
        # Create quantum device
        self.dev = create_quantum_device(total_qubits, config.QUANTUM_DEVICE)
        
        # Qubit allocation
        self.condition_qubits = list(range(config.NUM_CONDITION_QUBITS))
        
        # Data qubits (for computation)
        data_start = config.NUM_CONDITION_QUBITS
        data_end = data_start + config.NUM_QUBITS
        self.data_qubits = list(range(data_start, data_end))
        
        # Auxiliary qubits (won't be measured)
        aux_start = data_end
        aux_end = aux_start + config.NUM_AUXILIARY_QUBITS
        self.aux_qubits = list(range(aux_start, aux_end))
        
        # Measurement qubits (data qubits MINUS auxiliary)
        # We'll use the first (NUM_QUBITS - NUM_AUXILIARY_QUBITS) qubits for measurement
        self.measurement_qubits = self.data_qubits[:-config.NUM_AUXILIARY_QUBITS]
        
        print(f"    Sub-gen {sub_gen_idx}:")
        print(f"      Total qubits: {total_qubits}")
        print(f"      Condition qubits: {self.condition_qubits}")
        print(f"      Data qubits: {self.data_qubits}")
        print(f"      Auxiliary qubits: {self.aux_qubits}")
        print(f"      Measurement qubits: {self.measurement_qubits}")
        print(f"      Expected output size: {2**len(self.measurement_qubits)}")
        
        # Initialize parameters
        n_params = len(self.data_qubits) * 3 * config.NUM_LAYERS
        self.register_buffer('_params_init', torch.randn(n_params) * 0.1)
        self.params = nn.Parameter(self._params_init.clone())
    
    def _circuit(self, noise, condition, params):
        """
        Quantum circuit for sub-generator - pure numpy version
        
        Args:
            noise: input noise (numpy array)
            condition: seizure type (int: 0 or 1)
            params: circuit parameters (numpy array)
            
        Returns:
            measurement probabilities
        """
        # Encode condition
        if condition == 1:
            qml.PauliX(wires=self.condition_qubits[0])
        
        # Encode noise on data qubits
        normalized = np.arctan(noise) + np.pi/4
        for i, q in enumerate(self.data_qubits):
            if i < len(normalized):
                qml.RY(float(normalized[i]), wires=q)
        
        # Parameterized layers
        params_per_layer = len(self.data_qubits) * 3
        
        for layer in range(self.config.NUM_LAYERS):
            layer_params = params[
                layer * params_per_layer : (layer + 1) * params_per_layer
            ]
            
            # Rotation gates on data qubits
            for i, q in enumerate(self.data_qubits):
                qml.Rot(
                    float(layer_params[i*3]), 
                    float(layer_params[i*3+1]), 
                    float(layer_params[i*3+2]), 
                    wires=q
                )
            
            # Entanglement (CB topology) on data qubits
            # Layer 1: even pairs
            for i in range(0, len(self.data_qubits)-1, 2):
                qml.CNOT(wires=[self.data_qubits[i], self.data_qubits[i+1]])
            
            # Layer 2: odd pairs
            for i in range(1, len(self.data_qubits)-1, 2):
                qml.CNOT(wires=[self.data_qubits[i], self.data_qubits[i+1]])
        
        # Measure only the measurement qubits (excluding auxiliary)
        return qml.probs(wires=self.measurement_qubits)
    
    def forward(self, noise, condition):
        """
        Generate patch
        
        Args:
            noise: (batch_size, noise_dim) - PyTorch tensor
            condition: (batch_size,) - PyTorch tensor
            
        Returns:
            patch: (batch_size, quantum_patch_size) - PyTorch tensor
        """
        batch_size = noise.shape[0]
        device = noise.device
        
        # Convert params to numpy once
        params_np = self.params.detach().cpu().numpy()
        
        # Create QNode
        qnode = qml.QNode(self._circuit, self.dev, interface='autograd')
        
        # Process each sample in batch
        outputs = []
        for i in range(batch_size):
            # Convert to numpy
            noise_np = noise[i].detach().cpu().numpy()
            condition_int = int(condition[i].item())
            
            # Get probabilities from quantum circuit
            probs = qnode(noise_np, condition_int, params_np)
            
            # Convert back to tensor
            probs_tensor = torch.from_numpy(np.array(probs)).float().to(device)
            outputs.append(probs_tensor)
        
        # Stack outputs
        output = torch.stack(outputs)
        
        return output


class QuantumGenerator(nn.Module):
    """
    Complete Quantum Generator with patch method
    """
    
    def __init__(self, config: Config = Config):
        super().__init__()
        self.config = config
        
        # Calculate actual quantum output size
        # It's 2^(NUM_QUBITS - NUM_AUXILIARY_QUBITS)
        actual_quantum_output = 2 ** (config.NUM_QUBITS - config.NUM_AUXILIARY_QUBITS)
        
        print(f"  Quantum output size per patch: {actual_quantum_output}")
        print(f"  Target patch size: {config.PATCH_SIZE}")
        print(f"  Creating {config.NUM_PATCHES} sub-generators...")
        
        # Create sub-generators
        self.sub_generators = nn.ModuleList([
            QuantumSubGenerator(config, i) 
            for i in range(config.NUM_PATCHES)
        ])
        
        # Classical post-processing layers
        self.post_process = nn.ModuleList([
            nn.Sequential(
                nn.Linear(actual_quantum_output, config.PATCH_SIZE // 2),
                nn.ReLU(),
                nn.Linear(config.PATCH_SIZE // 2, config.PATCH_SIZE),
                nn.Tanh()
            )
            for _ in range(config.NUM_PATCHES)
        ])
    
    def forward(self, noise, condition):
        """
        Generate complete EEG segment
        
        Args:
            noise: (batch_size, noise_dim)
            condition: (batch_size,) - 0 for normal, 1 for seizure
            
        Returns:
            eeg: (batch_size, n_channels, segment_length)
        """
        batch_size = noise.shape[0]
        device = noise.device
        
        # Generate each patch
        patches = []
        
        for idx, sub_gen in enumerate(self.sub_generators):
            # Sample noise for this patch
            patch_noise = noise[:, :self.config.NUM_QUBITS]
            
            # Generate quantum patch
            quantum_patch = sub_gen(patch_noise, condition)
            
            # Post-process
            patch = self.post_process[idx](quantum_patch)
            
            patches.append(patch)
        
        # Concatenate patches
        signal = torch.cat(patches, dim=1)  # (batch_size, segment_length)
        
        # Expand to multi-channel
        signal = signal.unsqueeze(1).repeat(1, self.config.NUM_CHANNELS, 1)
        
        return signal


if __name__ == "__main__":
    # Test generator
    print("Testing Quantum Generator")
    print("=" * 50)
    
    Config.set_seed()
    
    generator = QuantumGenerator()
    
    # Test forward pass
    batch_size = 2
    noise = torch.randn(batch_size, Config.NUM_QUBITS)
    condition = torch.LongTensor([0, 1])
    
    print(f"\nInput noise shape: {noise.shape}")
    print(f"Condition: {condition}")
    
    output = generator(noise, condition)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("\n✓ Generator working!")