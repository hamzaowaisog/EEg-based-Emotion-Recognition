import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# Define the quantum device
n_qubits = 10  # Based on PCA-reduced features
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Initialize QNode with torch interface
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

# Define quantum-classical hybrid model
weight_shapes = {"weights": (3, n_qubits, 3)}  # Three layers of weights
classifier = qml.qnn.TorchLayer(qnode, weight_shapes)

# Define the HybridClassifier
class HybridClassifier(nn.Module):
    def __init__(self):
        super(HybridClassifier, self).__init__()
        self.quantum_layer = classifier
        self.fc1 = nn.Linear(n_qubits, 32)
        self.fc2 = nn.Linear(32, 3)  # Assuming 3 emotion classes: Positive, Neutral, Negative

    def forward(self, x):
        x = self.quantum_layer(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
