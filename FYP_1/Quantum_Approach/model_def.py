import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as np

class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.num_qubits = 4
        self.weights = nn.Parameter(torch.randn(3, self.num_qubits))
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface='torch')

    def quantum_circuit(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(self.num_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(self.num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x):
        print("Forward pass in QuantumLayer")
        print("Input shape:", x.shape)
        print("Weights shape:", self.weights.shape)
        quantum_outputs = [torch.tensor(self.qnode(x[i], self.weights)).float() for i in range(x.size(0))]
        output = torch.stack(quantum_outputs)
        print("Output shape from QuantumLayer:", output.shape)
        return output


class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.pre_quantum = nn.Linear(4*9*9, 4)
        self.dropout = nn.Dropout(0.5)
        self.quantum_layer = QuantumLayer()
        self.post_quantum = nn.Linear(4, 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.pre_quantum(x)
        x = self.dropout(x)
        x = self.quantum_layer(x)
        x = self.post_quantum(x)
        return torch.log_softmax(x, dim=1)
