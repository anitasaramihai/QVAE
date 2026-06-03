from main import quantum_circuit
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np

sample_input = np.random.rand(4)
sample_weights = np.random.rand(2, 4, 3)

fig, ax = qml.draw_mpl(quantum_circuit)(sample_input, sample_weights)
plt.savefig("quantum_circuit.png", dpi=300, bbox_inches="tight")
plt.show()