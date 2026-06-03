import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# Quantum circuit configuration
nr_qubits = 4
nr_layers = 2

dev = qml.device("default.qubit", wires=nr_qubits)  # default.qubit is a PennyLane simulator with 4 qubits

@qml.qnode(dev, interface="torch")  # quantum circuit with inputs (classical values coming from the neural network)
def quantum_circuit(inputs, weights):  # and weights (trainable quantum parameters)
    
    for i in range(nr_qubits):  # encodes the classical data into the quantum circuit
        qml.RY(inputs[i], wires=i)  # Each input value is used as an angle for an RY rotation gate

    qml.templates.StronglyEntanglingLayers(  # the circuit receives 4 values because i have 4 qubits
        weights,  # add trainable quantum layers
        wires=range(nr_qubits)
    )
    # returns 4 classical values and each value is the expectation value of a Pauli-Z measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(nr_qubits)]  # (receives 4 numbers and returns 4 numbers)

weight_shapes = {  # the shape of the trainable quantum parameters
    "weights": (nr_layers, nr_qubits, 3)  # i have 2 layers, 4 qubits and 3 parameters
}


class HybridVAE(nn.Module):  # classical encoder, quantum layer, latent space, classical decoder
  
    def __init__(self, input_size, hidden_size=128, latent_size=4):  # defines layers of the model
        super(HybridVAE, self).__init__()  # 128 size of the hidden classical layer, 4 is the size of the latent representation

        self.fc1 = nn.Linear(input_size, hidden_size)  # transforms the input from 9 features to 128 hidden values
        # quantum circuit needs exactly 4 inputs because it has 4 qubits (128->4)
        self.fc_to_quantum = nn.Linear(hidden_size, nr_qubits)

        self.quantum_layer = qml.qnn.TorchLayer(  # converts the PennyLane circuit to torch
            quantum_circuit,  # it can be trained together with the nn classical
            weight_shapes
        )
        # latent distribution, it produces: mean of the latent distribution, log-variance of the latent distribution
        self.fc2_mu = nn.Linear(nr_qubits, latent_size)
        self.fc2_logvar = nn.Linear(nr_qubits, latent_size)
        # decoder reconstructs the original signal, transforms: latent space, hidden layer, reconstructed input
        self.fc3 = nn.Linear(latent_size, hidden_size)  # 4 → 128 → 9
        self.fc4_mu = nn.Linear(hidden_size, input_size)
        self.fc4_logvar = nn.Linear(hidden_size, input_size)

    def encode(self, x):  # encoder function
        
        h = torch.relu(self.fc1(x))  # input is passed through the first classical layer and ReLU activation
        q_input = self.fc_to_quantum(h)  # hidden representation is reduced to 4 values
        q_input = torch.tanh(q_input) * np.pi  # keeps the quantum input values in a stable range[-pi,+pi]
        q_output = torch.stack(  # because quantum rotation gates use angles
            [self.quantum_layer(sample) for sample in q_input]  # applies the quantum circuit to each sample
        )

        mu_z = self.fc2_mu(q_output)  # quantum layer returns 4 values for each sample
        logvar_z = self.fc2_logvar(q_output)  # output is transformed into the latent distribution parameters

        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):  # samples a latent vector z from the latent distribution
       
        std = torch.exp(0.5 * logvar)  # computes the standard deviation from the log-variance
        eps = torch.randn_like(std)  # creates random noise with the same shape as std
        return mu + eps * std  # allows the VAE to sample from a distribution while still being trainable with backpropagation

    def decode(self, z):  # reconstructs the input from the latent variable
       
        h = torch.relu(self.fc3(z))  # latent vector is passed through a classical hidden layer

        mu_x = self.fc4_mu(h)     
        logvar_x = self.fc4_logvar(h)

        return mu_x, logvar_x  # decoder outputs: reconstructed signal, reconstruction uncertainty

    def forward(self, x):  # defines the full forward pass of the model
        mu_z, logvar_z = self.encode(x)  # input is encoded into a latent distribution
        z = self.reparameterize(mu_z, logvar_z)  # A latent vector is sampled
        mu_x, logvar_x = self.decode(z)  # latent vector is decoded into a reconstruction

        return mu_x, logvar_x, mu_z, logvar_z


def vae_loss(x, mu_x, logvar_x, mu_z, logvar_z, beta=1.0):  # loss has two parts: total loss = reconstruction loss + beta * KL loss
   
    recon_loss = 0.5 * torch.sum(  # measures how well the model reconstructs the input
        logvar_x  # the decoder outputs both mu_x and logvar_x, this is a Gaussian negative log-likelihood loss
        + ((x - mu_x) ** 2) / torch.exp(logvar_x)  # It does not only compare x and mu_x; it also uses the uncertainty logvar_x
        + np.log(2 * np.pi),
        dim=1
    )

    kl_loss = -0.5 * torch.sum(  # This forces the latent space to stay close to a standard normal distribution
        1 + logvar_z - mu_z.pow(2) - logvar_z.exp(),
        dim=1
    )

    total_loss = torch.mean(recon_loss + beta * kl_loss)

    return (
        total_loss,
        torch.mean(recon_loss),
        torch.mean(kl_loss)
    )
