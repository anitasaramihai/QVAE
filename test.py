# `test.py` — Evaluate the Trained Hybrid VAE

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pennylane as qml

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# Quantum configuration
# ============================================================

N_QUBITS = 4
N_Q_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


weight_shapes = {
    "weights": (N_Q_LAYERS, N_QUBITS, 3)
}


# ============================================================
# Hybrid VAE model (same architecture as in main.py)
# ============================================================

class HybridVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=4):
        super(HybridVAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_to_quantum = nn.Linear(hidden_dim, N_QUBITS)

        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes
        )

        self.fc2_mu = nn.Linear(N_QUBITS, latent_dim)
        self.fc2_logvar = nn.Linear(N_QUBITS, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4_mu = nn.Linear(hidden_dim, input_dim)
        self.fc4_logvar = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))

        q_input = self.fc_to_quantum(h)
        q_input = torch.tanh(q_input) * np.pi

        q_output = torch.stack(
            [self.quantum_layer(sample) for sample in q_input]
        )

        mu_z = self.fc2_mu(q_output)
        logvar_z = self.fc2_logvar(q_output)

        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):
        # For evaluation we use the mean directly for deterministic output
        return mu

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        mu_x = self.fc4_mu(h)
        logvar_x = self.fc4_logvar(h)
        return mu_x, logvar_x

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu_z, logvar_z


# ============================================================
# Data functions
# ============================================================

def preprocessing(df):
    df = df.dropna()
    df = df.drop_duplicates(subset=["timestamp"])

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%dT%H:%M"
    )

    df = df.set_index("timestamp")

    return df


def read_dataset(name):
    path = os.getcwd()

    if name == "O3":
        file_path = os.path.join(path, "Ref-Data", "O3_all.csv")
    elif name == "NO2":
        file_path = os.path.join(path, "Ref-Data", "NO2_all.csv")
    elif name == "NO2_O3":
        file_path = os.path.join(path, "Ref-Data", "NO2_O3_all.csv")
    else:
        raise ValueError("Dataset name must be: O3, NO2, or NO2_O3")

    return pd.read_csv(file_path, sep=";", header=0)


# ============================================================
# Main evaluation function
# ============================================================

def main():
    dataset_name = "O3"      # Same dataset used in training
    hidden_dim = 128
    latent_dim = 4

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = read_dataset(dataset_name)
    df = preprocessing(df)

    df_train = df["2023-01-01":"2023-12-31"].copy()
    df_test = df["2024-01-01":"2024-12-31"].copy()

    input_dim = df_train.shape[1]

    print("Training shape:", df_train.shape)
    print("Testing shape:", df_test.shape)

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------
    scaler_path = os.path.join(
        os.getcwd(),
        "Results",
        "HybridVAE",
        "scaler.npy"
    )

    scaler_data = np.load(scaler_path, allow_pickle=True).item()

    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = input_dim

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------
    x_test_scaled = scaler.transform(df_test)
    test_tensor = torch.FloatTensor(x_test_scaled)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model_path = os.path.join(
        os.getcwd(),
        "Results",
        "HybridVAE",
        "hybrid_vae_model.pth"
    )

    model = HybridVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()

    # --------------------------------------------------------
    # Reconstruct test data
    # --------------------------------------------------------
    with torch.no_grad():
        mu_x, _, _, _ = model(test_tensor)

    reconstructed_scaled = mu_x.numpy()

    # Convert back to original scale
    reconstructed = scaler.inverse_transform(reconstructed_scaled)
    original = df_test.values

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    mse = mean_squared_error(original, reconstructed)
    mae = mean_absolute_error(original, reconstructed)
    rmse = np.sqrt(mse)

    print(f"\nTest MSE:  {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE:  {mae:.6f}")

    # --------------------------------------------------------
    # Plot first variable
    # --------------------------------------------------------
    column_index = 0
    column_name = df_test.columns[column_index]

    plt.figure(figsize=(12, 5))
    plt.plot(
        original[:200, column_index],
        label="Original"
    )
    plt.plot(
        reconstructed[:200, column_index],
        label="Reconstructed"
    )

    plt.title(f"Hybrid VAE Reconstruction - {column_name}")
    plt.xlabel("Time step")
    plt.ylabel(column_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()