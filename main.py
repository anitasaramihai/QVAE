
import os #paths and folders
import time #measures training time
import numpy as np #numerical operations
import pandas as pd #reads and processes CSV files
import pennylane as qml  
import torch #works with tensors
import torch.nn as nn
import torch.optim as optim #provides optimizers
from torch.utils.data import DataLoader, TensorDataset #prepare data for training
from sklearn.preprocessing import StandardScaler

nr_qubits  = 4   #quantum circuit uses 4 qubits and 2 quantum layers
nr_layers = 2

dev = qml.device("default.qubit", wires=nr_qubits )  #default.qubit is a PennyLane simulator with 4 qubits 


@qml.qnode(dev, interface="torch")  #quantum circuit with inputs(classical values coming from the neural network)
def quantum_circuit(inputs, weights): #amd weights(trainable quantum parameters)
   
    for i in range(nr_qubits ):  #encodes the classical data into the quantum circuit
        qml.RY(inputs[i], wires=i)  #Each input value is used as an angle for an RY rotation gate

   
    qml.templates.StronglyEntanglingLayers( #the circuit receives 4 values because i have 4 qubits
        weights,           # add trainable quantum layers
        wires=range(nr_qubits )
    )
   #returns 4 classical values and each value is the expectation value of a Pauli-Z measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(nr_qubits )]  #(receives 4 numbers and returns 4 numbers)

weight_shapes = {    #the shape of the trainable quantum parameters
    "weights": (nr_layers, nr_qubits , 3)  # i have 2 layers, 4 qubits and 3 parameters
}

class HybridVAE(nn.Module): #classical encoder,quantum layer,latent space,classical decoder 
  
    def __init__(self, input_size, hidden_size=128, latent_size=4): # defines layers of the model
        super(HybridVAE, self).__init__()  #128 size of the hidden classical layer, 4 is the size of the latent representation

        self.fc1 = nn.Linear(input_size, hidden_size) #transforms the input from 9 features to 128 hidden values
       #quantum circuit needs exactly 4 inputs because it has 4 qubits (128->4)
        self.fc_to_quantum = nn.Linear(hidden_size, nr_qubits )

        self.quantum_layer = qml.qnn.TorchLayer(   #converts the PennyLane circuit to torch 
            quantum_circuit,      #it can be trained together with the nn clasical 
            weight_shapes
        )
             #latent distribution,it produces: mean of the latent distribution,log-variance of the latent distribution
        self.fc2_mu = nn.Linear(nr_qubits , latent_size)
        self.fc2_logvar = nn.Linear(nr_qubits , latent_size)
             #decoder reconstructs the original signal, transforms:latent space,hidden layer,reconstructed input
        self.fc3 = nn.Linear(latent_size, hidden_size)     #4 → 128 → 9
        self.fc4_mu = nn.Linear(hidden_size, input_size)
        self.fc4_logvar = nn.Linear(hidden_size, input_size)

    def encode(self, x):   #enceder function 
        
        h = torch.relu(self.fc1(x))   #input is passed through the first classical layer and ReLU activation
        q_input = self.fc_to_quantum(h)  #hidden representation is reduced to 4 values
        q_input = torch.tanh(q_input) * np.pi  #keeps the quantum input values in a stable range[-pi,+pi]
        q_output = torch.stack(             #because quantum rotation gates use angles
            [self.quantum_layer(sample) for sample in q_input]   #applies the quantum circuit to each sample
        )

        mu_z = self.fc2_mu(q_output)  #quantum layer returns 4 values for each sample
        logvar_z = self.fc2_logvar(q_output)  #output is transformed into the latent distribution parameters

        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):    #samples a latent vector z from the latent distribution
       
        std = torch.exp(0.5 * logvar)      #computes the standard deviation from the log-variance
        eps = torch.randn_like(std)        #creates random noise with the same shape as std
        return mu + eps * std # allows the VAE to sample from a distribution while still being trainable with backpropagation

    def decode(self, z):   #reconstructs the input from the latent variable
       
        h = torch.relu(self.fc3(z))   # latent vector is passed through a classical hidden layer

        mu_x = self.fc4_mu(h)     
        logvar_x = self.fc4_logvar(h)

        return mu_x, logvar_x  #decoder outputs: reconstructed signal, reconstruction uncertainty

    def forward(self, x):    #defines the full forward pass of the model
        mu_z, logvar_z = self.encode(x)  #input is encoded into a latent distribution
        z = self.reparameterize(mu_z, logvar_z)  #A latent vector is sampled
        mu_x, logvar_x = self.decode(z) #latent vector is decoded into a reconstruction

        return mu_x, logvar_x, mu_z, logvar_z


def vae_loss(x, mu_x, logvar_x, mu_z, logvar_z, beta=1.0):  #loss has two parts:total loss = reconstruction loss + beta * KL loss
   
    recon_loss = 0.5 * torch.sum( #measures how well the model reconstructs the input
        logvar_x    #the decoder outputs both mu_x and logvar_x, this is a Gaussian negative log-likelihood loss
        + ((x - mu_x) ** 2) / torch.exp(logvar_x)  #It does not only compare x and mu_x; it also uses the uncertainty logvar_x
        + np.log(2 * np.pi),
        dim=1
    )

    kl_loss = -0.5 * torch.sum(        #This forces the latent space to stay close to a standard normal distribution
        1 + logvar_z - mu_z.pow(2) - logvar_z.exp(),
        dim=1
    )

    total_loss = torch.mean(recon_loss + beta * kl_loss)

    return (
        total_loss,
        torch.mean(recon_loss),
        torch.mean(kl_loss)
    )

def preprocessing(df): # this function cleans the dataset
    
    n_before = df.shape[0]  #Stores the number of rows before cleaning

    df = df.dropna()  # Removes rows with missing values
    df = df.drop_duplicates(subset=["timestamp"])  #Removes duplicate timestamps

    n_after = df.shape[0]  # 
    print(f"Size before: {n_before} | Size after: {n_after}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%dT%H:%M"     #Converts the timestamp column to datetime format
    )

    df = df.set_index("timestamp")#Uses the timestamp as the dataframe index

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
        raise ValueError(
            "Dataset name must be: O3, NO2, or NO2_O3"
        )

    df = pd.read_csv(file_path, sep=";", header=0)

    return df


def main():
   
    dataset_name = "O3"      # Change to "NO2" or "NO2_O3"
    model_name = "HybridVAE"

    latent_size = 4 #latent representation size
    hidden_size = 128 #hidden layer size
    batch_size = 128       
    nr_epochs = 100         # Increase later 
    learning_rate = 1e-3  #optimizer step size
    kl_weight = 1.0 #weight of the KL loss

    path = os.getcwd() 

    folder_results = os.path.join(path, "Results")
    folder_model = os.path.join(folder_results, model_name)
    folder_masks = os.path.join(folder_results, "Masks")

    os.makedirs(folder_results, exist_ok=True) #code creates folders for saving results
    os.makedirs(folder_model, exist_ok=True)
    os.makedirs(folder_masks, exist_ok=True)

    df = read_dataset(dataset_name) #The dataset is loaded and cleaned
    df = preprocessing(df)
    df_train = df["2023-01-01":"2023-12-31"].copy() # # Training period: 2023
    df_test = df["2024-01-01":"2024-12-31"].copy() ## Testing period: 2024
   
    print("Training data shape:", df_train.shape)
    print("Testing data shape:", df_test.shape)

    if df_train.empty:
        raise ValueError("Training set is empty.")

    scaler = StandardScaler()  #training data is standardized, means each feature is transformed to have approximately:
                                                       #mean = 0,standard deviation = 1
    scaled_train_data = scaler.fit_transform(df_train)

    input_dim = df_train.shape[1] #This gets the number of input features, in my case 9 

   
    vae = HybridVAE(  #model (9,128,4 quantum inputs,quantum circuit, 4 latent values, 128,9)
        input_size=input_dim,
        hidden_size=hidden_size,
        latent_size=latent_size
    )

    optimizer = optim.Adam( #Adam is used to update all trainable parameters:classical neural network weights
        vae.parameters(),     # and quantum circuit parameters
        lr=learning_rate
    )

    train_tensor = torch.FloatTensor(scaled_train_data)   #The training data is converted into PyTorch tensors
    train_dataset = TensorDataset(train_tensor)

    train_loader = DataLoader(     #data is divided into batches of 8 samples
        train_dataset,                 
        batch_size=batch_size,
        shuffle=True               # means the data is randomly shuffled at each epoch
    )

  
    train_losses = []  
    recon_losses = []
    kl_losses = []

    vae.train()
    start_time = time.time()

    for epoch in range(nr_epochs):  #model trains for 5 epochs
        epoch_loss = 0.0      #initialization 
        epoch_recon = 0.0
        epoch_kl = 0.0

        for (data,) in train_loader:  #for each batch the model processes one batch at a time
            optimizer.zero_grad() #clears old gradients

            mu_x, logvar_x, mu_z, logvar_z = vae(data) #The batch is passed through the hybrid VAE

            loss, recon, kl = vae_loss(  #The loss is computed
                data,
                mu_x,
                logvar_x,
                mu_z,
                logvar_z,
                kl_weight
            )

            loss.backward() #Gradients are calculated
            optimizer.step() #model parameters are updated

            epoch_loss += loss.item() #stores the loss value for this batch
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        avg_loss = epoch_loss / len(train_loader)  #average loss per epoch
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        print(
            f"Epoch [{epoch + 1}/{nr_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f}"
        )

    training_time = time.time() - start_time #calculates how long training took
    print(f"\nTraining completed in {training_time:.2f} seconds") 
   
    vae.eval()  #freezinf the model 
    for param in vae.parameters(): #freezes the parameters, so they are no longer updated
        param.requires_grad = False  #ready for testing 

    print("Hybrid VAE is now frozen and ready for testing.")

    model_path = os.path.join( #saving the model 
        folder_model,
        "hybrid_vae_model.pth"
    )

    scaler_path = os.path.join( #saving the scaler
        folder_model,
        "scaler.npy"
    )

    torch.save(vae.state_dict(), model_path)

    np.save(
        scaler_path,
        {
            "mean": scaler.mean_,
            "scale": scaler.scale_
        }
    )

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


if __name__ == "__main__":
    main()

