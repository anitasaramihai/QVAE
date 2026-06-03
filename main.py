
import os  # paths and folders
import time  # measures training time
import numpy as np  # numerical operations
import torch  # works with tensors
import torch.optim as optim  # provides optimizers
from torch.utils.data import DataLoader, TensorDataset  # prepare data for training
from sklearn.preprocessing import StandardScaler
from models import HybridVAE, vae_loss
from utils import read_dataset, preprocessing


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

