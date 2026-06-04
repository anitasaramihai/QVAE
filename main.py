
import os  # paths and folders
import time  # measures training time
import numpy as np  # numerical operations
import torch  # works with tensors
import torch.optim as optim  # provides optimizers
from torch.utils.data import DataLoader, TensorDataset  # prepare data for training
from torch.optim.lr_scheduler import CosineAnnealingLR  # OPTIMIZARE: Learning rate scheduler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models import HybridVAE, vae_loss
from utils import read_dataset, preprocessing


def main():
   
    dataset_name = "O3"      # Change to "NO2" or "NO2_O3"
    model_name = "HybridVAE"

    latent_size = 4 #latent representation size
    hidden_size = 128 #OPTIMIZARE: Redus de la 192 la 128 (2x mai rapid)
    batch_size = 256       # OPTIMIZARE: Mărit de la 128 la 256 (procesare mai eficientă)
    nr_epochs = 30         # OPTIMIZARE: Redus de la 50 la 30 (testare rapidă)
    learning_rate = 1e-4  #balanced optimizer step size
    kl_weight = 0.75 #balanced KL weight

    # OPTIMIZARE: Detectează GPU dacă este disponibil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    path = os.getcwd() 

    folder_results = os.path.join(path, "Results")
    folder_model = os.path.join(folder_results, model_name)
    folder_masks = os.path.join(folder_results, "Masks")

    os.makedirs(folder_results, exist_ok=True) #code creates folders for saving results
    os.makedirs(folder_model, exist_ok=True)
    os.makedirs(folder_masks, exist_ok=True)

    df = read_dataset(dataset_name) #The dataset is loaded and cleaned
    df = preprocessing(df)
    df_train = df["2023-01-01":"2023-06-30"].copy() # # Training period: First 6 months of 2023 for balanced training
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
    ).to(device)  # OPTIMIZARE: Mută modelul pe device (GPU/CPU)

    optimizer = optim.Adam( #Adam is used to update all trainable parameters:classical neural network weights
        vae.parameters(),     # and quantum circuit parameters
        lr=learning_rate
    )

    # OPTIMIZARE: Learning rate scheduler care decrementează LR pe parcursul training-ului
    scheduler = CosineAnnealingLR(optimizer, T_max=nr_epochs, eta_min=1e-6)

    train_tensor = torch.FloatTensor(scaled_train_data).to(device)  # OPTIMIZARE: Mută tensorii pe device
    train_dataset = TensorDataset(train_tensor)

    # OPTIMIZARE: num_workers pentru parallelizare (îl setezi depinde de CPU cores - 0 dacă pe GPU)
    num_workers = 0 if device.type == "cuda" else 2
    train_loader = DataLoader(     
        train_dataset,                 
        batch_size=batch_size,
        shuffle=True,              # means the data is randomly shuffled at each epoch
        num_workers=num_workers,   # OPTIMIZARE: Paralelizare încărcare date
        pin_memory=(device.type == "cuda")  # OPTIMIZARE: Dacă e GPU, pin memory
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
            # data e deja pe device din DataLoader
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

        # OPTIMIZARE: Step learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch + 1}/{nr_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f} | "
            f"LR: {current_lr:.2e}"  # Arată learning rate-ul curent
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

    # Plot training curves for monitoring
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, linewidth=2)
    plt.title("Total Loss per Epoch", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label="Reconstruction Loss", linewidth=2, color="blue")
    plt.title("Reconstruction Loss per Epoch", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, label="KL Loss", linewidth=2, color="orange")
    plt.title("KL Loss per Epoch", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_curves_path = os.path.join(folder_model, "training_curves.png")
    plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {training_curves_path}")
    plt.show()


if __name__ == "__main__":
    main()

