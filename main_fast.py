import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models_fast import FastVAE, vae_loss
from utils import read_dataset, preprocessing


def main():
    """
    VERSIUNE RAPIDĂ - fără circuit cuantic
    Folosit pentru testare și benchmark
    """
   
    dataset_name = "O3"
    model_name = "FastVAE"

    latent_size = 4
    hidden_size = 128
    batch_size = 256
    nr_epochs = 30
    learning_rate = 1e-4
    kl_weight = 0.75

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    path = os.getcwd() 

    folder_results = os.path.join(path, "Results")
    folder_model = os.path.join(folder_results, model_name)

    os.makedirs(folder_results, exist_ok=True)
    os.makedirs(folder_model, exist_ok=True)

    df = read_dataset(dataset_name)
    df = preprocessing(df)
    df_train = df["2023-01-01":"2023-06-30"].copy()
    df_test = df["2024-01-01":"2024-12-31"].copy()
   
    print("Training data shape:", df_train.shape)
    print("Testing data shape:", df_test.shape)

    if df_train.empty:
        raise ValueError("Training set is empty.")

    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(df_train)

    input_dim = df_train.shape[1]

   
    vae = FastVAE(
        input_size=input_dim,
        hidden_size=hidden_size,
        latent_size=latent_size
    ).to(device)

    optimizer = optim.Adam(
        vae.parameters(),
        lr=learning_rate
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=nr_epochs, eta_min=1e-6)

    train_tensor = torch.FloatTensor(scaled_train_data).to(device)
    train_dataset = TensorDataset(train_tensor)

    num_workers = 0 if device.type == "cuda" else 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

  
    train_losses = []  
    recon_losses = []
    kl_losses = []

    vae.train()
    start_time = time.time()

    print("=" * 60)
    print("FAST VAE TRAINING (NO QUANTUM CIRCUIT)")
    print("=" * 60 + "\n")

    for epoch in range(nr_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        epoch_start = time.time()

        for (data,) in train_loader:
            optimizer.zero_grad()

            mu_x, logvar_x, mu_z, logvar_z = vae(data)

            loss, recon, kl = vae_loss(
                data,
                mu_x,
                logvar_x,
                mu_z,
                logvar_z,
                kl_weight
            )
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1:2d}/{nr_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f} | "
            f"Time: {epoch_time:.2f}s | "
            f"LR: {current_lr:.2e}"
        )

    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Average per epoch: {training_time/nr_epochs:.2f} seconds")
    print(f"{'='*60}\n")
   
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    model_path = os.path.join(folder_model, "fastvae_model.pth")
    scaler_path = os.path.join(folder_model, "scaler.npy")

    torch.save(vae.state_dict(), model_path)
    np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Plot training curves
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, linewidth=2)
    plt.title("Total Loss per Epoch (FastVAE)", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label="Reconstruction Loss", linewidth=2, color="blue")
    plt.title("Reconstruction Loss per Epoch (FastVAE)", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, label="KL Loss", linewidth=2, color="orange")
    plt.title("KL Loss per Epoch (FastVAE)", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_curves_path = os.path.join(folder_model, "training_curves.png")
    plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {training_curves_path}")
    plt.close()


if __name__ == "__main__":
    main()
