import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from models import HybridVAE
from utils import read_dataset, preprocessing


def main():
    dataset_name = "O3"
    hidden_size = 192
    latent_size = 4
    batch_size = 128  # OPTIMIZARE: Batch processing pentru test

    # OPTIMIZARE: Detectează device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load and prepare data
    df = read_dataset(dataset_name)
    df = preprocessing(df)
    df_train = df["2023-01-01":"2023-12-31"].copy()
    df_test = df["2024-01-01":"2024-12-31"].copy()
    input_dim = df_train.shape[1]

    print("Training shape:", df_train.shape)
    print("Testing shape:", df_test.shape)

    # Load scaler
    scaler_path = os.path.join(os.getcwd(), "Results", "HybridVAE", "scaler.npy")
    scaler_data = np.load(scaler_path, allow_pickle=True).item()

    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = input_dim

    # Prepare test data
    x_test_scaled = scaler.transform(df_test)
    test_tensor = torch.FloatTensor(x_test_scaled).to(device)  # OPTIMIZARE: Mută pe device
    test_dataset = TensorDataset(test_tensor)

    # OPTIMIZARE: DataLoader pentru batch processing
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type == "cuda" else 2,
        pin_memory=(device.type == "cuda")
    )

    # Load model
    model_path = os.path.join(os.getcwd(), "Results", "HybridVAE", "hybrid_vae_model.pth")
    model = HybridVAE(input_size=input_dim, hidden_size=hidden_size, latent_size=latent_size)
    model = model.to(device)  # OPTIMIZARE: Mută modelul pe device

    state_dict = torch.load(model_path, map_location=device)  # OPTIMIZARE: Load pe device
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.\n")

    # OPTIMIZARE: Batch reconstruction
    start_time = time.time()
    reconstructed_list = []

    print("Running reconstruction (batch processing)...")
    with torch.no_grad():
        for batch_data in test_loader:
            # batch_data e o tuple cu 1 element din DataLoader
            batch_tensor = batch_data[0]
            mu_x, _, _, _ = model(batch_tensor)
            reconstructed_list.append(mu_x.cpu().numpy())

    reconstruction_time = time.time() - start_time
    print(f"Reconstruction completed in {reconstruction_time:.2f} seconds\n")

    # Combine all batches
    reconstructed_scaled = np.vstack(reconstructed_list)
    reconstructed = scaler.inverse_transform(reconstructed_scaled)
    original = df_test.values

    # Metrics
    mse = mean_squared_error(original, reconstructed)
    mae = mean_absolute_error(original, reconstructed)
    rmse = np.sqrt(mse)

    print(f"Test MSE:  {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE:  {mae:.6f}\n")

    # OPTIMIZARE: Salvare rezultate
    results_folder = os.path.join(os.getcwd(), "Results", "HybridVAE")
    
    # 1. Salvare metrici în CSV
    metrics_data = {
        "Metric": ["MSE", "RMSE", "MAE"],
        "Value": [mse, rmse, mae],
        "Dataset": [dataset_name] * 3,
        "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")] * 3,
        "Model": ["HybridVAE"] * 3
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_folder, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to: {metrics_path}")

    # 2. Salvare predictions vs original (pentru analiză)
    predictions_data = {
        "timestamp": df_test.index
    }
    
    # Adaug coloane pentru fiecare feature: original + reconstructed
    for i, col in enumerate(df_test.columns):
        predictions_data[f"{col}_original"] = original[:, i]
        predictions_data[f"{col}_reconstructed"] = reconstructed[:, i]
        predictions_data[f"{col}_error"] = original[:, i] - reconstructed[:, i]
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_path = os.path.join(results_folder, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to: {predictions_path}")

    # 3. Salvare error statistics
    errors = original - reconstructed
    error_stats = {
        "Feature": df_test.columns,
        "Mean_Error": [errors[:, i].mean() for i in range(errors.shape[1])],
        "Std_Error": [errors[:, i].std() for i in range(errors.shape[1])],
        "Min_Error": [errors[:, i].min() for i in range(errors.shape[1])],
        "Max_Error": [errors[:, i].max() for i in range(errors.shape[1])],
        "RMSE": [np.sqrt(np.mean(errors[:, i]**2)) for i in range(errors.shape[1])],
        "MAE": [np.mean(np.abs(errors[:, i])) for i in range(errors.shape[1])]
    }
    error_df = pd.DataFrame(error_stats)
    error_path = os.path.join(results_folder, "error_statistics.csv")
    error_df.to_csv(error_path, index=False)
    print(f"✓ Error statistics saved to: {error_path}")

    # OPTIMIZARE: Grafice mai detaliate cu salvare
    print("\nGenerating plots...\n")

    # Plot 1: All features reconstruction
    fig, axes = plt.subplots(len(df_test.columns), 1, figsize=(14, 12))
    if len(df_test.columns) == 1:
        axes = [axes]

    for i, col in enumerate(df_test.columns):
        axes[i].plot(original[:, i], label="Original", linewidth=2, alpha=0.7)
        axes[i].plot(reconstructed[:, i], label="Reconstructed", linewidth=2, alpha=0.7)
        axes[i].set_title(f"Reconstruction - {col}")
        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = os.path.join(results_folder, "reconstruction_all_features.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot 1 saved: {plot1_path}")
    plt.close()

    # Plot 2: Reconstruction error per feature
    fig, axes = plt.subplots(len(df_test.columns), 1, figsize=(14, 12))
    if len(df_test.columns) == 1:
        axes = [axes]

    for i, col in enumerate(df_test.columns):
        error = errors[:, i]
        axes[i].plot(error, label="Error", linewidth=1.5, color="red", alpha=0.7)
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[i].fill_between(range(len(error)), error, alpha=0.3, color="red")
        axes[i].set_title(f"Reconstruction Error - {col}")
        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Error")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plot2_path = os.path.join(results_folder, "reconstruction_errors.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot 2 saved: {plot2_path}")
    plt.close()

    # Plot 3: Metrics summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Error distribution
    all_errors = errors.flatten()
    axes[0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title("Error Distribution")
    axes[0].set_xlabel("Error Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # RMSE per feature
    rmse_per_feature = [np.sqrt(np.mean(errors[:, i]**2)) for i in range(errors.shape[1])]
    axes[1].bar(range(len(df_test.columns)), rmse_per_feature, color='steelblue', alpha=0.7)
    axes[1].set_title("RMSE per Feature")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(range(len(df_test.columns)))
    axes[1].set_xticklabels(df_test.columns, rotation=45)
    axes[1].grid(True, alpha=0.3)

    # MAE per feature
    mae_per_feature = [np.mean(np.abs(errors[:, i])) for i in range(errors.shape[1])]
    axes[2].bar(range(len(df_test.columns)), mae_per_feature, color='orange', alpha=0.7)
    axes[2].set_title("MAE per Feature")
    axes[2].set_xlabel("Feature")
    axes[2].set_ylabel("MAE")
    axes[2].set_xticks(range(len(df_test.columns)))
    axes[2].set_xticklabels(df_test.columns, rotation=45)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot3_path = os.path.join(results_folder, "metrics_summary.png")
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {plot3_path}")
    plt.close()

    # Plot 4: Time series comparison (first 500 points)
    column_index = 0
    column_name = df_test.columns[column_index]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full time series
    axes[0].plot(original[:, column_index], label="Original", linewidth=2, alpha=0.7)
    axes[0].plot(reconstructed[:, column_index], label="Reconstructed", linewidth=2, alpha=0.7)
    axes[0].set_title(f"Full Time Series - {column_name}")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed (first 300 points)
    axes[1].plot(original[:300, column_index], label="Original", linewidth=2, alpha=0.7)
    axes[1].plot(reconstructed[:300, column_index], label="Reconstructed", linewidth=2, alpha=0.7)
    axes[1].set_title(f"Zoomed View (first 300 steps) - {column_name}")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot4_path = os.path.join(results_folder, "detailed_comparison.png")
    plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot 4 saved: {plot4_path}")
    plt.close()

    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll results saved in: {results_folder}")
    print(f"  - test_metrics.csv (overall metrics)")
    print(f"  - test_predictions.csv (all predictions & errors)")
    print(f"  - error_statistics.csv (per-feature error stats)")
    print(f"  - reconstruction_all_features.png")
    print(f"  - reconstruction_errors.png")
    print(f"  - metrics_summary.png")
    print(f"  - detailed_comparison.png")
    print(f"\nTotal execution time: {reconstruction_time:.2f} seconds")


if __name__ == "__main__":
    main()