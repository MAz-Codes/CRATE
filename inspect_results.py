import json
import matplotlib.pyplot as plt
import os

HISTORY_FILE = "checkpoints_phase3/training_history.json"

if not os.path.exists(HISTORY_FILE):
    print(f"Error: {HISTORY_FILE} not found.")
    exit(1)

with open(HISTORY_FILE, 'r') as f:
    history = json.load(f)

print("Training History Summary:")
print("-" * 30)
epochs = len(history['train_loss'])
print(f"Total Epochs: {epochs}")

print("\nLast Epoch Metrics:")
print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
print(f"  Val Recon Loss: {history['val_recon_loss'][-1]:.4f}")
print(f"  Val KL Loss: {history['val_kl_loss'][-1]:.4f}")
print(f"  Active Units: {history['active_units'][-1]}")
print(f"  Beta: {history['beta'][-1]:.4f}")

# Plotting
try:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Reconstruction vs KL
    axs[0, 1].plot(history['train_recon_loss'], label='Train Recon')
    axs[0, 1].plot(history['val_recon_loss'], label='Val Recon')
    axs[0, 1].set_title('Reconstruction Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # KL Divergence
    axs[1, 0].plot(history['train_kl_loss'], label='Train KL')
    axs[1, 0].plot(history['val_kl_loss'], label='Val KL')
    axs[1, 0].set_title('KL Divergence')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Active Units & Beta
    ax2 = axs[1, 1].twinx()
    axs[1, 1].plot(history['active_units'], label='Active Units', color='purple')
    ax2.plot(history['beta'], label='Beta', color='orange', linestyle='--')
    axs[1, 1].set_title('Active Units & Beta')
    axs[1, 1].set_ylabel('Active Units')
    ax2.set_ylabel('Beta')
    axs[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_plot_phase3.png')
    print("\n✓ Saved training plot to training_plot_phase3.png")
except ImportError:
    print("\nMatplotlib not found. Skipping plot generation.")
except Exception as e:
    print(f"\nCould not plot: {e}")
