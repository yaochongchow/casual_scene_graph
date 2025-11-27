import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def parse_log_and_analyze(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Error: File not found at {log_file_path}")
        return

    # Data containers
    train_losses = []
    train_epochs = []
    
    val_losses = []
    val_epochs = []
    
    validity_scores = []
    validity_epochs = []
    
    config_info = {}
    
    # Regex patterns
    train_pattern = re.compile(r"Epoch (\d+): X_CE: ([\d\.]+) -- E_CE: ([\d\.]+)")
    val_loss_pattern = re.compile(r"Val loss: ([\d\.]+)")
    val_epoch_pattern = re.compile(r"Epoch (\d+): Val NLL")
    validity_pattern = re.compile(r"Validity over \d+ molecules: ([\d\.]+)%")
    
    current_val_epoch = -1

    print(f"Analyzing log file: {log_file_path}...")
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Parse Training Loss
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                x_ce = float(train_match.group(2))
                e_ce = float(train_match.group(3))
                train_losses.append(x_ce + e_ce)
                train_epochs.append(epoch)
                continue

            # Parse Validation Epoch
            val_epoch_match = val_epoch_pattern.search(line)
            if val_epoch_match:
                current_val_epoch = int(val_epoch_match.group(1))
                continue

            # Parse Validation Loss
            val_loss_match = val_loss_pattern.search(line)
            if val_loss_match and current_val_epoch != -1:
                val_loss = float(val_loss_match.group(1))
                val_losses.append(val_loss)
                val_epochs.append(current_val_epoch)
                # Reset current_val_epoch to avoid duplicate assignments if multiple lines match
                # (though usually Val loss line follows immediately)
                continue

            # Parse Validity
            validity_match = validity_pattern.search(line)
            if validity_match and current_val_epoch != -1:
                score = float(validity_match.group(1))
                validity_scores.append(score)
                validity_epochs.append(current_val_epoch)
                continue

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Losses
    if train_epochs:
        ax1.plot(train_epochs, train_losses, label='Train Loss (X_CE + E_CE)', alpha=0.7)
    if val_epochs:
        ax1.plot(val_epochs, val_losses, label='Validation Loss', marker='o', linestyle='--')
    
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Validity
    if validity_epochs:
        ax2.plot(validity_epochs, validity_scores, label='Validity (%)', color='green', marker='s')
        ax2.set_ylabel("Validity (%)")
        ax2.set_title("Generated Molecule Validity")
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "No validity data found", ha='center', va='center')

    ax2.set_xlabel("Epoch")
    
    output_plot = "log/training_analysis.png"
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"\nAnalysis plot saved to: {output_plot}")

    # --- Summary Report ---
    print("\n" + "="*40)
    print("       TRAINING ANALYSIS REPORT       ")
    print("="*40)
    
    if train_epochs:
        print(f"Total Epochs Logged: {max(train_epochs)}")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
    
    if val_losses:
        min_val_loss = min(val_losses)
        min_val_idx = val_losses.index(min_val_loss)
        best_epoch = val_epochs[min_val_idx]
        print(f"Best Validation Loss: {min_val_loss:.4f} (Epoch {best_epoch})")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    
    if validity_scores:
        print(f"Best Validity: {max(validity_scores)}%")
        print(f"Final Validity: {validity_scores[-1]}%")

    print("\n" + "="*40)

if __name__ == "__main__":
    log_path = "log/qm9_training.log"
    # Fallback if running from scripts dir
    if not os.path.exists(log_path) and os.path.exists("../log/qm9_training.log"):
        log_path = "../log/qm9_training.log"
        
    parse_log_and_analyze(log_path)
