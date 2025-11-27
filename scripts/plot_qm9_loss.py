import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def parse_log_and_plot(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Error: File not found at {log_file_path}")
        return

    losses = []
    epochs = []

    # Regex to match lines like: Epoch 0: X_CE: 0.548 -- E_CE: 0.519 -- y_CE: -1.000 -- 42.2s
    # We will sum X_CE and E_CE as the total loss proxy
    pattern = re.compile(r"Epoch (\d+): X_CE: ([\d\.]+) -- E_CE: ([\d\.]+)")

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                x_ce = float(match.group(2))
                e_ce = float(match.group(3))
                
                total_loss = x_ce + e_ce
                losses.append(total_loss)
                epochs.append(epoch)

    if not losses:
        print("No training loss data found in log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Total Train Loss (X_CE + E_CE)')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("QM9 Training Loss Curve")
    plt.legend()
    plt.grid(True)
    
    output_file = "log/qm9_loss_curve.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Assuming the script is run from the project root or scripts directory
    # Adjust path as needed. The user said file is at log/qm9_training.log
    # relative to project root.
    log_path = "log/qm9_training.log"
    # If running from scripts dir, go up one level
    if not os.path.exists(log_path) and os.path.exists("../log/qm9_training.log"):
        log_path = "../log/qm9_training.log"
        
    parse_log_and_plot(log_path)
