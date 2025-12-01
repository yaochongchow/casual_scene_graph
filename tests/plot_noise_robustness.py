import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def plot_noise_robustness(csv_file="qm9_noise_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: File not found at {csv_file}")
        print("Please ensure the results file exists before running this script.")
        return

    try:
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if "p_edge_drop" not in df.columns or "loss" not in df.columns:
            print(f"Error: CSV file must contain 'p_edge_drop' and 'loss' columns.")
            print(f"Found columns: {df.columns.tolist()}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(df["p_edge_drop"], df["loss"], marker="o", linestyle='-', linewidth=2)
        plt.xlabel("Edge drop probability (noise level)")
        plt.ylabel("Average loss on QM9 test set")
        plt.title("QM9 robustness to edge noise")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        output_file = "qm9_noise_curve.png"
        plt.savefig(output_file, dpi=200)
        print(f"Plot saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_noise_robustness()
