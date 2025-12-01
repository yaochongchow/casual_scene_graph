import optuna
import subprocess
import sys
import re
import os
import time
import random

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET = "ogbg-molhiv"
BIAS = 0.9
NUM_TRIALS = 50
STORAGE_DB = "sqlite:///causal_tuning_multi.db"  # New DB for multi-objective
STUDY_NAME = "causal_gnn_multi_objective"

def objective(trial):
    # -------------------------------------------------
    # 1. Define Search Space
    # -------------------------------------------------
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = 256
    
    # We broaden the search for sparsity weight since we are optimizing for it now
    sparsity_weight = trial.suggest_float("sparsity_weight", 1e-5, 0.1, log=True)
    
    contrastive_weight = trial.suggest_float("contrastive_weight", 0.5, 5.0)
    entropy_weight = trial.suggest_float("entropy_weight", 0.0, 0.2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    
    # -------------------------------------------------
    # 2. Construct Command
    # -------------------------------------------------
    save_dir = f"./checkpoints/tuning_multi/trial_{trial.number}"
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "train_causal_gnn.py",
        "--dataset", DATASET,
        "--bias", str(BIAS),
        "--epochs", "40",           
        "--warmup_epochs", "5",
        
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--dropout", str(dropout),
        "--contrastive_weight", str(contrastive_weight),
        "--sparsity_weight", str(sparsity_weight),
        "--entropy_weight", str(entropy_weight),
        "--cf_cls_weight", "0.0",
        "--target_sparsity", "0.5",  # We want it to aim for 0.5
        
        "--save_dir", save_dir,
        "--single_gpu"
    ]
    
    print(f"\n[Trial {trial.number}] Started...")
    
    # -------------------------------------------------
    # 3. Run Training & Parse Output
    # -------------------------------------------------
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        output = result.stdout
        
        # We need to parse BOTH Val AUC and Sparsity from the logs
        # Format: Epoch  X | ... Val AUC: 0.7097 ... Sparsity: 0.99 ...
        
        best_auc = 0.0
        associated_sparsity = 1.0 # Default to bad sparsity
        
        # Iterate through lines to find the best epoch
        for line in output.split('\n'):
            if "Val AUC:" in line and "Sparsity:" in line:
                # Extract AUC
                auc_match = re.search(r"Val AUC: (\d+\.\d+)", line)
                if auc_match:
                    current_auc = float(auc_match.group(1))
                    
                    # Extract Sparsity
                    sparsity_match = re.search(r"Sparsity: (\d+\.\d+)", line)
                    if sparsity_match:
                        current_sparsity = float(sparsity_match.group(1))
                        
                        # Store the metrics associated with the BEST AUC seen so far
                        if current_auc > best_auc:
                            best_auc = current_auc
                            associated_sparsity = current_sparsity

        print(f"[Trial {trial.number}] Best AUC: {best_auc}, Sparsity: {associated_sparsity}")
        
        # OBJECTIVE 1: Maximize AUC (Higher is better)
        # OBJECTIVE 2: Minimize Distance to 0.5 (Lower is better)
        sparsity_error = abs(associated_sparsity - 0.5)
        
        return best_auc, sparsity_error
            
    except subprocess.CalledProcessError:
        print(f"[Trial {trial.number}] Crashed!")
        return 0.0, 1.0 # Return worst possible scores

if __name__ == "__main__":
    time.sleep(random.uniform(0, 2))
    
    # Create Multi-Objective Study
    # directions=["maximize", "minimize"] means:
    # 1. Maximize AUC
    # 2. Minimize Sparsity Error (distance from 0.5)
    study = optuna.create_study(
        directions=["maximize", "minimize"], 
        storage=STORAGE_DB,
        study_name=STUDY_NAME,
        load_if_exists=True
    )
    
    print(f"Starting Multi-Objective Optimization...")
    study.optimize(objective, n_trials=NUM_TRIALS)