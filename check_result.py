import optuna

# 1. Load the existing study
# must match the storage and study_name from your tuning script
study = optuna.load_study(
    study_name="causal_gnn_molhiv_optimization", 
    storage="sqlite:///causal_tuning.db"
)

# 2. Print the best result so far
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
trial = study.best_trial

print(f"  Value (AUC): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 3. (Optional) Convert to a Pandas DataFrame for easier viewing
# pip install pandas
import pandas as pd
df = study.trials_dataframe()
# Sort by value (AUC) descending
print("\nTop 5 Trials:")
print(df.sort_values(by="value", ascending=False).head(5)[['number', 'value', 'params_lr', 'params_sparsity_weight']])