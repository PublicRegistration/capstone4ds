import wandb
import pandas as pd

api = wandb.Api()

# Replace <run_id> with your actual run ID
run = api.run("/rsellers-projects/trackmania-model_LIDARPROGRESS_curriculum_teast1/runs/LIDARPROGRESS_curriculum_slalom_2")


# scan_history() returns an iterable of all logged steps — no 500-row cap
history = run.scan_history()

# Collect ALL rows into a list of dicts
# Each row is a dict of {metric_name: value} for that step
all_rows = [row for row in history]

# Convert to a DataFrame — this automatically handles missing keys per step
df = pd.DataFrame(all_rows)

# Optional: sort by step if wandb's internal step column is present
if "_step" in df.columns:
    df = df.sort_values("_step").reset_index(drop=True)

# Export to CSV
output_path = "run_metrics.csv"
df.to_csv(output_path, index=False)

print(f"Exported {len(df)} rows and {len(df.columns)} columns to '{output_path}'")
print("Columns found:", df.columns.tolist())