import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your results (assuming your file is named v6_summary.csv)
# If your file is named differently, change the name below.
df = pd.read_csv('v6_summary.csv')

# 2. Define the 4 lines we actually want to show
conditions = ['oracle', 'naive', 'gated_hard', 'fellegi_sunter']

# Style settings to make it look like a published paper
styles = {
    'oracle': {'color': 'black', 'marker': 'o', 'label': 'Oracle (Clean Data)', 'linestyle': '--'},
    'naive': {'color': 'red', 'marker': 's', 'label': 'Naive (No Repair)', 'linestyle': '-'},
    'gated_hard': {'color': 'blue', 'marker': '^', 'label': 'Quarantine (Gated)', 'linestyle': '-'},
    'fellegi_sunter': {'color': 'green', 'marker': 'D', 'label': 'Repair (Fellegi-Sunter)', 'linestyle': '-'}
}

plt.figure(figsize=(8, 5))

# 3. Plot each line with error bars
for cond in conditions:
    subset = df[df['cond'] == cond]
    plt.errorbar(
        subset['rate'], 
        subset['mae3'], 
        yerr=subset['mae3_sd'], # Adds the standard deviation error bars
        color=styles[cond]['color'],
        marker=styles[cond]['marker'],
        label=styles[cond]['label'],
        linestyle=styles[cond]['linestyle'],
        capsize=4, 
        linewidth=2, 
        markersize=7
    )

# 4. ZOOM IN on the Y-axis so the differences are visible
# Based on your data, the lowest is ~0.343 and highest is ~0.351
plt.ylim(0.340, 0.355) 

# 5. Formatting for AAAI paper
plt.xlabel('Identity Corruption Rate (%)', fontsize=12, fontweight='bold')
plt.ylabel('3-Step Rollout MAE (Lower is Better)', fontsize=12, fontweight='bold')
plt.title('Impact of Identity Repair on GRU Trajectory Prediction', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks([0.0, 0.05, 0.1, 0.2, 0.3, 0.5]) # Forces the x-axis to match your test rates

# 6. Save the high-res image
plt.tight_layout()
plt.savefig('v6_results_paper_ready.png', dpi=300)
print("Graph saved as v6_results_paper_ready.png")