import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import globals
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from datetime import datetime
import os
import torch


def append_values_to_json(directory_path, num_samples):
# List all .json files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            # Open the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Check the length of "mean_values" and append if necessary
                mean_values = data.get("mean_values", [])
                
                while len(mean_values) < num_samples:
                    mean_values.append(mean_values[-1])  # Append the last value
                
                # Update the "mean_values" in the data
                data["mean_values"] = mean_values

            # Save the updated JSON file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Updated {filename}")


num_latest = 50

iterations = np.arange(1, num_latest + 1)
# data_dir = 'comp'+globals.comp+'/data'
data_dir = 'data'
# append_values_to_json(data_dir, num_latest)



if not os.path.exists(data_dir):
    print("No convergence data found")
    

# Get list of all convergence data files
files = sorted(glob.glob(f'{data_dir}/convergence_data_*.json'))
if len(files) == 0:
    print("No convergence data files found")
    

# Take the latest n files
executions = 100
files = files[-executions:]

fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)

# Plot each convergence curve
mean_values_all = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        y_max = data['y_max']
        mean_values = data['mean_values']
        conv = []
        for i in range(len(mean_values)):
            conv.append(y_max - mean_values[i])
        mean_values_all.append(conv)

mean_values_all = np.array(mean_values_all)
mean_values_mean = np.mean(mean_values_all, axis=0)
mean_values_stddev = np.std(mean_values_all, axis=0)

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        y_max = data['y_max']
        mean_values = data['mean_values']
        conv = []
        for i in range(len(mean_values)):
            conv.append(y_max - mean_values[i])
        # iterations = np.arange(1, len(mean_values) + 1)
        ax.plot(iterations, conv, color='grey', alpha=0.1)

# Plot mean
ax.plot(iterations, mean_values_mean, linewidth=2, color='black', label='Mean', marker='*')

ax.fill_between(
    iterations,
    mean_values_mean - mean_values_stddev,
    mean_values_mean + mean_values_stddev,
    color='gray',
    alpha=0.4,
    label='Mean ± Std. Dev.'
)

# Calculate and plot mean convergence
ax.axhline(0, color='black', linewidth=1)
# ax.axhline(0.003, color='red', linewidth=1)
ax.set_xlabel('Iterations', fontsize=12)
ax.set_xticks(iterations)
ax.set_ylabel('Regret R(t)', fontsize=12)
# ax.set_title(f'Convergence Plot: SafeOpt - Frequentist View (n={len(files)})')
ax.set_xlim(1,40)
ax.grid(True, alpha=0.2)


# Find iteration where mean convergence function reaches or falls below 0.003
convergence_iteration = None
for i, val in enumerate(mean_values_mean):
    if val <= 3*torch.sqrt(globals.noise_var):
        convergence_iteration = iterations[i]
        break

if convergence_iteration is not None:
    ax.axvline(convergence_iteration, color='red', linestyle='--', label=f'Converged at iteration {convergence_iteration}')
    # ax.text(convergence_iteration, max(mean_values_mean) * 0.5, f'Converged at {convergence_iteration}', 
    #         color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right', rotation=90)

ax.legend()

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# string = f'comp' + globals.comp + '/plot_mean_convergence_' + globals.comp + '.png'
string = 'plot_conv_RQ_Freq.png'
fig.savefig(string, format='png')
