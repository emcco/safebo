import os
import json

# Define the folder path
folder_path = "data"

# Initialize variables to track the file with the most entries
max_entries = 0
max_file = None

# Iterate through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)

        # Read the JSON file
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
                mean_values = data.get("mean_values", [])  # Get the mean_values list
                num_entries = len(mean_values)  # Count the number of entries
                
                # Check if this file has the most entries so far
                if num_entries > max_entries:
                    max_entries = num_entries
                    max_file = filename
            except json.JSONDecodeError:
                print(f"Error reading {filename}, skipping...")

# Print the result
if max_file:
    print(f"The file with the most entries in 'mean_values' is: {max_file} with {max_entries} entries.")
else:
    print("No valid JSON files found or no 'mean_values' data available.")
