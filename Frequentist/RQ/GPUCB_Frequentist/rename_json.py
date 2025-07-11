import os

def rename_json_files(directory):
    # List all json files in sorted order to maintain sequence
    json_files = sorted([f for f in os.listdir(directory) if f.startswith("convergence_data_SUCB_Frq_RQ-") and f.endswith(".json")])

    # Only process the first 21 files
    for i in range(21):
        old_file = json_files[i]  # Get the old filename
        old_path = os.path.join(directory, old_file)

        # Construct the new filename
        new_number = 80 + i  # Start numbering from 80 upwards
        new_file = f"convergence_data_SUCB_Frq_RQ-{new_number}.json"
        new_path = os.path.join(directory, new_file)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_file} → {new_file}")

# Usage
rename_json_files("data58")
