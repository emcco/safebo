import subprocess

# Define paths to your main.py files
scripts = [
    "/Users/kaidahasanovic/Documents/mathesis_emir/code/SafeOpt_Frequentist/main.py",
    "/Users/kaidahasanovic/Documents/mathesis_emir/code/GPUCB_Frequentist/main.py",
    # "/Users/kaidahasanovic/Documents/mathesis_emir/code/SafeOpt_Frequentist//main.py"
]

# Execute each script sequentially
for script in scripts:
    print(f"Running: {script}")
    subprocess.run(["python3", script], check=True)
