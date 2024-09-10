#!/bin/bash

# Set variables
LOCAL_DIR=~/Desktop/vlm_for_IoT/pythonProject/
CONDA_ENV=semcom
CONDA_SH_PATH="/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
PYTHON_SCRIPT="generate_report.py"  # Replace with your Python script name
REPORT_PATH="/Users/nemo/Desktop/vlm_for_IoT/Surveillance_Summary_Report.pdf"
EMAIL="tmoshubh@gmail.com"  # Replace with the recipient's email
SUBJECT="Daily Surveillance Summary Report"
BODY="Please find attached the Surveillance Summary Report."

# Step 1: Navigate to the local directory
cd "$LOCAL_DIR" || exit
echo "Navigated to $LOCAL_DIR"

# Step 2: Source conda.sh to enable conda commands
source "$CONDA_SH_PATH"

# Step 3: Deactivate any existing conda environment
conda deactivate

# Step 4: Activate the semcom conda environment
conda activate "$CONDA_ENV"

# Step 5: Run the Python script
python "$PYTHON_SCRIPT"
