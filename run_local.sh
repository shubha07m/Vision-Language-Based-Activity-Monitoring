#!/bin/bash

# Define the local directory and conda environment
LOCAL_DIR=~/Desktop/vlm_for_IoT/pythonProject/
CONDA_ENV=semcom
CONDA_SH_PATH="/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"

# Step 1: Navigate to the local directory
cd "$LOCAL_DIR" || exit
echo "Navigated to $LOCAL_DIR"

# Step 2: Source conda.sh to enable conda commands
source "$CONDA_SH_PATH"

# Step 3: Deactivate any existing conda environment
conda deactivate

# Step 4: Activate the semcom conda environment
conda activate "$CONDA_ENV"

# Step 5: Run show_feed.py
echo "Running show_feed.py..."
python show_feed.py

