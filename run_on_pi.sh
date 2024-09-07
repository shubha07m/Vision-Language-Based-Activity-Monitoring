#!/bin/bash

# Define variables for Raspberry Pi
RASPBERRY_PI_IP="ener raspberry ip here"
PI_PASSWORD="enter raspberry pi password here"
PORT=5000

# Function to run commands on Raspberry Pi
run_on_raspberry_pi() {
    sshpass -p "$PI_PASSWORD" ssh -t -o StrictHostKeyChecking=no $RASPBERRY_PI_IP << EOF
        # Check for process using the port and kill it if found
        sudo lsof -t -i:$PORT | xargs -r sudo kill -9

        # Navigate to the project directory
        cd /home/nemo/Desktop/my_projects/video_streaming/

        # Source conda.sh to use conda commands
        source /home/nemo/miniforge3/etc/profile.d/conda.sh

        # Deactivate any active conda environment
        conda deactivate

        # Activate the conda environment
        conda activate video

        # Run the Python script in the background
        nohup python stream.py > stream.log 2>&1 &
        echo "Flask server is running in the background."
EOF
}

# Run the function
run_on_raspberry_pi

