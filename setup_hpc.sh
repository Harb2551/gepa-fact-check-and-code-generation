#!/bin/bash

# Name of the conda environment
ENV_NAME="gepa_env"

# Load conda (adjust this line if conda is not in your path or requires a module load)
# module load anaconda3/2023.03 # Example for some HPCs
# source ~/.bashrc

echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment..."
source activate $ENV_NAME || conda activate $ENV_NAME

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Logging in to Hugging Face..."
# Check if .env exists and source it to get HF_TOKEN
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Found HF_TOKEN in .env, logging in..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN not found in .env. Please run 'huggingface-cli login' manually."
fi

echo "Setup complete! To activate the environment, run:"
echo "conda activate $ENV_NAME"
