#!/bin/bash
#SBATCH --account=training2562
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --output=output/inference_%j.out
#SBATCH --error=output/inference_%j.err
#SBATCH --time=01:00:00
#SBATCH --reservation=cispahack

# Dataset Inference Pipeline Job Script
# ======================================
# Runs the membership inference pipeline on all 1000 subsets

echo "=========================================="
echo "Dataset Inference Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Navigate to project directory
cd $PROJECT/$USER/Hackathon/task_1_dataset_inference

# Activate the environment
source $PROJECT/$USER/Hackathon/.venv/bin/activate

# Create output directory
mkdir -p output
mkdir -p features

# Print environment info
echo ""
echo "Environment Info:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run the inference pipeline
echo "Starting inference pipeline..."
srun python dataset_inference.py --method gmm --output submission.csv

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output file: submission.csv"
echo "Finished at: $(date)"
