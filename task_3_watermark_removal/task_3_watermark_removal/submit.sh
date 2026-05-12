#!/bin/bash
#SBATCH --account=training2562
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu
#SBATCH --output=logs/submit_%j.out
#SBATCH --error=logs/submit_%j.err
#SBATCH --reservation=cispahack

cd /p/project1/training2562/syed1/Hackathon/task_3_watermark_removal
source .venv/bin/activate

srun python submit.py

echo "done!"
