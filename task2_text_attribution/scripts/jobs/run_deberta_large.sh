#!/bin/bash
#SBATCH --account=training2562
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/log_deberta_large_%j.out
#SBATCH --error=logs/log_deberta_large_%j.err
#SBATCH --reservation=cispahack

cd /p/home/jusers/zainzinger1/jureca/HackathonVienna/task_2_text_attribution
source .venv/bin/activate

echo "=== STARTE DEBERTA-LARGE TRAINING ==="
srun python scripts/training/train_deberta_large.py
