#!/bin/bash
#SBATCH --account=training2562
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=dc-gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --reservation=cispahack

echo "=========================================="
echo "Job: $SLURM_JOB_ID @ $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

# Wechsle ins Projektverzeichnis
cd /p/project1/training2562/syed1/Hackathon/task_3_watermark_removal

# UV Paths setzen
export UV_PYTHON_INSTALL_DIR=/p/project1/training2562/syed1/uv/python
export UV_CACHE_DIR=/p/project1/training2562/syed1/uv/cache
export UV_TOOL_DIR=/p/project1/training2562/syed1/uv/tools

# Aktiviere UV Environment
source .venv/bin/activate

# Lade CUDA Module
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12

# Zeige GPU Info
nvidia-smi
echo ""

# Progress File
PROGRESS_FILE="logs/progress_${SLURM_JOB_ID}.log"
echo "Progress wird geschrieben nach: $PROGRESS_FILE"
echo "=========================================="

# Installiere gsplat falls nicht vorhanden (JIT kompilierung auf A100)
echo "Checking gsplat installation..." | tee -a $PROGRESS_FILE
python -c "import gsplat; print('✓ gsplat already installed')" 2>/dev/null || {
    echo "Installing gsplat from image-gs repo..." | tee -a $PROGRESS_FILE
    cd repo/image-gs/gsplat
    uv pip install -e . --no-build-isolation 2>&1 | tee -a $PROGRESS_FILE
    cd /p/project1/training2562/syed1/Hackathon/task_3_watermark_removal
    echo "✓ gsplat installation complete" | tee -a $PROGRESS_FILE
}

echo "" | tee -a $PROGRESS_FILE
echo "=========================================" | tee -a $PROGRESS_FILE
echo "STARTING FULL PIPELINE WITH IMAGE-GS" | tee -a $PROGRESS_FILE
echo "=========================================" | tee -a $PROGRESS_FILE
echo "Time: $(date)" | tee -a $PROGRESS_FILE

# Run full pipeline with all attacks including Image-GS
# --no-submit: Nicht an Server senden (nur lokal speichern)
python scripts/run_attack.py \
    --preset full \
    --output submission.npz \
    2>&1 | tee -a $PROGRESS_FILE

echo "" | tee -a $PROGRESS_FILE
echo "=========================================" | tee -a $PROGRESS_FILE
echo "PIPELINE COMPLETE" | tee -a $PROGRESS_FILE
echo "Time: $(date)" | tee -a $PROGRESS_FILE
echo "=========================================" | tee -a $PROGRESS_FILE

# Zeige Submission Info
echo "" | tee -a $PROGRESS_FILE
echo "Submission file info:" | tee -a $PROGRESS_FILE
python -c "
import numpy as np
data = np.load('submission.npz')
print(f'  images shape: {data[\"images\"].shape}')
print(f'  names count: {len(data[\"names\"])}')
print(f'  First 5 names: {data[\"names\"][:5]}')
" 2>&1 | tee -a $PROGRESS_FILE

echo ""
echo "=========================================="
echo "Ende: $(date)"
echo "Progress log: $PROGRESS_FILE"
echo "Submission: submission.npz"
echo "=========================================="
