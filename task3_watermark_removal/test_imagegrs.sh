#!/bin/bash
#SBATCH --account=training2562
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=dc-gpu
#SBATCH --output=logs/gstest_%j.out
#SBATCH --error=logs/gstest_%j.err
#SBATCH --reservation=cispahack

echo "=== TESTING IMAGE-GS ENVIRONMENT ==="
echo "Time: $(date)"
echo "Node: $SLURMD_NODENAME"

cd /p/project1/training2562/syed1/Hackathon/task_3_watermark_removal/repo/image-gs
source .venv/bin/activate

module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12

nvidia-smi
echo ""

# Test imports
echo "Testing imports..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')

print('Importing gsplat (may take time for JIT compilation)...')
import gsplat
print('✓ gsplat imported!')
"

# If gsplat works, test fused_ssim
echo ""
echo "Testing fused_ssim..."
python -c "from fused_ssim import fused_ssim; print('✓ fused_ssim imported!')" || {
    echo "Installing fused_ssim..."
    pip install fused-ssim
    python -c "from fused_ssim import fused_ssim; print('✓ fused_ssim installed!')"
}

echo ""
echo "=== IMAGE-GS ENVIRONMENT READY ==="
echo "Time: $(date)"
