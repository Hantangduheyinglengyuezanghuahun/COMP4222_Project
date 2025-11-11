#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-share

# export CUDA_VISIBLE_DEVICES=0
# /mnt/home/zpengac/.Miniconda3/envs/dnn/bin/pip install --force-reinstall natten
echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# (Optional) create logs dir
mkdir -p logs

# Load modules (example; adjust to your environment)
# module purge
# module load python/3.10 cuda/12.1

# Activate virtual environment (adjust path)
# source ~/venvs/graphsage/bin/activate

# Show Python & CUDA info
python - <<'EOF'
import torch, sys
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device name:", torch.cuda.get_device_name(0))
EOF

# Run training/inference
srun python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --use-a2


echo "Job finished on $(date)"