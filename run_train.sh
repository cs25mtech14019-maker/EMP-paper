#!/bin/bash
#SBATCH --job-name=emp_train
#SBATCH --partition=cse-gpu-all
#SBATCH --nodelist=dgx-a100-02,dgx-v100-01   # Using only the working Big GPUs
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=4-00:00:00             # Reserve for 4 Days (96 hours)
#SBATCH --output=train_log_%j.txt     # Save output to file
#SBATCH --error=train_err_%j.txt      # Save errors to file
#SBATCH --mail-type=END,FAIL          # Email alerts
#SBATCH --mail-user=cs25mtech14019@iith.ac.in

# 1. Load Environment
source ~/.bashrc
conda activate emp_verify

# 2. Go to code folder
cd /u/student/2025/cs25mtech14019/emp

echo "Training Job started at $(date)"
echo "Running on node: $(hostname)"

# 3. YOUR COMMAND (Optimized with correct path)
# Note: I replaced '/path/to/data_root' with your actual path
python train.py data_root=/u/student/2025/cs25mtech14019/emp/data/emp model=emp gpus=1 batch_size=96 monitor=val_minFDE6 model.target.decoder=mlp

echo "Job finished at $(date)"
