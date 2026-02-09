#!/bin/bash
#SBATCH --job-name=emp_eval
#SBATCH --partition=cse-gpu-all
#SBATCH --nodelist=dgx-a100-02,dgx-v100-01   # Using only the working Big GPUs
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00               # Eval is fast (4 hours is plenty)
#SBATCH --output=eval_log_%j.txt
#SBATCH --error=eval_err_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cs25mtech14019@iith.ac.in

# 1. Load Environment
source ~/.bashrc
conda activate emp_verify

# 2. Go to code folder
cd /u/student/2025/cs25mtech14019/emp

echo "Evaluation Job started at $(date)"
echo "Running on node: $(hostname)"

# 3. RUN EVALUATION
# Note: Using your custom path '.../data/emp' so it finds '.../data/emp/emp'
python eval.py data_root=/u/student/2025/cs25mtech14019/emp/data/emp split=val checkpoint=checkpoints/empd.ckpt

echo "Job finished at $(date)"
