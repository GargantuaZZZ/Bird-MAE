#!/bin/bash
#SBATCH -A a_xinweichen
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J birdmae-hsn-probe
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=1200
#SBATCH --output=vitresult.%j.out
#SBATCH --error=vitresult.%j.err
#SBATCH --gres=gpu:1

# ============================================
# 环境设置
# ============================================
module load anaconda/3-2023.09
module load cuda/12.1
source activate birdmae
export PYTHONNOUSERSITE=1

# 项目根目录
export PROJECT_ROOT=/share/home/202420164351/Bird-MAE-main/Bird-MAE-main
cd $PROJECT_ROOT

# HuggingFace 离线模式（集群无外网）
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export ANALYSIS_OUTPUT_NAME=outputs_supcon.npz

# ============================================
# 创建必要目录
# ============================================
mkdir -p logs
mkdir -p $PROJECT_ROOT/logs/model_checkpoints

# ============================================
# 运行实验：Bird-MAE-Base Frozen + Prototypical Probing on HSN
# ============================================
echo "=========================================="
echo "Starting: Bird-MAE-Base Frozen Proto Probing on HSN"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

python hsn_contrastive_step1_5.py \
  --vit_npz logs/ab/VIT.npz \
  --supcon_npz logs/ab/VIT_Contrastive_test_outputs.npz \
  --annotations_csv /share/home/202420164351/Bird-MAE-main/Bird-MAE-main/data/HSNoriginal/annotations.csv \
  --out_dir logs/analysis_hsn_vit_vs_duibi

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="