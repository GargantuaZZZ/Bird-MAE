# Bird-MAE 分阶段复现计划
# 环境：集群 + SLURM | 模型：先 Base 后扩展 | 目标：先跑通下游，再预训练

---

## ═══════════════════════════════════════════
## 阶段 0：环境搭建（在登录节点执行）
## ═══════════════════════════════════════════

### Step 0.1 克隆仓库

```bash
cd /home/$USER/projects  # 或你的工作目录
git clone https://github.com/DBD-research-group/Bird-MAE.git
cd Bird-MAE
```

### Step 0.2 创建 conda 环境

```bash
conda create -n birdmae python=3.10.14 -y
conda activate birdmae
```

### Step 0.3 安装依赖（⚠️ 已知问题需注意）

requirements.txt 中存在版本冲突问题（GitHub Issue #4）。推荐以下方式处理：

```bash
# 方法 A：直接安装，忽略版本冲突（推荐先试）
pip install -r requirements.txt

# 如果报错（特别是 datasets、mkl_fft、mkl_random 版本冲突），用方法 B：
# 方法 B：去掉严格版本号后安装
sed 's/==.*//g' requirements.txt > requirements_relaxed.txt
# 然后手动固定关键包的版本
pip install -r requirements_relaxed.txt
pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
pip install lightning==2.4.0 pytorch-lightning==2.4.0
pip install timm==1.0.9 transformers==4.44.2
pip install datasets==3.0.0  # BirdSet 需要 datasets<=3.6.0
pip install hydra-core==1.3.2 hydra-colorlog==1.2.0
```

**关键依赖说明：**
- `birdset`：通过 `-e git+` 从 GitHub 安装，是数据管道核心
- `torch==2.2.2` + `CUDA 12.1`：requirements 锁定了 nvidia-cuda-* 12.1 系列
- `timm==1.0.9`：ViT 模型实现依赖
- `lightning==2.4.0`：训练框架
- `hydra-core==1.3.2`：实验配置管理

### Step 0.4 验证环境

```bash
conda activate birdmae
python -c "
import torch
import torchaudio
import timm
import lightning
import hydra
from datasets import load_dataset
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'timm: {timm.__version__}')
print(f'lightning: {lightning.__version__}')
print('All imports OK!')
"
```

如果 birdset 导入失败，单独检查：
```bash
python -c "import birdset; print('BirdSet OK')"
```

---

## ═══════════════════════════════════════════
## 阶段 1：下载 HSN 数据 + 跑通下游验证（最快路线）
## ═══════════════════════════════════════════

### Step 1.1 准备 HSN 下游数据

```bash
# 在登录节点或数据节点执行（需要网络）
# 根据你集群的存储位置修改路径
export BIRDSET_DATA="/path/to/your/scratch/birdset"  # ← 改成你的路径
mkdir -p $BIRDSET_DATA

python util/prepare_data/downstream.py \
    --dataset-names HSN \
    --cache-dir-base $BIRDSET_DATA
```

> HSN 是最小的数据集（21类），下载 + 处理约几十分钟。

### Step 1.2 验证数据是否正确

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('$BIRDSET_DATA/HSN/HSN_processed')  # 路径可能需调整
print(ds)
print(f'Train samples: {len(ds[\"train\"])}')
print(f'Test samples: {len(ds[\"test_5s\"])}')
"
```

### Step 1.3 编写 SLURM 脚本：Frozen Prototypical Probing on HSN

创建文件 `slurm/my_first_run.sh`：

```bash
#!/bin/bash
#SBATCH --job-name=birdmae-hsn-probe
#SBATCH --partition=gpu           # ← 改成你的GPU分区名
#SBATCH --gres=gpu:1              # 单卡即可
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00           # Probing 很快，2小时足够
#SBATCH --output=logs/hsn_probe_%j.log
#SBATCH --error=logs/hsn_probe_%j.err

# 加载环境
source activate birdmae
# 或者：conda activate birdmae / module load ... 取决于你的集群

cd /home/$USER/projects/Bird-MAE  # ← 改成你的路径

# 创建日志目录
mkdir -p logs

# 运行 frozen + prototypical probing
# 需要检查 configs/ 下的实际配置文件路径
python train.py experiment="paper/bigshot/birdmae_base/frozen/proto/HSN"
```

> **⚠️ 重要**：你需要先查看 `configs/experiment/paper/bigshot/` 目录结构，确认
> 配置文件名是否匹配。Hydra 配置路径 = 文件系统路径。

### Step 1.4 提交任务

```bash
mkdir -p logs
sbatch slurm/my_first_run.sh
```

### Step 1.5 检查结果

任务完成后检查日志中的 mAP 值。论文中 Bird-MAE-Base frozen + prototypical probing
在 HSN 上的 mAP 为 **43.84%**。

---

## ═══════════════════════════════════════════
## 阶段 2：扩展到 Fine-tuning + 更多数据集
## ═══════════════════════════════════════════

### Step 2.1 下载全部 8 个下游数据集

```bash
python util/prepare_data/downstream.py \
    --dataset-names HSN POW PER NES UHH NBP SSW SNE \
    --cache-dir-base $BIRDSET_DATA
```

### Step 2.2 Fine-tuning 脚本

```bash
#!/bin/bash
#SBATCH --job-name=birdmae-hsn-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00           # Fine-tuning 耗时更长
#SBATCH --output=logs/hsn_ft_%j.log

source activate birdmae
cd /home/$USER/projects/Bird-MAE

python train.py experiment="paper/bigshot/birdmae_base/finetune/proto/HSN"
```

论文参考值（Bird-MAE-Base fine-tuning + proto on HSN）：**mAP = 52.06%**

### Step 2.3 批量跑所有数据集

```bash
#!/bin/bash
# batch_eval.sh — 批量提交所有数据集的评估任务

DATASETS="HSN POW PER NES UHH NBP SSW SNE"
MODEL="birdmae_base"
MODES="finetune frozen"
HEAD="proto"

for dataset in $DATASETS; do
    for mode in $MODES; do
        echo "Submitting $MODEL / $mode / $HEAD / $dataset"
        sbatch --job-name="bm-${dataset}-${mode}" \
               --partition=gpu \
               --gres=gpu:1 \
               --cpus-per-task=8 \
               --mem=64G \
               --time=06:00:00 \
               --output="logs/${dataset}_${mode}_%j.log" \
               --wrap="source activate birdmae && cd /home/$USER/projects/Bird-MAE && python train.py experiment='paper/bigshot/${MODEL}/${mode}/${HEAD}/${dataset}'"
    done
done
```

---

## ═══════════════════════════════════════════
## 阶段 3：从头预训练 Bird-MAE-Base
## ═══════════════════════════════════════════

### Step 3.1 下载预训练数据（~500GB）

```bash
# 提交为长时间数据下载任务，或在数据节点执行
python util/prepare_data/pretraining.py \
    --dataset_name "XCL" \
    --hf_path "DBD-research-group/BirdSet" \
    --cache_dir "$BIRDSET_DATA/XCL" \
    --save_path "$BIRDSET_DATA/XCL/XCL_processed_curated" \
    --class_limit 500 \
    --event_limit 2 \
    --audio_sampling_rate 32000 \
    --num_proc 4 \
    --mapping_num_proc 8
```

> class_limit=500 + event_limit=2 → 论文使用的 XCL-1.7M 筛选策略。
> 如果磁盘充足，也可以先下载全部（class_limit=0, event_limit=0），再筛选。

### Step 3.2 预训练 SLURM 脚本

```bash
#!/bin/bash
#SBATCH --job-name=birdmae-pretrain-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4              # 论文用多卡，Base 模型 2-4 卡合理
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00           # 预训练耗时长，按需调整
#SBATCH --output=logs/pretrain_base_%j.log

source activate birdmae
cd /home/$USER/projects/Bird-MAE

# 使用仓库自带的预训练配置
# 先查看 configs/experiment/paper/pretrain/base 下的配置
# 确认数据路径已修改
python pretrain.py experiment="paper/pretrain/base"
```

> **关键配置需要修改**（在 Hydra yaml 中）：
> - 数据路径：指向你的 XCL_processed_curated 目录
> - 输出路径：checkpoint 保存位置
> - GPU 数量：与 SLURM 申请一致
> - wandb/mlflow 配置：可选的实验追踪

### Step 3.3 预训练关键参数参考（论文 Table 1）

| 参数 | Bird-MAE-Base |
|------|---------------|
| 输入尺寸 | 512×128 (5s @ 32kHz) |
| Decoder | ViT |
| Epochs | 150 |
| Masking ratio | 0.75 |
| Batch size | 1024 |
| Mixup | 0.3 |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Normalization mean | -7.2 |
| Normalization std | 4.43 |

### Step 3.4 用自己预训练的权重做下游评估

预训练完成后，将 checkpoint 路径填入下游任务配置中，重新跑阶段 2。

---

## ═══════════════════════════════════════════
## 排障清单
## ═══════════════════════════════════════════

### 问题 1：requirements.txt 安装报错
**现象**：datasets、mkl_fft、mkl_random 版本冲突
**方案**：见 Step 0.3 方法 B。核心是确保 datasets>=3.0.0 且 <=3.6.0

### 问题 2：Hydra 配置路径找不到
**现象**：`Could not find experiment=paper/bigshot/...`
**方案**：
```bash
# 先检查实际的配置目录结构
find configs/ -name "*.yaml" | head -30
# 确认路径是否匹配
ls configs/experiment/paper/bigshot/
```
Hydra 的 experiment 路径对应 configs/experiment/ 下的 yaml 文件路径（不含 .yaml 后缀）

### 问题 3：CUDA OOM
**现象**：`RuntimeError: CUDA out of memory`
**方案**：
- Frozen probing：单卡 16GB 应该够（Base 模型）
- Fine-tuning Base：需要约 24GB+
- 降低 batch size：在 Hydra config 中覆盖
  ```bash
  python train.py experiment="..." datamodule.batch_size=64
  ```
- 使用 mixed precision（已默认启用 16-mixed）

### 问题 4：数据路径问题
**现象**：找不到数据集
**方案**：Hydra config 中有数据路径的默认值，需要通过命令行覆盖或修改 yaml：
```bash
python train.py experiment="..." datamodule.dataset_dir="/your/path/birdset"
```
具体参数名需要查看 `configs/` 下的 datamodule 配置。

### 问题 5：BirdSet 包安装失败
**现象**：`-e git+https://github.com/DBD-research-group/BirdSet.git@...` 报错
**方案**：
```bash
# 单独安装 BirdSet
pip install git+https://github.com/DBD-research-group/BirdSet.git@6e4bbcbbb2e24eaa44eea7ee45c10d9df39d2348
```

---

## ═══════════════════════════════════════════
## 阶段性成功标志
## ═══════════════════════════════════════════

| 阶段 | 成功标志 | 预计耗时 |
|------|---------|---------|
| 阶段 0 | `python -c "import birdset; import timm; ..."` 全部通过 | 30 分钟 |
| 阶段 1 | HSN frozen proto probing mAP ≈ 43.84% | 2-3 小时 |
| 阶段 2 | HSN fine-tuning proto mAP ≈ 52.06% | 半天 |
| 阶段 2+ | 8 个数据集结果与论文 Table 6/7 一致 | 1-2 天 |
| 阶段 3 | 预训练 loss 收敛，下游评估与官方权重接近 | 1-2 周 |

---

## 立即执行的第一条命令

```bash
git clone https://github.com/DBD-research-group/Bird-MAE.git && cd Bird-MAE
```
