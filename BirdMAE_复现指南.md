# Bird-MAE 完整复现指南

## 概览

本指南基于论文 "Can Masked Autoencoders Also Listen to Birds?" 和官方仓库 README，覆盖从环境搭建到预训练、微调、评估的完整流程。

---

## 第一步：克隆仓库 & 搭建环境

```bash
# 1. 克隆仓库
git clone https://github.com/DBD-research-group/Bird-MAE.git
cd Bird-MAE

# 2. 创建 conda 环境
conda create -n birdmae python=3.10.14 -y
conda activate birdmae

# 3. 安装依赖
pip install -r requirements.txt
```

### 环境验证

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('HuggingFace OK')"
```

---

## 第二步：数据准备

Bird-MAE 有三类数据，按需下载即可。

### 2A. 下游评估数据（推荐先准备，体积较小）

用于 BirdSet 8 个多标签分类下游任务的微调 / probing。

```bash
# 下载所有 8 个数据集
python util/prepare_data/downstream.py --cache-dir-base /data/birdset

# 或只下载部分（HSN 最小，适合调试）
python util/prepare_data/downstream.py \
    --dataset-names HSN \
    --cache-dir-base /data/birdset
```

**各数据集说明：**

| 数据集 | 地区 | 类别数 | 训练样本 | 测试样本 |
|--------|------|--------|----------|----------|
| HSN (High Sierra Nevada) | 美国 | 21 | 17,938 | 12,000 |
| POW (Powdermill) | 美国东部 | 48 | 2,586 | 4,560 |
| PER (Amazon Basin) | 秘鲁 | 132 | 5,743 | 15,120 |
| NES (Colombia Costa Rica) | 中美 | 89 | 4,034 | 24,480 |
| UHH (Hawaiian Islands) | 夏威夷 | 27 | 12,978 | 36,637 |
| NBP (France and Spain) | 欧洲 | 51 | 76,438 | 563 |
| SSW (Sapsucker Woods) | 美国 | 81 | 4,285 | 205,200 |
| SNE (Sierra Nevada) | 美国 | 56 | 2,557 | 23,756 |

> **建议**：调试阶段先用 HSN（21类，数据量小），跑通后再扩展。

### 2B. 预训练数据（仅从头预训练时需要）

下载 Xeno-Canto 数据并处理为 XCL-1.7M 格式。

⚠️ **需约 500GB 磁盘空间**

```bash
# 带数据筛选的版本（论文使用的 XCL-1.7M，推荐）
python util/prepare_data/pretraining.py \
    --dataset_name "XCL" \
    --hf_path "DBD-research-group/BirdSet" \
    --cache_dir "/data/birdset/XCL" \
    --save_path "/data/birdset/XCL/XCL_processed_curated" \
    --class_limit 500 \
    --event_limit 2 \
    --audio_sampling_rate 32000 \
    --num_proc 1 \
    --mapping_num_proc 4

# 如果不做筛选，下载全部事件（XCL-3.4M）
python util/prepare_data/pretraining.py \
    --dataset_name "XCL" \
    --hf_path "DBD-research-group/BirdSet" \
    --cache_dir "/data/birdset/XCL" \
    --save_path "/data/birdset/XCL/XCL_processed_allevents" \
    --class_limit 0 \
    --event_limit 0 \
    --audio_sampling_rate 32000 \
    --num_proc 1 \
    --mapping_num_proc 4
```

**数据筛选参数说明（论文 Appendix A）：**
- `class_limit=500`：每个物种最多保留 500 个事件
- `event_limit=2`：每个录音文件最多提取 2 个事件
- 这两个参数将 3.4M 事件缩减为 1.7M，减少冗余和类别不平衡

### 2C. Few-shot 数据（可选）

```bash
python util/prepare_data/fewshot.py \
    --dataset-names HSN POW PER NES UHH NBP SSW SNE \
    --shots 1 5 10 \
    --seeds 1 2 3 \
    --condition lenient \
    --cache-dir-base /data/birdset
```

---

## 第三步：下载预训练权重

如果不从头预训练，直接下载 checkpoint 即可。

### 方式一：通过 HuggingFace Transformers（推荐，含模型架构代码）

```python
from transformers import AutoFeatureExtractor, AutoModel
import librosa

# 选择模型规模：Bird-MAE-Base / Bird-MAE-Large / Bird-MAE-Huge
model_name = "DBD-research-group/Bird-MAE-Large"

model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# 测试推理
audio, sr = librosa.load(librosa.ex('robin'), sr=32000)
mel = feature_extractor(audio)
embedding = model(mel)
print(f"Embedding shape: {embedding.shape}")
```

### 方式二：直接下载 checkpoint 文件

从 https://huggingface.co/DBD-research-group/Bird-MAE 下载纯权重。

### 模型规模对比

| 模型 | 参数量 | HSN mAP (fine-tune) | HSN mAP (proto probe) | 推荐场景 |
|------|--------|--------------------|-----------------------|----------|
| Bird-MAE-Base | 86M | 52.06% | 43.84% | 资源有限 / 快速验证 |
| Bird-MAE-Large | 307M | 55.28% | 49.97% | 论文最优 / 推荐 |
| Bird-MAE-Huge | 632M | 54.80% | 47.52% | 边际收益递减 |

> **注意**：Large 在几乎所有任务上都优于 Huge，推荐使用 Large。

---

## 第四步：下游任务 —— 微调（Fine-tuning）

使用预训练权重在 BirdSet 下游任务上微调整个模型。

### 运行方式

```bash
# 格式：python train.py experiment="paper/bigshot/$model/$type/$head/$dataset"
# 其中：
#   $model: birdmae_base, birdmae_large, birdmae_huge, audiomae 等
#   $type: finetune, frozen
#   $head: proto (prototypical pooling), linear, attentive
#   $dataset: HSN, POW, PER, NES, UHH, NBP, SSW, SNE

# 示例：Bird-MAE-Large + Fine-tuning + Prototypical Pooling + HSN
python train.py experiment="paper/bigshot/birdmae_large/finetune/proto/HSN"

# 示例：Bird-MAE-Base + Fine-tuning + Prototypical Pooling + POW
python train.py experiment="paper/bigshot/birdmae_base/finetune/proto/POW"
```

### 关键超参数（论文 Table 10）

| 参数 | 值 |
|------|-----|
| Batch size | 128（Base/Large），64（Huge） |
| Epochs | 30 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Weight decay | 3e-4 |
| Layer decay | 0.75 |
| Loss | Asymmetric Loss |
| Pooling | Prototypical（20 prototypes） |
| Prototype LR | 4e-2 |
| Scheduler | Cosine Annealing |
| Precision | 16-mixed |

### 数据增强（论文 Table 9，所有实验统一使用）

| 增强方式 | 概率 | 参数 |
|----------|------|------|
| Cyclic rolling start | 1.0 | - |
| Multi-label mixup | 0.9 | snr=2-30, max-samples=3 |
| Background noise | 0.5 | snr=3-30 |
| Colored noise | 0.2 | snr=3-30 |
| Gain adjustment | 0.2 | gain=-18~6 |
| No-call mixing | 0.075 | - |
| Frequency masking | 0.3 | param=50 |
| Time masking | 0.3 | param=100 |

---

## 第五步：下游任务 —— 冻结表示 + Probing

冻结预训练编码器，只训练轻量级 probe head。

### Prototypical Probing（论文推荐，效果最佳）

```bash
# Bird-MAE-Large + Frozen + Prototypical Probing + HSN
python train.py experiment="paper/bigshot/birdmae_large/frozen/proto/HSN"
```

### Linear Probing

```bash
python train.py experiment="paper/bigshot/birdmae_large/frozen/linear/HSN"
```

### Probing 方法对比（HSN，Bird-MAE-L）

| Probing 方法 | mAP | 额外参数量 |
|-------------|-----|-----------|
| Linear | 12.44% | 21k |
| MLP | 15.22% | 535k |
| Attentive | 47.81% | 2.1M |
| **Prototypical (J=20)** | **49.97%** | **430k** |
| Fine-tuning (上限) | 55.28% | - |

> Prototypical probing 用最少的参数（430k）获得了最接近 fine-tuning 的效果。

---

## 第六步：Few-Shot 评估（可选）

```bash
# 格式：python train.py experiment="paper/fewshot/$probing/$probing/$dataset_kshots"
# 示例
python train.py experiment="paper/fewshot/proto/proto/HSN_5shot"
```

---

## 第七步：从头预训练（可选，需大量计算资源）

### 资源需求

- 论文使用 NVIDIA L40s 和 A100 GPU
- Base 模型预训练约需 150 epochs
- Large/Huge 需更多 GPU 时间

### 运行预训练

```bash
# 配置文件在 configs/experiment/paper/pretrain/
# SLURM 脚本在 slurm/pretrain/{base,large,huge}/

# 直接运行（如果不用 SLURM）
python pretrain.py experiment="paper/pretrain/large"

# 通过 SLURM 提交
sbatch slurm/pretrain/large/large.sh
```

### 预训练关键配置（论文 Table 1）

| 参数 | Audio-MAE (baseline) | Bird-MAE (ours) |
|------|---------------------|-----------------|
| 数据集 | AudioSet-2M | XCL-1.7M (curated) |
| 输入尺寸 | 1024×128 | 512×128 |
| Decoder | Swin | ViT |
| Epochs | 32 | 150 |
| Masking ratio | 0.8 | 0.75 |
| Batch size | 512 | 1024 |
| Mixup | 0 | 0.3 |
| Learning rate | 2e-4 | 2e-4 |

---

## 项目结构参考

```
Bird-MAE/
├── pretrain.py              # 预训练主脚本
├── train.py                 # 下游任务训练主脚本
├── configs/
│   └── experiment/
│       └── paper/
│           ├── pretrain/     # 预训练配置（base/large/huge）
│           ├── bigshot/      # 全量微调/probing 配置
│           └── fewshot/      # Few-shot 配置
├── util/
│   └── prepare_data/
│       ├── pretraining.py   # 预训练数据准备
│       ├── downstream.py    # 下游数据准备
│       └── fewshot.py       # Few-shot 数据准备
├── slurm/                   # SLURM 提交脚本
└── requirements.txt
```

---

## 推荐复现路线

### 最快验证路线（1-2 小时）

1. 搭建环境
2. 下载 HSN 下游数据
3. 下载 Bird-MAE-Base 权重
4. 运行 frozen + prototypical probing on HSN
5. 验证结果接近论文报告值（mAP ~43.84%）

### 标准复现路线（1-2 天）

1. 搭建环境
2. 下载全部 8 个下游数据集
3. 下载 Bird-MAE-Large 权重
4. 分别运行 fine-tuning 和 prototypical probing
5. 对比论文 Table 6 和 Table 7

### 完整复现路线（1-2 周）

1. 以上全部
2. 下载预训练数据 XCL-1.7M
3. 从头预训练 Bird-MAE-Large
4. 在所有下游任务上评估
5. 运行 few-shot benchmark

---

## 常见问题

**Q: 论文的评估指标是什么？**
主要是 class-based mean Average Precision (mAP)。也报告了 AUROC 和 Top-1 Accuracy。

**Q: HSN 在论文中作为什么角色？**
HSN 被用作开发集（validation set），所有超参数在 HSN 上调优后固定，再应用到其他数据集。

**Q: Prototypical pooling 和 prototypical probing 有什么区别？**
- Prototypical pooling（M2）：fine-tuning 时使用，替代全局平均池化，编码器参数也更新
- Prototypical probing（M3）：frozen 时使用，编码器固定，只训练 prototype 向量和分类层

**Q: 需要修改哪些路径？**
主要修改 Hydra 配置文件和 SLURM 脚本中的数据路径、输出路径和 checkpoint 路径。
