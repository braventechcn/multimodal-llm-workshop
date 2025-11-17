# multimodal-llm-workshop

## 项目结构

```
multimodal-llm-workshop/
├── core/
│   ├── datas/
│   │   ├── module_01/
│   │   │   └── NLP-text_classification/
│   │   │       └── training.csv
│   │   └── module_02/
│   │       ├── mini_translation_pairs.txt
│   │       └── bank_intent_data/
│   │           ├── train.jsonl
│   │           ├── val.jsonl
│   │           └── test.jsonl
│   └── models/
│       ├── Qwen2.5-1.5B-Instruct/         # 基座大模型（本地权重与分词器）
│       └── bge-large-zh-v1.5/             # 中文向量模型（可用于嵌入/检索）
├── module/
│   ├── module_01/                         # Module 1: Fundamentals of ML & DL
│   │   ├── src/
│   │   │   ├── linear_regression.py       # 线性回归示例
│   │   │   ├── pca.py                     # 主成分分析示例
│   │   │   ├── tfidf.py                   # 文本特征工程（TF‑IDF）
│   │   │   ├── rl.py                      # 强化学习入门实验
│   │   │   └── Assignment_*               # 课程作业代码
│   │   ├── img_classifying.py             # CV 图像分类小实验
│   │   ├── text_sum.py                    # NLP 文本摘要小实验
│   │   └── outputs/                       # 训练/预测产物（如 checkpoints）
│   └── module_02/                         # Module 2: Large Model Fine-tuning & Deployment
│       ├── src/
│       │   ├── qwen2.5_1.5b_lora.py       # Qwen2.5-1.5B LoRA 微调主脚本
│       │   ├── config_lora.py             # LoRA 训练配置（路径与超参；可任意目录运行）
│       │   ├── mini_transformer.py        # 教学用小型 Transformer 实现
│       │   ├── mini_decoder_only.py       # 迷你 Decoder-only 模型
│       │   ├── train_transformer.py       # 训练脚本（Encoder‑Decoder）
│       │   └── train_decoder_only.py      # 训练脚本（Decoder‑only）
│       └── outputs/
│           ├── bank_lora_model/           # LoRA 适配器权重与评测结果
│           ├── ckpt/                      # 小模型 checkpoint（Encoder‑Decoder）
│           └── ckpt_decoder_only/         # 小模型 checkpoint（Decoder‑only）
├── requirements.txt
└── README.md

持续更新 ...
```

### Module 1: Fundamentals of Machine Learning and Deep Learning
- 核心目标：夯实传统机器学习与深度学习基础，覆盖线性回归、主成分分析、特征工程（TF‑IDF）、强化学习入门等。
- 代码位置：`module/module_01/`
- 数据示例：`core/datas/module_01/`
- 产物目录：`module/module_01/outputs/`

### Module 2: Large Model Fine-tuning and Deployment
- 核心目标：以 Qwen2.5-1.5B 为例，完成意图分类任务的 LoRA 微调与评测，理解部署前后流程；同时包含简单 Transformer/Decoder-only 实现脚本。
- 代码位置：`module/module_02/src/`
	- `qwen2.5_1.5b_lora.py`：LoRA 微调主入口；
	- `qwen_lora_config.py`：统一配置，自动解析仓库根路径，支持环境变量/参数覆盖；
	- `mini_transformer.py`、`mini_decoder_only.py` 及相应训练脚本
- 数据位置：`core/datas/module_02/bank_intent_data/`（`train/val/test.jsonl`）。
- 模型权重：`core/models/Qwen2.5-1.5B-Instruct/`。
- 产物目录：`module/module_02/outputs/`（包含 `bank_lora_model/` 与小模型 ckpt）。

运行 LoRA 微调（可在任意目录执行，配置由 `config_lora.py` 动态解析）：
```bash
# 示例：在仓库根目录
python module/module_02/src/qwen2.5_1.5b_lora.py

# 或绝对路径调用
python /home/<user>/workspace/multimodal-llm-workshop/module/module_02/src/qwen2.5_1.5b_lora.py
```


## 项目环境设置指南

### 创建虚拟环境

#### 方法一：使用 venv（推荐）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

#### 方法二：使用 conda

```bash
# 创建虚拟环境
conda create -n multimodal-env python=3.10

# 激活虚拟环境
conda activate multimodal-env
```

### 安装依赖

```bash
pip install --upgrade pip
# 如果项目有 requirements.txt
pip install -r requirements.txt

# 或者逐个安装主要依赖
pip install torch torchvision
pip install transformers
pip install datasets
pip install jupyter notebook
...
```

### 验证安装

```python
# 在 Python 中测试关键包是否正常导入
python -c "import torch; import transformers; print('环境配置成功！')"
```

### 运行项目(更新中)

```bash

```



## 致谢与版权说明
- 本项目部分代码基于相关课程授课老师提供的参考代码进行修改与优化，特此感谢！其版权归原作者所有，除非获得原作者明确授权，否则不得用于商业用途。 
- 本项目部分代码为**课程实战任务**与**课后实战任务**的具体实现。版权归仓库所属者所有，除非获得原作者明确授权，否则不得用于商业用途。 
