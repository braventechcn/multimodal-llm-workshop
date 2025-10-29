# multimodal-llm-workshop

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
```

### 运行项目(更新中)

```bash

```

### 验证安装

```python
# 在 Python 中测试关键包是否正常导入
python -c "import torch; import transformers; print('环境配置成功！')"
```

