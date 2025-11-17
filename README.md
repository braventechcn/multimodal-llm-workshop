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


## 致谢与版权说明
- 本项目部分代码基于相关课程授课老师提供的参考代码进行修改与优化，特此感谢！其版权归原作者所有，除非获得原作者明确授权，否则不得用于商业用途。 
- 本项目部分代码为**课程实战任务**与**课后实战任务**的具体实现。版权归仓库所属者所有，除非获得原作者明确授权，否则不得用于商业用途。 
