# Task Description and Code Running Guide

## 1. 任务简介
本项目脚本 `text_classification.py` 用于 **企业类型文本分类**。支持两种特征与分类范式：
- 传统特征: TF-IDF（可选分词方式：jieba分词或字符 n-gram） + 线性分类器（LinearSVC / LogisticRegression）
- 预训练表示: BERT 句向量（CLS / Mean Pool） + 线性分类器（LinearSVC / LogisticRegression）

数据文件格式：CSV 或 JSONL。要求包含两列/字段：
- `label`：类别标签（整数，目标预期 0..9，但当前数据中有 11 类）
- `text`：文本内容（中文描述）

脚本输出包括：
1. Hold-out（默认 80/20）验证的准确率、分类报告、混淆矩阵。
2. K 折（默认 5 折）交叉验证的均值和标准差。
3. 可选保存最终训练好的流水线模型（`joblib` 序列化）。
4. 返回码：CV 均值准确率 >= 0.80 则退出码 0，否则 1（便于 CI 检查）。

## 2. 核心功能结构与原理
### 2.1 加载数据 `load_dataset`
- 自动识别扩展名：`.csv` 使用 `pandas.read_csv`；`.jsonl` 按行解析 JSON。
- 自动尝试无表头 / 有表头两种 CSV 结构，提高兼容性。
- 标签安全转换 `_safe_int`，过滤无法转换的行、空文本行。
- 返回 `X (np.ndarray[str])` 与 `y (np.ndarray[int])`。

### 2.2 BERT 特征抽取 `BertEncoder`
- 惰性加载：首次调用 `transform` 时才加载 `AutoTokenizer` 与 `AutoModel`。
- 批处理：按 `batch_size` 切分文本，降低显存峰值。
- 池化策略：
  - CLS 池化：取 `last_hidden_state[:,0]` 作为句向量。
  - Mean 池化：用 `attention_mask` 屏蔽 padding 后对真实 token 求平均。
- 输出：形状 `(n_samples, hidden_size)` 的 `float32` 矩阵，供 sklearn 分类器使用。

### 2.3 构建流水线 `build_pipeline`
根据参数选择：
- 若 `use_bert=True`：Pipeline = `[('embed', BertEncoder), ('clf', classifier)]`
- 否则：
  - 分词路径：结巴分词或字符 n-gram。
  - TF-IDF 特征：`TfidfVectorizer`（可控 n-gram 与 `max_features`）。
  - 分类器：`LinearSVC(C)` 或 `LogisticRegression(C, solver, max_iter)`。

流水线优点：
- 统一接口：`fit/predict` 可直接用于 `cross_val_score`。
- 避免数据泄漏：所有预处理放在 Pipeline 中。
- 模型保存简单：直接序列化整条 Pipeline。

### 2.4 模型评估
- Hold-out：使用 `train_test_split(stratify=y)` 保持类别分布区域一致。
- CV：`StratifiedKFold(shuffle=True, random_state)` 提供更稳健的泛化估计。
- 指标：默认仅准确率，可根据需要扩展（如 F1、宏平均）。

### 2.5 退出码与阈值
- 交叉验证均值准确率 >= 0.80 判定为 PASS（退出码 0）；否则 FAIL（退出码 1）。
- 若当前数据类别不止 10 类，可调整阈值或清洗数据。

## 3. 运行环境依赖
`requirements.txt` 主要需：
- 传统 ML：`numpy`, `pandas`, `scikit-learn`, `jieba`
- 深度学习：`torch`, `transformers`
- 评估与工具：`tqdm`, `joblib`

安装示例：
```bash
pip install -r requirements.txt
```

## 4. 运行命令示例
### 4.1 默认 TF-IDF + SVM（jieba分词）
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py
```

### 4.2 字符 n-gram TF-IDF + LinearSVC
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py \
  --use-char-ngrams --ngram-min 1 --ngram-max 3 --max-features 50000
```

### 4.3 TF-IDF + LogisticRegression 调参
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py \
  --classifier logreg --logreg-C 2.0 --logreg-max-iter 1500 --logreg-solver lbfgs
```

### 4.4 BERT + LinearSVC（CLS 池化）
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py \
  --use-bert --bert-model-name bert-base-chinese --bert-batch-size 8
```

### 4.5 BERT + LogisticRegression + Mean Pool
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py \
  --use-bert --bert-mean-pool --classifier logreg --logreg-C 1.5
```

### 4.6 指定自定义数据路径（CSV/JSONL）
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py --data-path /path/to/data.jsonl
```

### 4.7 保存最终模型
```bash
./module/module_01/src/Assignment_1_NLP/text_classification.py --save-model
```
模型会保存到：`module/module_01/outputs/checkpoints/text_classifier_<mode>_<clf>.joblib`

## 5. 参数说明速览
| 参数 | 说明 | 示例 |
|------|------|------|
| `--use-char-ngrams` | 使用字符 n-gram TF-IDF | `--use-char-ngrams` |
| `--ngram-min / --ngram-max` | n-gram 范围 | `--ngram-min 1 --ngram-max 3` |
| `--max-features` | TF-IDF 词表上限 | `--max-features 50000` |
| `--use-bert` | 切换 BERT 模式 | `--use-bert` |
| `--bert-model-name` | HuggingFace 模型名 | `--bert-model-name bert-base-chinese` |
| `--bert-max-length` | BERT 截断长度 | `--bert-max-length 192` |
| `--bert-batch-size` | 抽取批大小 | `--bert-batch-size 8` |
| `--bert-mean-pool` | 使用均值池化 | `--bert-mean-pool` |
| `--classifier` | 分类器类型 | `--classifier logreg` |
| `--C` | LinearSVC 正则 C | `--C 0.5` |
| `--logreg-C` | Logistic 回归 C | `--logreg-C 2.0` |
| `--logreg-max-iter` | Logistic 最大迭代 | `--logreg-max-iter 1500` |
| `--logreg-solver` | Logistic 求解器 | `--logreg-solver saga` |
| `--cv` | 交叉验证折数 | `--cv 10` |
| `--save-model` | 持久化模型 | `--save-model` |

## 6. transform 函数关键设计解析
`BertEncoder.transform` 核心流程：
1. 惰性加载模型与分词器（减少未使用时的启动成本）。
2. 输入归一化：兼容多种序列类型，统一转 `list[str]`，提升健壮性。
3. 文本清洗：过滤空值与纯空白，避免浪费计算。
4. 批处理：通过 `batch_size` 控制显存与速度平衡。
5. Tokenization：自动补齐 + 截断，保持批内长度统一。
6. 前向计算：`torch.no_grad()` 避免梯度存储。
7. 池化策略：CLS 与 Mean 二选一，可扩展为拼接或注意力加权。
8. 数据回收：转 CPU + `float32` 减少内存占用。
9. 拼接输出：`np.vstack` 得到规范的二维特征矩阵。

优势：
- 与 sklearn Pipeline 完美兼容（返回 numpy 矩阵）。
- 支持 BERT 与传统 TF-IDF 直接互换，方便实验对比。
- 容易扩展更多预训练模型（替换 `model_name` 即可）。

## 7. 结果判定与调优建议
- 如果准确率 < 0.80：
  - 增加 `--bert-max-length`（信息保留更充分）。
  - 调整分类器超参（提高 `--logreg-C` 或使用 `--C` 微调 SVM）。
  - 尝试 `--bert-mean-pool` 在较长文本或分布不均时的稳定性。
  - 使用字符 n-gram：`--use-char-ngrams`（加大覆盖，适合噪声拼写）。

- 如果出现类别预测稀疏：
  - 引入 `class_weight='balanced'`（需在代码中扩展参数）。
  - 检查原始数据分布是否严重不均。 

## 8. 常见问题与排查
| 问题 | 可能原因 | 解决建议 |
|------|----------|----------|
| BERT 下载失败 | 外网受限 | 使用镜像 `https://hf-mirror.com/` 或本地缓存 `local_files_only=True` |
| 精度低 | 数据类别多于预期或噪声高 | 清洗数据 / 调参 / 使用 BERT |
| 内存溢出 | `bert-max-length` 太大 + 批次过大 | 减小 `--bert-batch-size` 或长度 |
| 训练速度慢 | CPU 环境跑 BERT | 切换 GPU / 使用更小模型 / 降低长度 |

## 9. 扩展方向
- 支持多特征融合：BERT + TF-IDF 通过 `FeatureUnion` 拼接。
- 添加缓存：对已处理文本哈希缓存向量，加速多次 CV。
- 半精度：在 GPU 上使用 `torch.autocast` 或手动 `half()` 降低显存。
- 自动调参：结合 `GridSearchCV` 使用参数前缀 `embed__` / `tfidf__` / `clf__`。

## 10. 快速验证与调参示例
```bash
# 基线：结巴 + SVM
./module/module_01/src/Assignment_1_NLP/text_classification.py

# BERT + SVM 提升表示能力
./module/module_01/src/Assignment_1_NLP/text_classification.py --use-bert --bert-batch-size 8

# BERT + LogisticRegression + Mean Pool
./module/module_01/src/Assignment_1_NLP/text_classification.py --use-bert --bert-mean-pool --classifier logreg --logreg-C 2.0

# 提升长度 + 保存模型
./module/module_01/src/Assignment_1_NLP/text_classification.py --use-bert --bert-max-length 192 --save-model
```

## 11. 退出码引用（CI/CD）
- 成功（>= 0.80 CV 均值）：`echo $?` 为 0
- 失败：`echo $?` 为 1，可在 CI 中用于质量门控。
