# TF-IDF 代码解释与使用说明

本说明配合 `tfidf.py` 使用，帮助你理解实现细节、验证方法以及如何按需扩展。

- 代码位置：`module/module_01/src/tfidf.py`
- 文档位置：`module/module_01/src/TFIDF_EXPLAINED.md`

## 核心目标

1. 在中文文本上计算 TF-IDF：用 jieba 分词 + scikit-learn 的向量化工具。
2. 可验证：手动 `TF * IDF` 与 `TfidfVectorizer(norm=None)` 的输出一致；再对手动结果做 L2 归一化，与 sklearn 默认设置一致。
3. 可解释：逐步打印并展示词频 TF、IDF、未归一化 TF-IDF 以及 L2 归一化后的 TF-IDF，便于理解与排查。

## 关键概念

- TF（Term Frequency）：某个词在文档中的出现次数。本实现中 `sublinear_tf=False`，即使用原始词频，不做对数缩放。
- IDF（Inverse Document Frequency）：衡量词在语料中的泛用程度，稀有词权重大。使用 sklearn 的平滑公式：
  
  $$\mathrm{idf}(t) = \log\left(\frac{1 + n_{docs}}{1 + df(t)}\right) + 1$$
  
  其中 \(df(t)\) 为包含词 t 的文档数，`smooth_idf=True` 表示 +1 平滑。
- 归一化（Normalization）：常用 L2 归一化，让向量长度为 1，便于相似度比较（余弦相似度）。

## 实现结构

`tfidf.py` 的关键步骤：

1) `tok(s)`：
   - 使用 `jieba.lcut` 对中文句子分词；
   - 过滤长度为 1 的 token 以及纯空白，减少噪声。

2) 计算未归一化 TF-IDF：
   - `TfidfVectorizer(tokenizer=tok, token_pattern=None, norm=None, smooth_idf=True, sublinear_tf=False)`；
   - `norm=None` 的输出就是“原始 tf * idf”。

3) 计算原始 TF（词频矩阵）：
   - `CountVectorizer(tokenizer=tok, token_pattern=None, vocabulary=tfidf_raw.vocabulary_)`；
   - 复用上一步的 `vocabulary_`，保证两者列索引一致。

4) 手动构造 TF-IDF 并验证：
   - `X_tfidf_manual = X_tf @ diags(idf)`，即对每一列乘以对应的 IDF；
   - 与 sklearn 的未归一化结果对比：`np.allclose(X_tfidf_raw, X_tfidf_manual)` 应为 True。

5) L2 归一化并再次验证：
   - `normalize(X_tfidf_manual, norm='l2')`；
   - 与 `TfidfVectorizer(..., norm='l2')` 的输出再次对比，应为 True。

6) Top-K 关键词：
   - 取第一篇文档（`doc_id = 0`），基于未归一化 TF-IDF 排序；
   - 打印词、TF、IDF、TF-IDF 与 L2 后的 TF-IDF，便于直观理解。

## 运行方法（Linux, bash）

确保 Python 环境可用（3.8+ 推荐），然后：

```bash
# 安装依赖（建议在虚拟环境中执行）
pip install -r requirements.txt

# 运行脚本
python3 module/module_01/src/tfidf.py
```

正常输出示例（示意）：

```
验证(tf*idf 与 sklearn 未归一化) = True
验证(L2 归一化与 sklearn 默认)   = True

Top-5 关键词:
词项	TF(词频)	IDF	TF-IDF	TF-IDF(L2)
...
```

若两行验证均为 True，说明手动计算与 sklearn 的实现严格一致。

## 自定义与扩展

- 替换或扩充 `docs`：
  - 多文档能提供更稳定的 `df` 与 `idf` 分布；
  - 可将 `docs` 改为你自己的语料列表。
- 过滤策略：
  - `tok` 中的长度过滤可以调整；若需要保留单字，可去掉 `len(w) > 1` 的限制。
- 归一化方式：
  - 将 `norm` 更换为 `'l1'` 或 `None`，观察排序变化。
- 子线性 TF：
  - 将 `sublinear_tf=True` 可使用 `1 + log(tf)`，在长文档中可缓和高词频带来的偏置。
- 停用词：
  - 可在 `tok` 中主动过滤停用词，或使用 `Vectorizer(stop_words=...)`。

## 常见问题排查

- 包缺失：若导入报错，请先 `pip install -r requirements.txt`。
- 中文切词效果：jieba 默认词库，少量专有词可能切分不佳；可考虑自定义词典 `jieba.load_userdict(...)`。
- 稀疏矩阵乘法：`X_tf @ diags(idf)` 是稀疏友好的写法；若你将其转为稠密矩阵，内存可能显著增加。

## 小结

这份实现将 TF/IDF 的概念与 sklearn 的产出逐步对齐，并提供了可复现实验（双重验证）。你可以据此快速理解、检查与调整中文语料上的 TF-IDF 计算流程。
