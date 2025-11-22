# encoding=utf-8
#!/usr/bin/env python3
"""中文文本 TF-IDF 计算与验证示例。

步骤总览（保持教学、不过度封装，便于逐行理解）：
1. 分词：使用 jieba 获取中文 token 列表。
2. 原始 TF：用 CountVectorizer 按与 TF-IDF 相同的词表统计词频矩阵。
3. 未归一化 TF-IDF（tf * idf）：TfidfVectorizer(norm=None) 拟合得到 IDF 与 tf*idf 数值。
4. 手动验证：使用 TF 矩阵 * 对角 IDF 验证与 sklearn 未归一化输出一致。
5. 归一化验证：手动对 tf*idf 做 L2 归一化，与 sklearn 默认 norm='l2' 结果比对。
6. 关键词展示：针对首篇文档按未归一化 tf*idf 排序取 Top-K，打印多列对照。

关键公式（smooth_idf=True 与 sklearn 一致）：
    idf(t) = log((1 + n_docs) / (1 + df(t))) + 1
其中 df(t) 是包含词项 t 的文档数。平滑项避免 df=0 或极端值。

设计目标：示例明确、数值可验证、输出结构清晰；不对核心逻辑做多层封装，便于学习。
"""

import jieba
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import diags

# =============================
# 示例文档（可改为更大/外部语料）
# =============================
docs = [
    # 尝试覆盖“屏幕/电池/主板/进水/维修/数据恢复/评估回收”等主题词，
    # 以获得更稳健的文档频率 df，从而使 IDF 更有区分度。
    "苹果手机维修服务，提供屏幕更换、电池更换、主板维修与进水处理",
    "iPhone 13 更换电池价格与保修政策，支持上门维修",
    "苹果授权服务中心查询与预约说明",
    "iPad 屏幕触控失灵的常见原因与处理",
    "MacBook 进水后不开机的抢救与主板维修建议",
    "安卓手机屏幕摔碎怎么办，官方与第三方维修对比",
    "手机数据恢复与备份：误删照片如何找回",
    "电脑主板故障诊断：不通电、反复重启的常见原因",
    "智能手表电池续航测试与表带更换指南",
    "二手手机评估、回收定价与隐私清除流程",
]

top_k = 5

def tok(s: str) -> List[str]:
    """中文分词函数。

    策略：
    - 使用 jieba.lcut 做精确模式分词；
    - 过滤长度为 1 的 token（多为单字或符号，对 TF-IDF 贡献低）；
    - 去除纯空白；
    返回：用于后续 Count/TfidfVectorizer 的 token 列表。
    """
    return [w for w in jieba.lcut(s) if len(w) > 1 and w.strip()]

def topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """获取一维数组中 Top-K 最大值的索引（降序）。

    算法说明：
    - np.argpartition 仅定位 Top-K 位置，复杂度优于完整排序；
    - 对 Top-K 片段再次按值排序得到最终降序索引；
    - 自动处理 k 超界或有效元素不足的情况。
    """
    k = min(k, (arr != 0).sum() if arr.ndim == 1 else arr.shape[-1])
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(arr, -k)[-k:]
    return part[np.argsort(arr[part])[::-1]]

def main():
    """主流程：拟合、验证并展示 TF-IDF 结果。"""
    # 1) 拟合“未归一化”的 TF-IDF（便于直接对照 tf * idf）
    tfidf_raw = TfidfVectorizer(
        tokenizer=tok,          # tokenizer=tok 使用我们自定义的中文分词函数
        token_pattern=None,     # token_pattern=None 禁用默认的正则分词（否则会覆盖 tokenizer）
        norm=None,              # norm=None 表示不进行向量归一化，此时输出就是“原始 tf * idf”
        smooth_idf=True,        # smooth_idf=True 使用平滑 IDF：idf = log((1+n)/(1+df)) + 1
        sublinear_tf=False,     # sublinear_tf=False 表示 TF 不做 log 缩放，直接使用原始词频
        dtype=np.float32,
    )
    # 稀疏矩阵形状：[n_docs, n_terms]
    X_tfidf_raw = tfidf_raw.fit_transform(docs)
    vocab = tfidf_raw.get_feature_names_out()  # 词表（索引 -> 词项）
    idf = tfidf_raw.idf_                      # 每个词项的 IDF 值（与 vocab 对齐）

    # 2) 用完全相同的词表计算原始 TF（词频矩阵）
    # 说明：
    # - 指定 vocabulary=tfidf_raw.vocabulary_，保证列索引与上面的 TfidfVectorizer 对齐；
    counter = CountVectorizer(
        tokenizer=tok,
        token_pattern=None,
        vocabulary=tfidf_raw.vocabulary_,
        dtype=np.int32,
    )
    X_tf = counter.transform(docs).astype(np.float32)  # 词频矩阵（稀疏）

    # 3) 手动构造 TF-IDF = TF * IDF，并与 sklearn 未归一化结果比对
    # - 这里使用稀疏对角矩阵 diags(idf) 实现“对每一列乘以对应的 idf”
    X_tfidf_manual = X_tf @ diags(idf)
    verify_raw = np.allclose(
        X_tfidf_raw.toarray(), X_tfidf_manual.toarray(), atol=1e-6
    )

    # 4) 对手动 TF-IDF 做 L2 归一化，并验证与 sklearn 默认设置（norm='l2'）一致
    X_tfidf_manual_l2 = normalize(X_tfidf_manual, norm='l2')
    tfidf_l2 = TfidfVectorizer(
        tokenizer=tok,
        token_pattern=None,
        norm='l2',
        smooth_idf=True,
        sublinear_tf=False,
        dtype=np.float32,
    )
    X_tfidf_sklearn_l2 = tfidf_l2.fit_transform(docs)
    verify_l2 = np.allclose(
        X_tfidf_sklearn_l2.toarray(), X_tfidf_manual_l2.toarray(), atol=1e-6
    )

    # ========== 验证输出 ==========
    print("\n[验证结果]")
    print(f"未归一化 tf*idf 与 sklearn 对比一致: {verify_raw}")
    print(f"L2 归一化与 sklearn 默认结果一致 : {verify_l2}")
    print(f"文档数: {X_tf.shape[0]} | 词项数: {X_tf.shape[1]}")

    # 5) 展示首篇文档的 Top-K 关键词（基于未归一化 TF-IDF 排序）
    doc_id = 0
    tf_row        = X_tf[doc_id].toarray()[0]               # 原始词频
    tfidf_row     = X_tfidf_manual[doc_id].toarray()[0]     # 未归一化 TF-IDF
    tfidf_row_l2  = X_tfidf_manual_l2[doc_id].toarray()[0]  # L2 归一化后的 TF-IDF

    idx = topk_indices(tfidf_row, top_k)

    print(f"\n[Top-{len(idx)} 关键词 - 文档0 基于未归一化 TF-IDF]")
    header = f"{'词项':<12}{'TF':>4}{'IDF':>10}{'TF-IDF':>12}{'TF-IDF(L2)':>14}"
    print(header)
    print('-' * len(header))
    for i in idx:
        print(f"{vocab[i]:<12}{int(tf_row[i]):>4}{idf[i]:>10.3f}{tfidf_row[i]:>12.3f}{tfidf_row_l2[i]:>14.3f}")
    print('\n说明: TF-IDF(L2) 为向量整体 L2 归一化后的数值，便于余弦/距离度量。')

if __name__ == "__main__":
    main()