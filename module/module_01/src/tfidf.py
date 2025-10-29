# encoding=utf-8
#!/usr/bin/env python3
"""
本脚本演示如何在中文文本上计算与验证 TF-IDF。

实现要点：
1) 使用 jieba 进行中文分词，得到 token 序列；
2) 用 CountVectorizer 得到原始 TF（词频）；
3) 用 TfidfVectorizer 拟合得到 IDF，并设置 norm=None 以获得“未归一化”的 tf*idf；
4) 手动将 TF 与 IDF 相乘，验证与 sklearn 未归一化结果一致；
5) 对手动结果做 L2 归一化，验证与 sklearn 默认设置（norm='l2'）一致；
6) 打印首篇文档的 Top-K 关键词，展示 TF、IDF、TF-IDF 以及 L2 归一化后的值。

关键公式（与 sklearn 一致，smooth_idf=True）：
    idf(t) = log((1 + n_docs) / (1 + df(t))) + 1
其中 df(t) 为包含词项 t 的文档数。
"""

import jieba
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import diags

# 示例文档（可改为多文档列表）
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
    """将中文句子切分为词 tokens。

    说明：
    - 使用 jieba.lcut 进行中文分词；
    - 过滤掉长度为 1 的 token（如标点或单字）和纯空白；
    - 返回用于向量化器（Vectorizer）的 token 列表。
    """
    return [w for w in jieba.lcut(s) if len(w) > 1 and w.strip()]

def topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """返回一维数组 arr 中数值最大的前 k 个索引（降序）。

    实现细节：
    - 使用 np.argpartition 获取无序的 Top-K 索引，时间复杂度优于完整排序；
    - 再对 Top-K 片段做二次排序，得到降序排列的真实 Top-K 索引；
    - 若 k 超过非零元素个数或数组长度，自动截断。
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

    print(f"验证(tf*idf 与 sklearn 未归一化) = {verify_raw}")
    print(f"验证(L2 归一化与 sklearn 默认)   = {verify_l2}")

    # 5) 展示首篇文档的 Top-K 关键词（基于未归一化 TF-IDF 排序）
    doc_id = 0
    tf_row        = X_tf[doc_id].toarray()[0]               # 原始词频
    tfidf_row     = X_tfidf_manual[doc_id].toarray()[0]     # 未归一化 TF-IDF
    tfidf_row_l2  = X_tfidf_manual_l2[doc_id].toarray()[0]  # L2 归一化后的 TF-IDF

    idx = topk_indices(tfidf_row, top_k)

    print("\nTop-{} 关键词:".format(len(idx)))
    print("词项\tTF(词频)\tIDF\tTF-IDF\tTF-IDF(L2)")
    for i in idx:
        print(
            f"{vocab[i]}\t{int(tf_row[i])}\t\t{idf[i]:.3f}\t{tfidf_row[i]:.3f}\t{tfidf_row_l2[i]:.3f}"
        )

if __name__ == "__main__":
    main()