# -*- coding: utf-8 -*-
import os, sys, json
import torch
import numpy as np
from PIL import Image
import faiss
import open_clip

# 数据路径（image和caption）
IMG_DIR  = "/Users/zhaoshuai/Downloads/train2017"
CAP_JSON = "/Users/zhaoshuai/Downloads/annotations/captions_preview.json"

# 读取JSON，聚合相同image_id的captions
def load_pairs(caption_json_path, img_dir):
    with open(caption_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        anns = data["annotations"]
    elif isinstance(data, list):
        anns = data
    else:
        raise ValueError("JSON 格式不对")

    buckets = {}
    for a in anns:
        iid = a["image_id"]
        cap = a.get("caption", "").strip()
        if not cap:
            continue
        buckets.setdefault(iid, []).append(cap)

    pairs = []
    for iid, caps in buckets.items():
        fpath = os.path.join(img_dir, f"{iid:012d}.jpg")
        if os.path.exists(fpath):
            pairs.append({"image_id": iid, "path": fpath, "captions": caps})

    if not pairs:
        print("没有找到有效的‘图片-caption’对，请检查路径")
        sys.exit(1)

    return pairs

# 加在OpenCLIP模型与预处理，优先选择GPU
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess, device

# 对图像库中的所有图片提取、归一化
def encode_gallery(pairs, model, preprocess, device):
    img_feats, img_meta = [], []
    with torch.no_grad():
        for p in pairs:
            img = Image.open(p["path"]).convert("RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_feats.append(feat.cpu().numpy()[0])
            img_meta.append(p)
    img_feats = np.stack(img_feats).astype("float32")
    return img_feats, img_meta

# 建立FAISS内积索引
def build_index(img_feats):
    d = img_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_feats)
    return index

# 对要查询的图片提取归一化向量
def encode_query_image(query_path, model, preprocess, device):
    if not os.path.exists(query_path):
        print(f"错误: 查询图片 {os.path.basename(query_path)} 不存在于 {os.path.dirname(query_path)}")
        sys.exit(1)
    with torch.no_grad():
        qimg = Image.open(query_path).convert("RGB")
        qimg_t = preprocess(qimg).unsqueeze(0).to(device)
        qfeat = model.encode_image(qimg_t)
        qfeat = qfeat / qfeat.norm(dim=-1, keepdim=True)
        qfeat = qfeat.cpu().numpy().astype("float32")
    return qfeat

# 检索并且打印结果
def search_and_print(index, qvec, img_meta, top_k):
    if top_k < 1:
        print("TopK 必须 >= 1")
        sys.exit(1)
    if top_k > len(img_meta):
        print(f"警告: TopK={top_k} 超过库大小({len(img_meta)}), 自动设置为 {len(img_meta)}")
        top_k = len(img_meta)

    D, I = index.search(qvec, k=top_k)
    print("\n检索结果 TopK:")
    for rank, idx in enumerate(I[0]):
        meta = img_meta[idx]
        print(f"#{rank+1} | {meta['path']} | caption示例: {meta['captions'][0]} | 相似度={D[0][rank]:.4f}")

def main():
    # 命令行参数
    if len(sys.argv) < 3:
        print("用法: python search_demo.py <图片文件名> <TopK>")
        sys.exit(1)

    query_fname = sys.argv[1]
    top_k = int(sys.argv[2])

    # 加载数据
    pairs = load_pairs(CAP_JSON, IMG_DIR)
    query_path = os.path.join(IMG_DIR, query_fname)
    print(f"查询图片: {query_path}, TopK={top_k}")
    print(f"[INFO] 样本图像数: {len(pairs)}")

    # 模型与索引
    model, preprocess, device = load_model()
    img_feats, img_meta = encode_gallery(pairs, model, preprocess, device)
    index = build_index(img_feats)

    # 查询与检索
    qvec = encode_query_image(query_path, model, preprocess, device)
    search_and_print(index, qvec, img_meta, top_k)

if __name__ == "__main__":
    main()