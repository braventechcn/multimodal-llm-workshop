#!/usr/bin/env python3
# encoding=utf-8

import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch.optim import AdamW

# Core parameter configuration (modify paths and hyperparameters according to the actual environment)
from config_lora import get_config
# CONFIG = get_config()
CONFIG = {
    "base_model_path": "/home/wyj/workspace/multimodal-llm-workshop/core/models/Qwen2.5-1.5B-Instruct",  # Base model path or model ID
    "train_data_path": "/home/wyj/workspace/multimodal-llm-workshop/core/datas/module_02/bank_intent_data/train.jsonl",  # Training set path
    "val_data_path": "/home/wyj/workspace/multimodal-llm-workshop/core/datas/module_02/bank_intent_data/val.jsonl",      # Validation set path（仅用于调参与早停，不参与最终报告）
    "test_data_path": "/home/wyj/workspace/multimodal-llm-workshop/core/datas/module_02/bank_intent_data/test.jsonl",    # Test set path（仅最终评估使用，训练过程中绝不触碰，避免数据泄露）
    "lora_save_path": "/home/wyj/workspace/multimodal-llm-workshop/module/module_02/outputs/bank_lora_model",                # LoRA weights save path (all files stored together)
    "labels": ["fraud_risk", "refund", "balance", "card_lost", "other"],  # Intent label list
    "max_seq_len": 64,  # Maximum text truncation length (adapted to corpus length distribution)
    "batch_size": 1,  # Batch size (to avoid Qwen pad_token compatibility issues)
    "epochs": 1,  # Number of training epochs (balance fitting effsect and overfitting risk)
    "lr": 2e-5,  # LoRA learning rate (based on Qwen-1.5B optimal empirical value)
    "gradient_accumulation_steps": 1,  # 为后续扩大有效批次作准备（默认=1不改变现有行为）
    "seed": 42  # 设定随机种子保证可复现
}

# Config GPU Devices (Single GPU Environment, Multi-GPU can adjust accordingly, e.g. ["0,1"])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Disable Tokenizer Parallelism Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Label mapping (text label → numeric ID, required for model input)
label2id = {label: idx for idx, label in enumerate(CONFIG["labels"])}
id2label = {idx: label for idx, label in enumerate(CONFIG["labels"])}


# Data loading and processing
def load_and_process_data():
    """加载并处理训练/验证/测试数据集。

    重要数据泄露说明：
    - 训练阶段只使用 train + validation；validation 用于监控与选择最优权重。
    - test 集仅在所有参数与模型确定后进行一次最终评估，严格避免在训练过程中引入任何 test 信息。
    """
    # 使用更明确的 data_files 字典 + split 名称，避免语义混淆
    train_ds = load_dataset("json", data_files={"train": CONFIG["train_data_path"]}, split="train")
    val_ds = load_dataset("json", data_files={"validation": CONFIG["val_data_path"]}, split="validation")
    test_ds = load_dataset("json", data_files={"test": CONFIG["test_data_path"]}, split="test")

    # 数据清洗：过滤标签不在预设集合的样本，并映射到 id（保持原批处理逻辑，不改变核心行为）
    def clean_data(examples):
        valid_mask = [label in CONFIG["labels"] for label in examples["label"]]
        return {
            "text": [text for text, mask in zip(examples["text"], valid_mask) if mask],
            "label": [label2id[label] for label, mask in zip(examples["label"], valid_mask) if mask]
        }

    train_ds = train_ds.map(clean_data, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(clean_data, batched=True, remove_columns=val_ds.column_names)
    test_ds = test_ds.map(clean_data, batched=True, remove_columns=test_ds.column_names)

    # 初始化Tokenizer（Qwen模型需手动设置pad_token）
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_seq_len"],
            padding="max_length"
        )

    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = test_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    # HuggingFace Trainer 要求标签列名为 labels
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_val = tokenized_val.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # 基本数据集规模打印（验证无异常）
    print(
        f"数据预处理完成：训练集样本数={len(tokenized_train)}，验证集样本数={len(tokenized_val)}，测试集样本数={len(tokenized_test)}"
    )
    return tokenized_train, tokenized_val, tokenized_test, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 从logits中取概率最大的类别作为预测结果
    predictions = np.argmax(logits, axis=-1)
    # 计算Macro-F1（对所有类别平等加权，适配类别分布不均衡场景）
    macro_f1 = f1_score(
        y_true=labels,
        y_pred=predictions,
        average="macro",
        labels=list(label2id.values())  # 仅计算预设标签的F1，排除无效标签干扰
    )
    return {"macro_f1": round(macro_f1, 4)}


def main():
    # 确保目标文件夹存在（避免路径错误）
    os.makedirs(CONFIG["lora_save_path"], exist_ok=True)

    # 加载预处理后的数据
    tokenized_train, tokenized_val, tokenized_test, tokenizer = load_and_process_data()

    # 基础模型（无LoRA）评估：获取 baseline F1分数
    print("\n=== 基础模型（无LoRA）评估 ===")
    # 加载基础模型（序列分类任务）
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),  # 分类任务类别数=标签数
        id2label=id2label,
        label2id=label2id,
        device_map="auto",  # 自动分配设备（优先GPU，无GPU则用CPU）
        dtype=torch.float32  # 用FP32规避FP16梯度兼容问题
    )
    # 初始化分类头权重（Qwen预训练模型无分类头，需手动初始化）
    if hasattr(base_model, "score"):
        torch.nn.init.normal_(base_model.score.weight, mean=0.0, std=0.02)

    # 基础模型评估配置（日志放入bank_lora_model/base_eval）
    # 注意：这里使用验证集进行基线评估，而不是测试集，避免在训练前窥视测试集。
    base_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(
            output_dir=f"{CONFIG['lora_save_path']}/base_eval",
            per_device_eval_batch_size=CONFIG["batch_size"],
            do_train=False,
            do_eval=True,
            remove_unused_columns=False
        ),
        eval_dataset=tokenized_val,  # 使用验证集
        compute_metrics=compute_metrics
    )
    # 执行评估并记录baseline F1
    base_eval_result = base_trainer.evaluate()
    base_val_f1 = base_eval_result["eval_macro_f1"]
    print(f"基础模型（无LoRA）验证集 Macro-F1: {base_val_f1}")

    # 释放基础模型显存（避免占用后续训练资源）
    del base_model, base_trainer
    torch.cuda.empty_cache()

    # LoRA微调训练
    print("\n=== LoRA微调训练 ===")
    # 重新加载基础模型（用于注入LoRA）
    lora_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        dtype=torch.float32
    )
    # 初始化分类头权重
    if hasattr(lora_model, "score"):
        torch.nn.init.normal_(lora_model.score.weight, mean=0.0, std=0.02)
    elif hasattr(lora_model, "classifier"):
        torch.nn.init.normal_(lora_model.classifier.weight, mean=0.0, std=0.02)

    # 自动探测分类头名称，确保参与训练并保存(让 分类头参加训练并保存)
    cls_head = "score" if hasattr(lora_model, "score") else ("classifier" if hasattr(lora_model, "classifier") else None)
    modules_to_save = [cls_head] if cls_head is not None else []

    # 配置LoRA参数（基于Qwen-1.5B最优经验值）
    lora_config = LoraConfig(
        r=8,  # LoRA秩：控制低秩矩阵维度，平衡效果与参数量
        lora_alpha=16,  # 缩放因子：alpha = 2*r，增强梯度更新幅度
        # target_modules=["q_proj", "v_proj"],  # 目标层：仅训练注意力Q/V投影层
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 更充分的注意力层
        lora_dropout=0.05,  # Dropout：降低过拟合风险
        bias="none",  # 不训练偏置参数：减少计算量
        task_type=TaskType.SEQ_CLS,  # 任务类型：序列分类（Sequence Classification）
        modules_to_save=modules_to_save  # 让分类头参与训练并随适配器保存
    )
    
    # 注入LoRA到基础模型
    lora_model = get_peft_model(lora_model, lora_config)
    
    # 双保险：显式解冻分类头
    if cls_head is not None:
        for p in getattr(lora_model, cls_head).parameters():
            p.requires_grad = True
    
    # 打印可训练参数占比（验证LoRA参数效率）
    print("LoRA参数占比：")
    lora_model.print_trainable_parameters()

    # 配置训练参数（日志放入bank_lora_model/lora_training）
    training_args = TrainingArguments(
        output_dir=f"{CONFIG['lora_save_path']}/lora_training",
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        logging_steps=50,
        eval_strategy="epoch",  # 每个 epoch 后评估验证集
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # 以 Macro-F1 选择最佳 checkpoint
        greater_is_better=True,
        fp16=False,  # 可选：如显存紧张可改用 fp16 / bf16
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],  # 默认=1，不改变现有逻辑
        seed=CONFIG["seed"]
    )

    # 初始化优化器（AdamW：适配LoRA微调的常用优化器）
    optimizer = AdamW(
        lora_model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=0.01  # 权重衰减：降低过拟合风险
    )

    # 初始化Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,  # 仅使用验证集进行周期性评估
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    # 执行LoRA微调
    train_result = trainer.train()
    # 训练结束后（已自动加载 best checkpoint），在验证集上再评估一次
    lora_val_eval = trainer.evaluate()
    lora_val_f1 = lora_val_eval["eval_macro_f1"]
    print(f"LoRA微调后（验证集）Macro-F1: {lora_val_f1}")

    # 保存LoRA权重（放入bank_lora_model根目录）
    lora_model.save_pretrained(CONFIG["lora_save_path"])
    print(f"\nLoRA权重已保存至：{CONFIG['lora_save_path']}")

    # 释放训练显存
    del lora_model, trainer, optimizer
    torch.cuda.empty_cache()

    # LoRA微调后模型评估
    print("\n=== LoRA微调后模型评估 ===")
    # 加载基础模型 + LoRA权重
    final_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        dtype=torch.float32
    )
    # 合并LoRA权重到基础模型（模拟部署场景）
    final_model = PeftModel.from_pretrained(final_model, CONFIG["lora_save_path"])
    final_model.merge_and_unload()

    # 评估微调后模型（日志放入bank_lora_model/final_eval）
    final_trainer = Trainer(
        model=final_model,
        args=TrainingArguments(
            output_dir=f"{CONFIG['lora_save_path']}/final_eval",  # 统一路径
            per_device_eval_batch_size=CONFIG["batch_size"],
            do_train=False,
            do_eval=True,
            remove_unused_columns=False
        ),
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    final_eval_result = final_trainer.evaluate()
    final_test_f1 = final_eval_result["eval_macro_f1"]
    print(f"LoRA微调后模型测试集 Macro-F1: {final_test_f1}")

    # 输出微调前后效果对比
    print("\n=== LoRA微调前后效果对比（验证 + 测试）===")
    print(f"基础模型（验证集）Macro-F1: {base_val_f1}")
    print(f"LoRA微调后（验证集）Macro-F1: {lora_val_f1}")
    print(f"验证集提升幅度: {round(lora_val_f1 - base_val_f1, 4)}")
    print(f"最终测试集 Macro-F1: {final_test_f1}  (未参与任何训练或调参阶段)" )


if __name__ == "__main__":
    main()
