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

# Import unified configuration (dynamic path detection & overrides)
from qwen_lora_config import get_config, CONFIG_DESCRIPTIONS

# Load configuration (supports environment variable overrides or direct kwargs)
CONFIG = get_config()

# Config GPU Devices (Single GPU Environment, Multi-GPU can adjust accordingly, e.g. ["0,1"])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Disable Tokenizer Parallelism Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Label mapping (text label → numeric ID, required for model input)
label2id = {label: idx for idx, label in enumerate(CONFIG["labels"])}
id2label = {idx: label for idx, label in enumerate(CONFIG["labels"])}


# Data loading and processing
def load_and_process_data():
    """
    Load and preprocess dataset.
    Data Splitting Principles:
        - Only the training phase uses train + validation set; validation set is used for monitoring and selecting the best weights.
        - Test Set is only used for a final evaluation after all parameters and models are determined, strictly avoiding any test information leakage during training.
    """
    # Use a more explicit data_files dictionary + split names to avoid semantic confusion
    train_ds = load_dataset("json", data_files={"train": CONFIG["train_data_path"]}, split="train")
    val_ds = load_dataset("json", data_files={"validation": CONFIG["val_data_path"]}, split="validation")
    test_ds = load_dataset("json", data_files={"test": CONFIG["test_data_path"]}, split="test")

    # Data Cleaning: 
    # - Filter out samples with labels not in the predefined set
    # - and map to id (maintain original batch processing logic, do not change core behavior
    def clean_data(examples):
        valid_mask = [label in CONFIG["labels"] for label in examples["label"]]
        return {
            "text": [text for text, mask in zip(examples["text"], valid_mask) if mask],
            "label": [label2id[label] for label, mask in zip(examples["label"], valid_mask) if mask]
        }

    train_ds = train_ds.map(clean_data, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(clean_data, batched=True, remove_columns=val_ds.column_names)
    test_ds = test_ds.map(clean_data, batched=True, remove_columns=test_ds.column_names)

    # Initialize Tokenizer (Qwen model requires manual setting of pad_token)
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

    # HuggingFace Trainer requires the label column to be named "labels"
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_val = tokenized_val.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Basic info logging
    print(
        f"Data preprocessing completed: Training samples={len(tokenized_train)}, Validation samples={len(tokenized_val)}, Test samples={len(tokenized_test)}"
    )
    return tokenized_train, tokenized_val, tokenized_test, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # From logits, take the class with the highest probability as the prediction
    predictions = np.argmax(logits, axis=-1)
    # Calculate Macro-F1 (equally weighted across all classes, suitable for imbalanced class distributions)
    macro_f1 = f1_score(
        y_true=labels,
        y_pred=predictions,
        average="macro",
        labels=list(label2id.values())  # Ensure all labels are considered in F1 calculation
    )
    return {"macro_f1": round(macro_f1, 4)}


def main():
    # Ensure output directory exists (already created in get_config, but safe)
    os.makedirs(CONFIG["lora_save_path"], exist_ok=True)

    print("Loaded CONFIG parameters (key: value) ->")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # Load preprocessed data
    tokenized_train, tokenized_val, tokenized_test, tokenizer = load_and_process_data()

    # Base Model (without LoRA) evaluation: obtain baseline F1 score
    print("\n===== Base Model (without LoRA) Evaluation =====")
    
    # Load base model (sequence classification task)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],          # 
        trust_remote_code=True,             # Allow loading custom code from the model repository (Qwen requirement)
        num_labels=len(CONFIG["labels"]),   # set the number of labels for the sequence classification head. Determines the classification head dimension
        id2label=id2label,      # Bidirectional mapping between label names and IDs. Facilitates logging/evaluation with readable label names
        label2id=label2id,      # Bidirectional mapping between label names and IDs. Facilitates logging/evaluation with readable label names
        device_map="auto",      # Automatically allocate devices (prefer GPU, use CPU if no GPU)
        dtype=torch.float32     # Use FP32 to avoid FP16 gradient compatibility issues
    )
    # Ensure model knows the pad token id (required for batch size > 1)
    # - If not, Error: "Cannot handle batch sizes > 1 if no padding token is defined." From Transformers will be raised
    try:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    # Initialize classification head weights
    # - Qwen models use "score" as the classification head name,
    # - and the weights of the classification head were not initialized properly during pre-tarined model loading.
    if hasattr(base_model, "score"):
        torch.nn.init.normal_(base_model.score.weight, mean=0.0, std=0.02)
    elif hasattr(base_model, "classifier"):
        torch.nn.init.normal_(base_model.classifier.weight, mean=0.0, std=0.02)

    # Base Model Evaluation Configuration (logs saved in bank_lora_model/base_eval)
    # Notes: Here we use the validation set for baseline evaluation instead of the test set to avoid peeking at the test set before training.
    base_eval_dir = f"{CONFIG['lora_save_path']}/base_eval"
    os.makedirs(base_eval_dir, exist_ok=True)
    base_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(
            output_dir=base_eval_dir,
            per_device_eval_batch_size=CONFIG["eval_batch_size"],
            do_train=False,                 # disable training
            do_eval=True,                   # only evaluation
            remove_unused_columns=False,    # keep all columns for evaluation
            report_to="none"                # disable logging to external systems
        ),
        eval_dataset=tokenized_val,         # Use validation set for baseline evaluation
        compute_metrics=compute_metrics     # Metric computation function
    )
    # Evaluate base model
    base_eval_result = base_trainer.evaluate()
    # Log and save evaluation results
    base_trainer.log_metrics("eval", base_eval_result)  # Log metrics to the Trainer's logger
    base_trainer.save_metrics("eval", base_eval_result) # Save metrics to a JSON file
    base_trainer.save_state()
    # (Optional) If need save Base Model's prediction and hat{y}, Note: There is no original text in 'tokenized_val' anymore.）
    base_pred = base_trainer.predict(tokenized_val)
    np.savez(
        os.path.join(base_eval_dir, "predictions.npz"),
        predictions=base_pred.predictions,  # logits
        label_ids=base_pred.label_ids
    )
    base_val_f1 = base_eval_result["eval_macro_f1"]
    print(f"Base Model (without LoRA) Valiation Macro-F1: {base_val_f1}")

    # Release base model GPU memory, avoid memory overflow during LoRA fine-tuning
    del base_model, base_trainer
    torch.cuda.empty_cache()

    # LoRA Fine-Tuning
    print("\n===== LoRA Fine-Tuning =====")
    # Reload base model (for LoRA injection)
    lora_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        dtype=torch.float32
    )
    # Ensure model knows the pad token id
    try:
        lora_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(lora_model, "generation_config") and lora_model.generation_config is not None:
            lora_model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    # Initialize classification head weights
    if hasattr(lora_model, "score"):
        torch.nn.init.normal_(lora_model.score.weight, mean=0.0, std=0.02)
    elif hasattr(lora_model, "classifier"):
        torch.nn.init.normal_(lora_model.classifier.weight, mean=0.0, std=0.02)

    # Autocheck classification head name, ensuring it can be trained and saved
    cls_head = "score" if hasattr(lora_model, "score") else ("classifier" if hasattr(lora_model, "classifier") else None)
    modules_to_save = [cls_head] if cls_head is not None else []

    # Config LoRA Args (based on Qwen-1.5B best experience)
    lora_config = LoraConfig(
        r=8,                            # LoRA rank controls the capacity of the adaptation
        lora_alpha=16,                  # scaling factor 'alpha' = 2*r to balance the gradient update scale
        # target_modules=["q_proj", "v_proj"],  # target modules for LoRA injection (only q_proj and v_proj))
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # target modules for LoRA injection (all attention projections)
        lora_dropout=0.05,              # Dropout rate for LoRA layers to mitigate overfitting
        bias="none",                    # Do not add additional bias parameters, keep model size minimal
        task_type=TaskType.SEQ_CLS,     # Task Type: Sequence Classification
        modules_to_save=modules_to_save # Ensure classification head is trained and saved with the adapter
    )
    
    # Inject LoRA into the base model
    lora_model = get_peft_model(lora_model, lora_config)
    
    # Double check: explicitly unfreeze classification head
    if cls_head is not None:
        for p in getattr(lora_model, cls_head).parameters():
            p.requires_grad = True # ensure classification head is trainable
    
    # Print trainable parameters ratio (to verify LoRA parameter efficiency)
    print("LoRA parameter ratio:")
    lora_model.print_trainable_parameters()

    # Config Traing Args （The logger will be saved in bank_lora_model/lora_training）
    training_args = TrainingArguments(
        output_dir=f"{CONFIG['lora_save_path']}/lora_training",
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        num_train_epochs=CONFIG["epochs"],  # Number of training epochs
        learning_rate=CONFIG["lr"],
        logging_steps=50,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # 以 Macro-F1 选择最佳 checkpoint
        greater_is_better=True,
        fp16=False,  # 可选：如显存紧张可改用 fp16 / bf16
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],  # Gradient accumulation steps
        seed=CONFIG["seed"]
    )

    # Initialize optimizer (AdamW: commonly used optimizer adapted for LoRA fine-tuning)
    optimizer = AdamW(
        lora_model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=0.01  # Weight decay to reduce overfitting risk
    )

    # Initialize Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,  # Only use validation set for periodic evaluation
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)    # Trainer(..., optimizers=(optimizer, lr_scheduler)): 自定义优化器，禁用学习率调度器 lr_scheduler = None
    )

    # Start LoRA fine-tuning
    train_result = trainer.train()
    # Log and save training metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    # Evaluate LoRA fine-tuned model on validation set
    print("\n===== LoRA Fine-Tuned Model Evaluation on Validation Set =====")
    lora_val_eval = trainer.evaluate()
    lora_val_f1 = lora_val_eval["eval_macro_f1"]
    print(f"LoRA Fine-Tuned (Validation Set) Macro-F1: {lora_val_f1}")

    # # Save LoRA Weights（in bank_lora_model root path）
    # lora_model.save_pretrained(CONFIG["lora_save_path"])
    # print(f"\nLoRA weights have been saved：{CONFIG['lora_save_path']}")
    
    # Using the best Trainer recorded best checkpoint path to avoid duplicate saving
    best_ckpt_dir = trainer.state.best_model_checkpoint
    print(f"Best LoRA checkpoint: {best_ckpt_dir}")

    # Release LoRA fine-tuning GPU memory
    del lora_model, trainer, optimizer
    torch.cuda.empty_cache()

    # LoRA Fine-Tuned Model Final Evaluation in Test Set
    print("\n===== LoRA Fine-Tuned Model Evaluation on Test Set =====")
    # Load base model again
    final_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        dtype=torch.float32
    )
    # Ensure model knows the pad token id
    try:
        final_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(final_model, "generation_config") and final_model.generation_config is not None:
            final_model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    # Merge LoRA weights into base model (simulate deployment scenario)
    final_model = PeftModel.from_pretrained(final_model, best_ckpt_dir)
    # final_model = PeftModel.from_pretrained(final_model, CONFIG["lora_save_path"])
    final_model.merge_and_unload()

    # Evaluate fine-tuned model (logs saved in bank_lora_model/final_eval)
    final_trainer = Trainer(
        model=final_model,
        args=TrainingArguments(
            output_dir=f"{CONFIG['lora_save_path']}/final_eval",
            per_device_eval_batch_size=CONFIG["test_batch_size"],
            do_train=False,
            do_eval=True,
            remove_unused_columns=False
        ),
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    final_eval_result = final_trainer.evaluate()
    final_test_f1 = final_eval_result["eval_macro_f1"]
    print(f"LoRA Fine-Tuned (Test Set) Macro-F1: {final_test_f1}")

    # Optional: Save final model predictions
    print("\n=== Comparison of LoRA before and after Fine-Tuning (Valuation & Test)===")
    print(f"Base Model (Validation) Macro-F1: {base_val_f1}")
    print(f"LoRA Fine-Tuned (Validation) Macro-F1: {lora_val_f1}")
    print(f"Validation Improvement: {round(lora_val_f1 - base_val_f1, 4)}")
    print(f"Final Test Macro-F1: {final_test_f1}  (Not involved in any training or tuning phase)" )


if __name__ == "__main__":
    main()
