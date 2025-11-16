# module_02 运行命令速查

- 工作目录: `module/module_02`
- 依赖安装（从仓库根目录执行）:
  
  ```bash
  pip install -r requirements.txt
  ```

- 语料文件: `src/mini_translation_pairs.txt`
- 设备: 自动使用可用的 GPU（CUDA）；若需强制 CPU：
  
  ```bash
  export CUDA_VISIBLE_DEVICES=""
  ```

## Decoder-Only 语言模型（GPT 风格）
- 脚本: `src/train_decoder_only.py`
- 主要参数: `--train`、`--prompt`、`--corpus`、`--epochs`、`--batch_size`、`--lr`、`--save_dir`、`--max_len`

- 训练
  
  ```bash
  # 在 module/module_02 目录下
  python src/train_decoder_only.py \
    --train \
    --corpus src/mini_translation_pairs.txt \
    --save_dir ./outputs/ckpt_decoder_only \
    --epochs 3 \
    --batch_size 32 \
    --lr 3e-4 \
    --max_len 64
  ```

- 文本生成（需先完成训练，已存在权重 `outputs/ckpt_decoder_only/mini_decoder_only_model.pth` 时可直接生成）
  
  ```bash
  python src/train_decoder_only.py \
    --prompt "不可否认， 阅读有助于什么？" \
    --corpus src/mini_translation_pairs.txt \
    --save_dir ./outputs/ckpt_decoder_only
  ```

- 产物
  - 模型权重: `outputs/ckpt_decoder_only/mini_decoder_only_model.pth`
  - 词表随 checkpoint 一并保存

## Encoder-Decoder Transformer（翻译）
- 脚本: `src/train_transformer.py`
- 主要参数: `--train`、`--translate`、`--corpus`、`--epochs`、`--batch_size`、`--lr`、`--save_dir`

- 训练
  
  ```bash
  # 在 module/module_02 目录下
  python src/train_transformer.py \
    --train \
    --corpus src/mini_translation_pairs.txt \
    --save_dir ./outputs/ckpt \
    --epochs 3 \
    --batch_size 32 \
    --lr 3e-4
  ```

- 推断（中→英翻译）
  
  ```bash
  python src/train_transformer.py \
    --translate "今天天气不错" \
    --save_dir ./outputs/ckpt
  ```

- 产物
  - 模型权重: `outputs/ckpt/mini_transformer_ckpt.pt`
  - 词表与长度配置: `outputs/ckpt/meta.json`

## 备注
- 路径均以 `module/module_02` 为当前目录；若从其他目录执行，请相应调整相对路径。
- 可根据硬件与数据规模调整 `--epochs`、`--batch_size`、`--lr`、`--max_len` 等超参。
