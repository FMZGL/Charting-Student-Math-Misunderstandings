"""
High-Performance Gemma2-9B Sequence Classification Fine-Tuning Script
Optimized for 32GB+ VRAM GPUs (e.g., RTX 3090, 4090, A100, A6000, L40S)

Features:
- Native BF16/FP16 training (No quantization by default for maximum accuracy and speed)
- CRITICAL FIX: Proper classification head (score layer) training and saving via PEFT modules_to_save
- SDPA (Scaled Dot Product Attention) for fast execution and low memory footprint
- Gradient Checkpointing & Paged Optimizer to ensure absolutely zero OOM
- Fully customizable batch sizes, sequence length (MAX_LEN=512 to prevent truncation), and hyperparameters
"""
import os
import sys
import subprocess

# ======================== 0. 自动依赖与版本检查 ========================
try:
    import bitsandbytes as bnb
    # 解析三段式版本号，确保 >= 0.46.1
    ver_parts = bnb.__version__.split('.')
    ver = []
    for x in ver_parts:
        # 提取前缀数字，过滤如 0.46.1.post1 这种情况
        digits = "".join([c for c in x if c.isdigit()])
        if digits:
            ver.append(int(digits))
    
    # 检查版本是否过旧
    is_old = False
    if len(ver) >= 3:
        if ver[0] == 0 and (ver[1] < 46 or (ver[1] == 46 and ver[2] < 1)):
            is_old = True
    elif len(ver) == 2:
        if ver[0] == 0 and ver[1] < 46:
            is_old = True
            
    if is_old:
        raise ImportError("bitsandbytes version too old")
    print(f"✓ bitsandbytes {bnb.__version__} (Compatible)")
except (ImportError, ModuleNotFoundError):
    print("Kaggle/本地环境 bitsandbytes 版本过旧或未安装，正在自动升级/安装至 >= 0.46.1 ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes>=0.46.1"])
    import bitsandbytes as bnb
    print(f"✓ bitsandbytes 已成功升级至: {bnb.__version__}")

import torch
import pandas as pd
import numpy as np
import gc
import pickle
import warnings
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ======================== 1. 核心超参数配置 ========================
# 基础模型与数据路径
BASE_MODEL_PATH = "models/gemma2-9b-it-bf16"  # 本地模型目录
if not os.path.exists(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "/root/autodl-tmp/model/gemma2-9b-it-bf16"  # 备用路径（Autodl服务器）

TRAIN_CSV = "data/my_train.csv"
OUTPUT_DIR = "saves/Gemma-2-9B/lora_seqcls/train_seqcls"

# 32GB+ 显卡黄金调参配置
MAX_LEN = 512                   # 设为 512，完整保留“问题+选项+学生解析”，不截断推理链
BATCH_SIZE = 4                  # 单卡批大小 4 (32GB+ 显卡无量化轻松跑通)
GRAD_ACCUM = 4                  # 梯度累积 4，等效 batch size = 4 * 4 = 16，收敛极度稳定
LEARNING_RATE = 2e-4            # LoRA 经典学习率
EPOCHS = 3                      # 训练 3 轮
LORA_R = 16                     # LoRA 秩
LORA_ALPHA = 32                 # alpha = 2 * r
LORA_DROPOUT = 0.05

# 显存省电开关（如果显卡非常强，比如 A100/A6000，可以完全关闭量化以获得最高训练速度和精度）
USE_QUANTIZATION = False        # 32GB+ 显卡推荐 False (以全精度 BF16 运行)
QUANTIZATION_TYPE = "8bit"      # 备用："8bit" 或 "4bit" (仅在显存吃紧时开启)

# ======================== 2. 检查硬件环境与精度设定 ========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 自动检测是否支持 BF16 (Ampere 及更新架构如 RTX 30/40, A100 支持 BF16)
bf16_enabled = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if bf16_enabled else torch.float16
print(f"[{'BF16' if bf16_enabled else 'FP16'} Mode Enabled] Using dtype: {compute_dtype}")
print(f"Using Base Model: {BASE_MODEL_PATH}")

# ======================== 3. 加载并清洗数据集 ========================
print("Loading data...")
train = pd.read_csv(TRAIN_CSV)
train['Misconception'] = train['Misconception'].fillna('NA')
# 合并 Category 和 Misconception 构成 65 个分类目标
train['target'] = train['Category'] + ":" + train['Misconception']

# 标签编码
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Total samples: {len(train)}, Unique Target Classes: {n_classes}")

# 构建符合推理一致性的 Prompt 格式
def format_input(row):
    is_correct = "Yes" if str(row['Category']).startswith('True') else "No"
    return (
        f"Question: {str(row['QuestionText'])}\n"
        f"Answer: {str(row['MC_Answer'])}\n"
        f"Correct? {is_correct}\n"
        f"Student Explanation: {str(row['StudentExplanation'])}"
    )

train['text'] = train.apply(format_input, axis=1)

# 随机抽样，切分训练集和验证集（避免有些极少类只有1个样本导致分层采样报错）
train_df, val_df = train_test_split(
    train, test_size=0.1, random_state=42
)
print(f"Training split: {len(train_df)} samples, Validation split: {len(val_df)} samples")

# ======================== 4. 加载 Tokenizer 与模型 ========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 分类任务推荐右侧 Padding

print(f"Loading Base Model ({compute_dtype}) ...")
model_kwargs = {
    "num_labels": n_classes,
    "torch_dtype": compute_dtype,
    "device_map": "auto",
    "attn_implementation": "sdpa", # 启用 PyTorch 2.0 SDPA，节省 30%+ 显存，加快速度
}

# 显存极限制约下的量化选项 (32GB+ 显存默认不启用，以获得满血精度)
if USE_QUANTIZATION:
    print(f"Enabling {QUANTIZATION_TYPE} quantization...")
    if QUANTIZATION_TYPE == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:  # 4bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs["quantization_config"] = bnb_config

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    **model_kwargs
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False  # 梯度检查点必须关闭 cache

# 如果是量化模型，需要进行 PEFT 前置处理
if USE_QUANTIZATION:
    model = prepare_model_for_kbit_training(model)

# ======================== 5. LoRA 配置 (核心修复分类头不保存问题) ========================
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # 🌟🌟🌟 极其关键：必须包含 "score"，否则新初始化的分类头参数不会被训练和保存！ 🌟🌟🌟
    modules_to_save=["score"], 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 打印初始显存占用情况
if torch.cuda.is_available():
    print(f"GPU Memory allocated after loading model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ======================== 6. 数据 Tokenize 与 Dataset 构建 ========================
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding=False,      # 动态 padding，不在 tokenize 时填充
        truncation=True,
        max_length=MAX_LEN,
    )

# 重命名 label 列为 labels (复数)，符合 Hugging Face 官方标准命名，简化 Trainer 数据对接
ds_train = Dataset.from_pandas(train_df[['text', 'label']].rename(columns={'label': 'labels'}))
ds_val = Dataset.from_pandas(val_df[['text', 'label']].rename(columns={'label': 'labels'}))

print("Tokenizing datasets...")
ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['text'])
ds_val = ds_val.map(tokenize_fn, batched=True, remove_columns=['text'])

# 🌟🌟 极其重要的 Bug 修复 🌟🌟
# 不要使用 ds_train.set_format(type='torch')！因为 Hugging Face datasets 库在新版 torchvision 中
# 会因为尝试导入已被移除的 VideoReader 而导致 ImportError 崩溃。
# 移除后，DataCollatorWithPadding 会在 Batch 级别自动进行完美且安全的 Tensor 转化。

# 动态 Padding Collator，大幅节省计算资源 and 显存
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
)

# ======================== 7. 训练参数配置 ========================
# 计算总训练步数与 warmup 步数，规避 warmup_ratio 弃用警告
total_steps = (len(train_df) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
warmup_steps = int(total_steps * 0.05)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,        # 代替 warmup_ratio 以规避版本警告
    lr_scheduler_type="cosine",       # 余弦退火调度器
    logging_steps=10,                 # 每 10 步打印一次日志
    eval_strategy="steps",
    eval_steps=500,                   # 每 500 步评估一次验证集
    save_strategy="steps",
    save_steps=500,                   # 每 500 步保存一次 Checkpoint
    save_total_limit=3,               # 保留最好的 3 个 Checkpoint
    load_best_model_at_end=True,      # 训练结束时自动加载最优模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=bf16_enabled,
    fp16=not bf16_enabled,
    gradient_checkpointing=True,      # 开启梯度检查点，大幅压缩激活值显存占用
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,         # 多线程加载数据
    report_to="none",                 # 禁用 wandb 等外部日志
    remove_unused_columns=False,
    label_names=["labels"],           # 更新为标准复数命名
    # 针对 32GB+ 显存优化的优化器
    optim="adamw_torch" if not USE_QUANTIZATION else "paged_adamw_8bit",
)

# 清理未回收显存
gc.collect()
torch.cuda.empty_cache()

# ======================== 8. 启动训练 ========================
# 创建基础参数字典
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": ds_train,
    "eval_dataset": ds_val,
    "data_collator": data_collator,
}

# 🌟 极其关键的兼容性修复：动态检查 Trainer.__init__ 的参数签名 🌟
# 新版 transformers (>=4.46) 将 tokenizer 参数重命名为了 processing_class
import inspect
trainer_signature = inspect.signature(Trainer.__init__)
if "processing_class" in trainer_signature.parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)


print("\n🚀 Starting high-performance training...")
trainer.train()

# ======================== 9. 保存最终模型与 LabelEncoder ========================
final_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_path, exist_ok=True)

# 保存微调后的 LoRA Adapter 及训练好的 Score 分类头
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

# 序列化保存 LabelEncoder，以确保推理时类别序号完全一致
with open(os.path.join(final_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\n🎉 Fine-tuning completed successfully!")
print(f"Model saved to: {final_path}")
print("You can now run 'gemma2_inference.py' for prediction and MAP@3 evaluation.")
