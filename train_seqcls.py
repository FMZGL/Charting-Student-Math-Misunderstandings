"""
High-Performance Gemma2-9B Sequence Classification Fine-Tuning Script
Optimized for 32GB+ VRAM GPUs (e.g., RTX 5090, A100, A6000, L40S)

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

# ======================== 0. Automated Dependency & Version Check ========================
try:
    import bitsandbytes as bnb
    # Parse 3-part version number, ensuring >= 0.46.1
    ver_parts = bnb.__version__.split('.')
    ver = []
    for x in ver_parts:
        # Extract numeric prefix to filter out post-releases like 0.46.1.post1
        digits = "".join([c for c in x if c.isdigit()])
        if digits:
            ver.append(int(digits))
    
    # Check if version is outdated
    is_old = False
    if len(ver) >= 3:
        if ver[0] == 0 and (ver[1] < 46 or (ver[1] == 46 and ver[2] < 1)):
            is_old = True
    elif len(ver) == 2:
        if ver[0] == 0 and ver[1] < 46:
            is_old = True
            
    if is_old:
        raise ImportError("bitsandbytes version too old")
    print(f"[OK] bitsandbytes {bnb.__version__} (Compatible)")
except (ImportError, ModuleNotFoundError):
    print("Kaggle/local bitsandbytes version is outdated or not installed, upgrading/installing to >= 0.46.1 ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes>=0.46.1"])
    import bitsandbytes as bnb
    print(f"[OK] bitsandbytes upgraded successfully to: {bnb.__version__}")

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

# ======================== 1. Core Hyperparameter Configuration ========================
# Base model and dataset paths
BASE_MODEL_PATH = "models/gemma2-9b-it-bf16"  # Local model directory
if not os.path.exists(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "/root/autodl-tmp/model/gemma2-9b-it-bf16"  # Backup path (Autodl server)

TRAIN_CSV = "data/my_train.csv"
OUTPUT_DIR = "saves/Gemma-2-9B/lora_seqcls/train_seqcls"

# Golden hyperparameter tuning configuration for 32GB+ VRAM GPUs
MAX_LEN = 512                   # Set to 512 to preserve complete Question + Option + Explanation, avoiding sequence truncation
BATCH_SIZE = 4                  # Single GPU batch size 4 (runs smoothly without quantization on 32GB+ VRAM)
GRAD_ACCUM = 4                  # Gradient accumulation 4, effective batch size = 4 * 4 = 16, highly stable convergence
LEARNING_RATE = 2e-4            # Classic LoRA learning rate
EPOCHS = 3                      # Train for 3 epochs
LORA_R = 16                     # LoRA Rank
LORA_ALPHA = 32                 # alpha = 2 * r
LORA_DROPOUT = 0.05

# Memory saving toggle (if using ultra-high-end GPUs like A100/A6000, keep quantization disabled for peak training speed and maximum accuracy)
USE_QUANTIZATION = False        # Recommended False on 32GB+ VRAM (runs in full native BF16 precision)
QUANTIZATION_TYPE = "8bit"      # Backup: "8bit" or "4bit" (only enable if VRAM is severely constrained)

# ======================== 2. Hardware Environment Verification & Precision Settings ========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Automatically detect BF16 support (supported by Ampere and newer architectures such as RTX 30/40/50, A100, etc.)
bf16_enabled = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if bf16_enabled else torch.float16
print(f"[{'BF16' if bf16_enabled else 'FP16'} Mode Enabled] Using dtype: {compute_dtype}")
print(f"Using Base Model: {BASE_MODEL_PATH}")

# ======================== 3. Load & Clean Dataset ========================
print("Loading data...")
train = pd.read_csv(TRAIN_CSV)
train['Misconception'] = train['Misconception'].fillna('NA')
# Combine Category and Misconception to construct 65 classification targets
train['target'] = train['Category'] + ":" + train['Misconception']

# Label Encoding
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Total samples: {len(train)}, Unique Target Classes: {n_classes}")

# Build Prompt format matching inference requirements
def format_input(row):
    is_correct = "Yes" if str(row['Category']).startswith('True') else "No"
    return (
        f"Question: {str(row['QuestionText'])}\n"
        f"Answer: {str(row['MC_Answer'])}\n"
        f"Correct? {is_correct}\n"
        f"Student Explanation: {str(row['StudentExplanation'])}"
    )

train['text'] = train.apply(format_input, axis=1)

# Split train and validation datasets (uses random split instead of stratified to avoid sklearn errors on extremely rare classes with only 1 sample)
train_df, val_df = train_test_split(
    train, test_size=0.1, random_state=42
)
print(f"Training split: {len(train_df)} samples, Validation split: {len(val_df)} samples")

# ======================== 4. Load Tokenizer & Model ========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Padding on the right is recommended for classification tasks

print(f"Loading Base Model ({compute_dtype}) ...")
model_kwargs = {
    "num_labels": n_classes,
    "torch_dtype": compute_dtype,
    "device_map": "auto",
    "attn_implementation": "sdpa", # Enable PyTorch 2.0 SDPA to save 30%+ VRAM and accelerate training
}

# Quantization configurations for VRAM-constrained setups (disabled by default on 32GB+ VRAM for full precision)
if USE_QUANTIZATION:
    print(f"Enabling {QUANTIZATION_TYPE} quantization...")
    if QUANTIZATION_TYPE == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:  # 4-bit
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
model.config.use_cache = False  # Cache must be disabled when using gradient checkpointing

# Pre-process model for PEFT if quantization is enabled
if USE_QUANTIZATION:
    model = prepare_model_for_kbit_training(model)

# ======================== 5. LoRA Configuration (Fixing Classification Head Persistence) ========================
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
    # *** CRITICAL: Must include "score" or the newly initialized classification head will not be trained and saved! ***
    modules_to_save=["score"], 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Print initial VRAM allocation
if torch.cuda.is_available():
    print(f"GPU Memory allocated after loading model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ======================== 6. Data Tokenization & Dataset Construction ========================
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding=False,      # Dynamic padding, do not pad during tokenization
        truncation=True,
        max_length=MAX_LEN,
    )

# Rename label column to labels (plural) to comply with HF official standards and simplify Trainer integration
ds_train = Dataset.from_pandas(train_df[['text', 'label']].rename(columns={'label': 'labels'}))
ds_val = Dataset.from_pandas(val_df[['text', 'label']].rename(columns={'label': 'labels'}))

print("Tokenizing datasets...")
ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['text'])
ds_val = ds_val.map(tokenize_fn, batched=True, remove_columns=['text'])

# *** CRITICAL BUG FIX ***
# DO NOT use ds_train.set_format(type='torch')! HF datasets package triggers an ImportError
# in newer torchvision versions by trying to import the removed VideoReader backend.
# After removing it, DataCollatorWithPadding natively handles clean and safe tensor conversion at batch level.

# Dynamic Padding Collator to significantly save compute and VRAM
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
)

# ======================== 7. Training Arguments Configuration ========================
# Calculate total training steps and warmup steps to avoid warmup_ratio deprecation warning
total_steps = (len(train_df) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
warmup_steps = int(total_steps * 0.05)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,        # Replaces warmup_ratio to avoid deprecation warnings
    lr_scheduler_type="cosine",       # Cosine annealing scheduler
    logging_steps=10,                 # Log metrics every 10 steps
    eval_strategy="steps",
    eval_steps=500,                   # Evaluate on validation dataset every 500 steps
    save_strategy="steps",
    save_steps=500,                   # Save checkpoint every 500 steps
    save_total_limit=3,               # Retain only the best 3 checkpoints
    load_best_model_at_end=True,      # Load the best model automatically at the end of training
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=bf16_enabled,
    fp16=not bf16_enabled,
    gradient_checkpointing=True,      # Enable gradient checkpointing to compress activation memory footprint
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,         # Multi-process data loading
    report_to="none",                 # Disable logging to external trackers like wandb
    remove_unused_columns=False,
    label_names=["labels"],           # Update to standard plural naming
    # Optimized optimizers for 32GB+ VRAM
    optim="adamw_torch" if not USE_QUANTIZATION else "paged_adamw_8bit",
)

# Garbage collection and clear VRAM cache
gc.collect()
torch.cuda.empty_cache()

# ======================== 8. Launch Training Pipeline ========================
# Create base trainer keyword arguments
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": ds_train,
    "eval_dataset": ds_val,
    "data_collator": data_collator,
}

# *** CRITICAL COMPATIBILITY FIX: Dynamically inspect Trainer.__init__ signature ***
# Transformers >=4.46 renamed the tokenizer parameter to processing_class
import inspect
trainer_signature = inspect.signature(Trainer.__init__)
if "processing_class" in trainer_signature.parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)


print("\nStarting high-performance training...")
trainer.train()

# ======================== 9. Save Final Model & LabelEncoder ========================
final_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_path, exist_ok=True)

# Save the fine-tuned LoRA Adapter and the trained Score classification head
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

# Serialize and save LabelEncoder to guarantee exact category index alignment during inference
with open(os.path.join(final_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\nFine-tuning completed successfully!")
print(f"Model saved to: {final_path}")
print("You can now run 'gemma2_inference.py' for prediction and MAP@3 evaluation.")
