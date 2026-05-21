from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from peft import PeftModel
from scipy.special import softmax
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Dedicated single RTX 5090 configuration

lora_path = "models/gemma2-9b-it-cv93"
if not os.path.exists(lora_path):
    lora_path = "saves/Gemma-2-9B/lora_seqcls/train_seqcls/final"  # Fallback to local output

# Auto model path detection
BASE_MODEL_PATH = "models/gemma2-9b-it-bf16"
if not os.path.exists(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "model/gemma2-9b-it-bf16"
if not os.path.exists(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "/root/autodl-tmp/model/gemma2-9b-it-bf16"

MAX_LEN = 256


# helpers
def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


le = LabelEncoder()

train = pd.read_csv('data/my_train.csv')

train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

# Prepare test data
test = pd.read_csv('data/my_test.csv', encoding_errors='replace')
test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)

# load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Loading base model in {compute_dtype} configuration on single GPU 0...")

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=n_classes,
    torch_dtype=compute_dtype,
    device_map="cuda:0",  # Map strictly to the single GPU 0
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# Tokenize dataset
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True, remove_columns=['text'])

# Create data collator for efficient batching with padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    max_length=MAX_LEN,
    return_tensors="pt")

dataloader = DataLoader(
    ds_test,
    batch_size=32,  # Upgraded from 8 to 32 to maximize RTX 5090 concurrent inference throughput
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,
    num_workers=2
)

# Fast inference loop
all_logits = []
device = next(model.parameters()).device

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        logits = outputs.logits

        # Convert bfloat16 to float32 then move to CPU and store
        all_logits.append(logits.float().cpu().numpy())

# Concatenate all logits
predictions = np.concatenate(all_logits, axis=0)

# Convert to probs
probs = softmax(predictions, axis=1)

# Get top predictions (all 65 classes ranked)
top_indices = np.argsort(-probs, axis=1)

# Decode to class names
flat_indices = top_indices.flatten()
decoded_labels = le.inverse_transform(flat_indices)
top_labels = decoded_labels.reshape(top_indices.shape)

# --- EVALUATION: Compute MAP@3 ---
# Load true labels
true_labels_df = pd.read_csv('data/my_test_labels.csv')
true_labels_df['Misconception'] = true_labels_df['Misconception'].fillna('NA')
true_labels_df['true_target'] = true_labels_df['Category'] + ":" + true_labels_df['Misconception']

# Merge to ensure alignment
eval_df = pd.DataFrame({
    "row_id": test.row_id.values,
    "pred_1": top_labels[:, 0],
    "pred_2": top_labels[:, 1],
    "pred_3": top_labels[:, 2]
})
eval_df = eval_df.merge(true_labels_df[['row_id', 'true_target']], on='row_id', how='left')

# Calculate scores
scores = []
hits_at_1 = 0
hits_at_2 = 0
hits_at_3 = 0

for _, row in eval_df.iterrows():
    target = row['true_target']
    if target == row['pred_1']:
        scores.append(1.0)
        hits_at_1 += 1
    elif target == row['pred_2']:
        scores.append(1.0 / 2.0)
        hits_at_2 += 1
    elif target == row['pred_3']:
        scores.append(1.0 / 3.0)
        hits_at_3 += 1
    else:
        scores.append(0.0)

map3 = np.mean(scores)
top1_acc = hits_at_1 / len(eval_df)
top2_acc = (hits_at_1 + hits_at_2) / len(eval_df)
top3_acc = (hits_at_1 + hits_at_2 + hits_at_3) / len(eval_df)

print("\n" + "=" * 40)
print(f"✅ Evaluation Complete!")
print(f"Total Test Samples: {len(eval_df)}")
print(f"MAP@3 Score:        {map3:.4f}")
print(f"Top-1 Accuracy:     {top1_acc:.2%}")
print(f"Top-2 Accuracy:     {top2_acc:.2%}")
print(f"Top-3 Accuracy:     {top3_acc:.2%}")
print("=" * 40 + "\n")

# --- EXPORT PREDICTIONS ---
output_df = eval_df[['row_id', 'pred_1', 'pred_2', 'pred_3']].copy()
output_df.to_csv('output/output.csv', index=False)
print("📝 测试数据的Top-3预测结果已保存至: output/output.csv\n")

# --- VISUALIZATION ---
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    metrics = ['Top-1 Acc', 'Top-2 Acc', 'Top-3 Acc', 'MAP@3']
    values = [top1_acc, top2_acc, top3_acc, map3]

    bars = plt.bar(metrics, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.ylim(0, 1.05)
    plt.title('Model Evaluation Metrics (Test Set)', fontsize=14)
    plt.ylabel('Score / Accuracy', fontsize=12)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('output/evaluation_results.png', dpi=300)
    print("📊 可视化图表已保存至: output/evaluation_results.png")
except ImportError:
    print("⚠️ 未安装 matplotlib，跳过可视化图表生成。你可以通过 pip install matplotlib 安装。")
except Exception as e:
    print(f"⚠️ 可视化图表生成失败: {e}")