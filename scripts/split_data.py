import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("加载原始训练集 data/train.csv...")
df = pd.read_csv('data/train.csv')

# 我们采用随机切分，切分出 10% 作为新的测试集（可以根据需要调整 test_size）
# 因为某些少数误区类别只有 1 个样本，所以我们不使用严格的 stratify，采用纯随机切分
print("切分数据集 (90% 训练, 10% 测试)...")
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# 保存新的训练集
train_df.to_csv('data/my_train.csv', index=False)
print(f"新的训练集已保存至 data/my_train.csv (共 {len(train_df)} 条)")

# 为了完全模拟官方 test.csv 的格式，我们需要从测试集中删掉 Category 和 Misconception 两列
# 但作为研究项目，你需要拿这些真实标签去评估模型的分数！
# 所以我们会生成两个文件：一个模拟官方格式的无标签测试集，一个仅包含标签的文件用于后续评估计算。

test_features = test_df.drop(columns=['Category', 'Misconception'])
test_features.to_csv('data/my_test.csv', index=False)
print(f"新的格式化测试集已保存至 data/my_test.csv (共 {len(test_features)} 条)")

test_labels = test_df[['row_id', 'Category', 'Misconception']]
test_labels.to_csv('data/my_test_labels.csv', index=False)
print(f"测试集的真实标签已保存至 data/my_test_labels.csv (用于后续模型评估计算)")

print("\n切分完成！接下来你可以修改 gemma2_inference.py 中的路径，让它读取 my_test.csv 进行推理。")
