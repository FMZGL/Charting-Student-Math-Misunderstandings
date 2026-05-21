import pandas as pd
import os

# 更新为原始数据的新路径
train_path = '../data/original/train.csv'
my_train_path = '../data/my_train.csv'

print(f"正在扫描 {train_path} 和 {my_train_path}...")

# 读取原始完整数据
df_full = pd.read_csv(train_path)
df_full_filled = df_full.copy()
df_full_filled['Misconception'] = df_full_filled['Misconception'].fillna('NA')
full_labels = set(df_full_filled['Category'] + ':' + df_full_filled['Misconception'])

# 读取切分后的 my_train 数据
df_my = pd.read_csv(my_train_path)
df_my['Misconception'] = df_my['Misconception'].fillna('NA')
my_labels = set(df_my['Category'] + ':' + df_my['Misconception'])

# 找出缺失的标签
missing = full_labels - my_labels
print('在 my_train.csv 中缺失的类别是:', missing)

if missing:
    for label in missing:
        # 在原始数据中找到第一条带有这个缺失标签的行
        missing_row_idx = df_full_filled[(df_full_filled['Category'] + ':' + df_full_filled['Misconception']) == label].index[0]
        # 提取出那一行最原始的数据（没有填充 NA 的）
        orig_row = df_full.iloc[[missing_row_idx]]
        # 将这一行追加到 my_train.csv 文件的末尾
        orig_row.to_csv(my_train_path, mode='a', header=False, index=False)
        print(f'✅ 已成功将缺失类别 [{label}] 的一条样本追加到 {my_train_path} 末尾！')
    print('大功告成！现在你的 my_train.csv 包含完整的 65 种标签了。')
else:
    print('没有缺失类别，my_train.csv 已经包含所有 65 种标签。')
