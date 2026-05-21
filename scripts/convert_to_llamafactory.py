import pandas as pd
import json
import os

# 确保脚本即使在项目的根目录下运行，路径也是正确的
# 假设脚本在 scripts/ 目录下运行，我们要访问根目录下的 data/
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_csv_path = os.path.join(base_dir, 'data', 'my_train.csv')
output_json_path = os.path.join(base_dir, 'data', 'dataset_llamafactory.json')

print(f"正在读取 {input_csv_path}...")
try:
    df = pd.read_csv(input_csv_path)
except FileNotFoundError:
    print(f"❌ 找不到文件：{input_csv_path}")
    print("请确保你是在项目根目录下运行此脚本，或者文件路径正确。")
    exit(1)

# 确保误区为空的地方填充为 NA
df['Misconception'] = df['Misconception'].fillna('NA')

llama_factory_data = []

for index, row in df.iterrows():
    # 判断是否正确
    is_correct_str = "Yes" if row.get('is_correct', 0) else "No"
    
    # 构造 input 字段 (这是之前大模型看到的上下文)
    input_text = (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {is_correct_str}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )
    
    # 构造 output 字段 (这是大模型需要生成的标签)
    output_text = f"{row['Category']}:{row['Misconception']}"
    
    # 组装成 LLaMA-Factory 接受的单条数据
    data_point = {
        # instruction 可以统一，告诉大模型它的任务是什么
        "instruction": "You are an expert math teacher. Based on the following math question, the student's chosen answer, and their written explanation, diagnose the specific mathematical misconception they have. Output ONLY the exact category and misconception label.",
        "input": input_text,
        "output": output_text
    }
    llama_factory_data.append(data_point)

print(f"正在保存为 {output_json_path}...")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成！共处理 {len(llama_factory_data)} 条数据。")
print(f"✅ 文件已保存至: {output_json_path}")
print("\n下一步提示：")
print("1. 将生成的数据集复制到服务器上 LLaMA-Factory 的 data/ 目录。")
print("2. 在 LLaMA-Factory 的 data/dataset_info.json 中注册此数据集。")
