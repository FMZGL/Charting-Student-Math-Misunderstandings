import os
from modelscope.hub.snapshot_download import snapshot_download

# 定义魔搭社区上的模型 ID 和本地保存目录
# 在 ModelScope 上，这个模型由 LLM-Research 团队同步
model_id = "LLM-Research/gemma-2-9b-it"
local_dir = "../models/gemma2-9b-it-bf16"

print(f"🚀 开始通过阿里魔搭社区 (ModelScope) 下载模型...")
print(f"模型 ID: {model_id}")
print(f"保存路径: {local_dir}")
print("注意: 走纯正国内网络，无需挂代理！支持断点续传。\n")

try:
    # 执行下载
    path = snapshot_download(
        model_id=model_id,
        local_dir=local_dir,
    )
    print("\n✅ 下载完成!")
    print(f"模型文件已就绪: {path}")
    
except Exception as e:
    print("\n❌ 下载失败，错误详情:", e)