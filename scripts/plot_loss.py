import os
import json
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.9):
    """使用指数移动平均对曲线进行平滑处理"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def main():
    json_path = "trainer_state.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} 不存在，请确保在包含该文件的目录下运行脚本。")
        return

    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    # 提取关键信息
    best_step = state.get("best_global_step", 5500)
    log_history = state.get("log_history", [])

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    lrs = []
    lr_steps = []

    for log in log_history:
        step = log.get("step")
        if step is None:
            continue
        
        # 提取训练 loss
        if "loss" in log:
            train_steps.append(step)
            train_losses.append(log["loss"])
            
        # 提取验证 loss
        if "eval_loss" in log:
            eval_steps.append(step)
            eval_losses.append(log["eval_loss"])
            
        # 提取学习率
        if "learning_rate" in log:
            lr_steps.append(step)
            lrs.append(log["learning_rate"])

    print(f"Data loaded successfully:")
    print(f"  - Train log steps: {len(train_losses)}")
    print(f"  - Eval log steps: {len(eval_losses)}")
    print(f"  - Best step from JSON: {best_step}")

    # 设置高级视觉样式 (使用标准 Arial 字体)
    plt.rcParams["font.sans-serif"] = ["Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 尝试使用美化样式
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("ggplot")

    # 创建双子图 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # 颜色配置 (高级莫兰迪/深海色系)
    c_train_raw = "#a2d2ff"     # 淡蓝 (原始训练损耗)
    c_train_smooth = "#0077b6"  # 深海蓝 (平滑后训练损耗)
    c_eval = "#d62828"          # 宝石红 (验证集损耗)
    c_lr = "#f77f00"            # 亮橙 (学习率)
    c_best = "#2a9d8f"          # 翡翠绿 (最佳Step标注)

    # ==================== 左图：Loss 曲线 ====================
    if train_losses:
        ax1.plot(train_steps, train_losses, color=c_train_raw, alpha=0.3, label="Train Loss (Raw)")
        train_smoothed = smooth_curve(train_losses, factor=0.85)
        ax1.plot(train_steps, train_smoothed, color=c_train_smooth, linewidth=2, label="Train Loss (EMA)")

    if eval_losses:
        ax1.plot(eval_steps, eval_losses, color=c_eval, marker="o", markersize=5, 
                 linewidth=2, label="Validation Loss")
        
        # 寻找验证集最低点
        min_eval_idx = np.argmin(eval_losses)
        min_eval_step = eval_steps[min_eval_idx]
        min_eval_loss = eval_losses[min_eval_idx]
        print(f"  - Min Validation Loss Step: {min_eval_step} (Loss: {min_eval_loss:.4f})")
        
        # 在最低点做标记
        ax1.scatter(min_eval_step, min_eval_loss, color="black", s=80, zorder=5)
        ax1.annotate(f"Min Val Loss\nStep {min_eval_step}: {min_eval_loss:.4f}",
                    xy=(min_eval_step, min_eval_loss),
                    xytext=(min_eval_step + (max(train_steps)*0.05), min_eval_loss + 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                    fontsize=9, fontweight='bold', color=c_eval)

    # 5500 步作为用户确认的最佳 checkpoint 进行标注
    user_best_step = 5500
    if user_best_step in train_steps or user_best_step in eval_steps:
        ax1.axvline(x=user_best_step, color=c_best, linestyle="--", linewidth=1.5, 
                    label=f"User Best Checkpoint (Step {user_best_step})")
        if user_best_step in eval_steps:
            best_eval_loss = eval_losses[eval_steps.index(user_best_step)]
            ax1.scatter(user_best_step, best_eval_loss, color=c_best, s=100, zorder=6, edgecolors='black')
            ax1.annotate(f"Selected Best\nStep {user_best_step}: {best_eval_loss:.4f}",
                        xy=(user_best_step, best_eval_loss),
                        xytext=(user_best_step - (max(train_steps)*0.25), best_eval_loss - 0.25),
                        arrowprops=dict(facecolor=c_best, shrink=0.08, width=1, headwidth=6),
                        fontsize=9, fontweight='bold', color=c_best)

    ax1.set_title("Training and Validation Loss Curves", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Steps", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ==================== 右图：Learning Rate 曲线 ====================
    if lrs:
        ax2.plot(lr_steps, lrs, color=c_lr, linewidth=2, label="Learning Rate")
        if user_best_step in lr_steps:
            ax2.axvline(x=user_best_step, color=c_best, linestyle="--", linewidth=1.5, 
                        label=f"User Best Checkpoint (Step {user_best_step})")
        
        max_lr = max(lrs)
        max_lr_step = lr_steps[lrs.index(max_lr)]
        ax2.scatter(max_lr_step, max_lr, color="red", s=40, zorder=5)
        ax2.annotate(f"Peak LR: {max_lr:.2e}",
                    xy=(max_lr_step, max_lr),
                    xytext=(max_lr_step + (max(lr_steps)*0.05), max_lr * 0.9),
                    arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=4),
                    fontsize=9)

        ax2.set_title("Learning Rate Scheduler", fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel("Steps", fontsize=11)
        ax2.set_ylabel("Learning Rate", fontsize=11)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 全局调整与保存
    plt.tight_layout()
    output_filename = "training_loss_curves.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print(f"Success: Loss and LR curves visualized successfully!")
    print(f"Saved to: {os.path.abspath(output_filename)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
