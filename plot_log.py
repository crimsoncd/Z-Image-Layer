import argparse
import os
import re
import matplotlib.pyplot as plt


def parse_log_file(file_path: str):
    """从纯文本日志文件中读取并解析包含 step 和各种损失/参数信息的行。

    :param file_path: 日志文件路径
    :return: (steps, metrics_dict)
             steps: step 列表 (int)
             metrics_dict: 包含各个指标数值列表的字典 { metric_name: [val1, val2, ...] }
    """
    steps = []
    metrics = {}

    # 正则表达式说明：
    # 寻找包含 step=X/Y 或 step=X 的行，并捕获该行中所有 key=value 的键值对
    step_pattern = re.compile(r"\bstep=(\d+)(?:/\d+)?")
    kv_pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([0-9.]+)")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            step_match = step_pattern.search(line)
            if not step_match:
                continue

            # 1. 提取当前行的 step 数值
            step_val = int(step_match.group(1))

            # 2. 提取当前行所有的 key=value 键值对
            kvs = kv_pattern.findall(line)
            if not kvs:
                continue

            current_row_metrics = {}
            for k, v in kvs:
                if k == "step":
                    continue
                try:
                    current_row_metrics[k] = float(v)
                except ValueError:
                    continue

            # 确认该行包含了有效指标数据后再记录
            if current_row_metrics:
                steps.append(step_val)
                for k, v in current_row_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)

    return steps, metrics


def plot_metrics(
    steps: list,
    metrics: dict,
    output_path: str = "training_metrics.png",
    log_name: str = "",
):
    """将提取到的各项指标绘制在多张子图中，合并保存为大图。"""
    num_metrics = len(metrics)
    if num_metrics == 0:
        print("[提示] 未在日志中解析到任何有效的参数指标！")
        return

    # 动态计算子图网格布局 (列数固定为 2 或 3)
    cols = 3 if num_metrics >= 5 else (2 if num_metrics >= 2 else 1)
    rows = (num_metrics + cols - 1) // cols

    # 创建大图画板
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True
    )

    # 统一将 axes 处理为一维数组，方便遍历
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 设置主题样式
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 逐个绘制各指标折线图
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]

        # 如果部分指标的数据点数量与 steps 不对齐（比如中间缺失），取最小长度截断
        min_len = min(len(steps), len(values))
        x_data = steps[:min_len]
        y_data = values[:min_len]

        ax.plot(
            x_data,
            y_data,
            linewidth=1.5,
            label=metric_name,
            color=plt.cm.tab10(idx % 10),
        )
        ax.set_title(f"{metric_name.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

        # 对学习率 lr 等较小数量级的参数使用科学计数法或适应格式
        if metric_name.lower() in ["lr", "learning_rate"]:
            ax.ticklabel_format(style="sci", axis="y")

    # 隐藏多余的空白子图 (如果网格数大于指标数)
    for idx in range(num_metrics, len(axes)):
        fig.delaxes(axes[idx])

    # 整体大图标题与布局优化
    title_str = f"Training Metrics Visualization ({log_name})" if log_name else "Training Metrics Visualization"
    fig.suptitle(title_str, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[成功] 图表已绘制并保存至: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="从训练日志中提取 step 及各项 loss/参数并绘制合并大图"
    )
    parser.add_argument("log_path", type=str, help="日志文件路径 (.log 或 .txt)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="training_metrics.png",
        help="输出图片路径 (默认: training_metrics.png)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"[错误] 文件不存在: {args.log_path}")
        return

    print(f"正在解析日志文件: {args.log_path} ...")
    steps, metrics = parse_log_file(args.log_path)

    print(f"共提取到 {len(steps)} 条符合条件的日志记录。")
    print(f"包含指标: {', '.join(metrics.keys())}")

    log_filename = os.path.basename(args.log_path)
    plot_metrics(
        steps, metrics, output_path=args.output, log_name=log_filename
    )


if __name__ == "__main__":
    main()