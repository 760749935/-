import csv
import ollama
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# 配置参数
MODEL_NAME = "gemma3:12b-it-qat"
FILE_PATH = r"D:\Python\pythonProject3\情感-语义检测分析模型\First Part\social_media_posts.csv"
DELAY_BETWEEN_REQUESTS = 1
OUTPUT_FILE = "task1_results.csv"

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']  # 尝试这些中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_dataset(file_path):
    """加载数据集"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            data = []
            for row in reader:
                if 'post_text' in row and 'label' in row:
                    text = row['post_text'].strip()
                    label = row['label'].strip().lower()
                    if text and label in ['fake', 'real']:
                        data.append({
                            'text': text,
                            'true_label': label
                        })
            return data
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return []


def classify_news(text):
    """判别新闻真假"""
    prompt = f"""
    你是一个新闻真假判别助手。请根据下列新闻内容判断其是真新闻(real)还是假新闻(fake)。
    请只输出单个单词"real"或"fake"，不要包含任何其他文字或解释。

    新闻内容: {text}
    判断结果:
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        result = response['message']['content'].strip().lower()

        # 确保只返回"real"或"fake"
        if "fake" in result:
            return "fake"
        elif "real" in result:
            return "real"
        else:
            print(f"无法解析结果: '{result}'")
            return "unknown"
    except Exception as e:
        print(f"分类错误: {str(e)}")
        return "error"


def calculate_metrics(true_labels, pred_labels):
    """计算评估指标"""
    valid_indices = [i for i, p in enumerate(pred_labels) if p in ['fake', 'real']]
    true_valid = [true_labels[i] for i in valid_indices]
    pred_valid = [pred_labels[i] for i in valid_indices]

    if not true_valid:
        return 0.0, [], []

    accuracy = accuracy_score(true_valid, pred_valid)
    cm = confusion_matrix(true_valid, pred_valid, labels=['fake', 'real'])
    return accuracy, cm, valid_indices


def plot_confusion_matrix(cm):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('真假新闻分类混淆矩阵')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['假新闻', '真新闻'])
    plt.yticks(tick_marks, ['假新闻', '真新闻'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('task1_confusion_matrix.png')
    plt.close()  # 关闭图形，避免内存泄漏
    print("混淆矩阵已保存为 task1_confusion_matrix.png")


def main():
    print(f"加载数据集: {FILE_PATH}")
    # 加载数据集
    dataset = load_dataset(FILE_PATH)
    if not dataset:
        print("未加载到有效数据，程序退出")
        return

    print(f"开始处理 {len(dataset)} 条新闻...")

    # 存储结果
    results = []
    true_labels = []
    pred_labels = []

    # 处理每条新闻
    for i, item in enumerate(dataset, 1):
        print(f"\n处理第 {i}/{len(dataset)} 条新闻...")
        print(f"内容: {item['text'][:50]}...")
        print(f"真实标签: {item['true_label']}")

        # 分类新闻
        pred_label = classify_news(item['text'])
        pred_labels.append(pred_label)
        print(f"预测标签: {pred_label}")

        # 存储结果
        results.append({
            "text": item['text'],
            "true_label": item['true_label'],
            "pred_label": pred_label
        })

        true_labels.append(item['true_label'])
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # 计算指标
    accuracy, cm, valid_indices = calculate_metrics(true_labels, pred_labels)

    print("\n" + "=" * 50)
    print("任务1结果统计:")
    print(f"准确率: {accuracy:.2%}")
    print("\n混淆矩阵:")
    print("        Predicted Fake  Predicted Real")
    print(f"Actual Fake  {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Real  {cm[1][0]:<14} {cm[1][1]}")
    print("=" * 50)

    # 绘制混淆矩阵
    plot_confusion_matrix(cm)

    # 保存结果
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label", "pred_label", "text"])
            for i, res in enumerate(results):
                if i in valid_indices:
                    writer.writerow([
                        res["true_label"],
                        res["pred_label"],
                        res["text"]
                    ])
        print(f"结果已保存到 {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")

    print("任务1完成!")


if __name__ == "__main__":
    main()