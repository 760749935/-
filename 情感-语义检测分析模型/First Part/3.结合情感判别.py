# -*- coding: utf-8 -*-
import csv
import ollama
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
import platform

# ==================== 配置区域 ====================
MODEL_NAME = "gemma3:12b-it-qat"  # 使用的模型名称
FILE_PATH = r"D:\Python\pythonProject3\情感-语义检测分析模型\First Part\social_media_posts.csv"  # 数据集路径
OUTPUT_FILE = "task3_results.csv"  # 结果保存文件名
DELAY_BETWEEN_REQUESTS = 1  # 请求间隔时间(秒)


# ================================================

def init_environment():
    """初始化运行环境（字体设置等）"""
    # 设置中文字体
    plt.style.use('ggplot')

    try:
        if platform.system() == 'Windows':
            # Windows系统字体
            font_path = 'C:/Windows/Fonts/simhei.ttf'
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
            else:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        else:
            # Mac/Linux系统字体
            font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # Mac
            if not os.path.exists(font_path):
                font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # Linux
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()

        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("字体设置完成")
    except Exception as e:
        print(f"字体设置失败: {str(e)}")


def load_dataset(file_path):
    """加载并预处理数据集"""
    print(f"\n正在加载数据集: {file_path}")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

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
            print(f"成功加载 {len(data)} 条有效数据")
            return data
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return []


def analyze_sentiment(text):
    """执行情感分析"""
    prompt = f"""
       ## 任务说明
       你是一位情感分析专家，请对以下社交媒体内容进行情感分析：
       ## 分析维度
       1. 情感分类: 中性/正面/负面
       4. 情感依据: 简要解释
       5. 请注意，第一次提出的情感结果将会录入，为不影响判别你只能提出一次中性、正面、负面
    待分析文本：
    "{text[:500]}"  # 限制输入长度
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 2048}
        )
        result = response['message']['content'].strip().lower()

        # 结果标准化处理
        if any(word in result for word in ["中性", "positive"]):
            return "中性"
        elif any(word in result for word in ["负面", "negative"]):
            return "负面"
        else:
            return "正面"
    except Exception as e:
        print(f"情感分析出错: {str(e)}")
        return "分析失败"


def classify_news(text, sentiment):
    """结合情感进行新闻真实性分类"""
    prompt = f"""
    ## 事实核查任务 ##
    你是一名新闻分析专家，其中有十条真新闻，十条假新闻，请结合情感仔细分析新闻内容做出判别
    情感分析结果: {sentiment}
    待核查内容: "{text[:500]}"

    要求:
     1. 消极内容需验证事实是否发生
     2. 积极内容需检查是否夸张
     3. 中性内容看事实完整性
     只需回复'假新闻'或'真新闻'
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        result = response['message']['content'].strip().lower()

        if any(word in result for word in ["假", "fake", "false"]):
            return "fake"
        elif any(word in result for word in ["真", "real", "true"]):
            return "real"
        else:
            print(f"无法解析分类结果: '{result}'")
            return "unknown"
    except Exception as e:
        print(f"分类出错: {str(e)}")
        return "error"


def visualize_results(true_labels, pred_labels, sentiments):
    """结果可视化"""
    # 1. 混淆矩阵
    labels = ['fake', 'real']
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('新闻真实性分类混淆矩阵', pad=20)
    plt.colorbar()

    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['预测假新闻', '预测真新闻'])
    plt.yticks(tick_marks, ['真实假新闻', '真实真新闻'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

    # 2. 情感分布饼图
    sentiment_counts = {}
    for s in sentiments:
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values(),
            labels=sentiment_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66b3ff', '#ff9999', '#99ff99'])
    plt.title('内容情感分布')
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')

    print("\n可视化结果已保存:")
    print(f"- confusion_matrix.png (混淆矩阵)")
    print(f"- sentiment_distribution.png (情感分布)")


def save_results(results, output_file):
    """保存结果到CSV"""
    try:
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig解决Excel中文乱码
            writer = csv.writer(f)
            writer.writerow(['真实标签', '情感分析', '预测标签', '文本内容'])
            for res in results:
                writer.writerow([
                    res['true_label'],
                    res['sentiment'],
                    res['pred_label'],
                    res['text'][:500] + "..."  # 限制保存长度
                ])
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"结果保存失败: {str(e)}")


def main():
    # 初始化环境
    init_environment()

    # 加载数据
    dataset = load_dataset(FILE_PATH)
    if not dataset:
        return

    print("\n开始分析...")
    results = []
    true_labels = []
    pred_labels = []
    sentiments = []

    # 处理每条数据
    for idx, item in enumerate(dataset, 1):
        print(f"\n[{idx}/{len(dataset)}] 处理中...")
        print(f"内容: {item['text'][:50]}...")

        # 情感分析
        sentiment = analyze_sentiment(item['text'])
        sentiments.append(sentiment)
        print(f"情感: {sentiment}")

        # 真实性分类
        pred_label = classify_news(item['text'], sentiment)
        pred_labels.append(pred_label)
        print(f"预测: {pred_label} (真实: {item['true_label']})")

        # 保存结果
        results.append({
            'text': item['text'],
            'true_label': item['true_label'],
            'sentiment': sentiment,
            'pred_label': pred_label
        })

        true_labels.append(item['true_label'])
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    print("\n" + "=" * 50)
    print(f"最终准确率: {accuracy:.2%}")
    print("=" * 50)

    # 可视化与保存
    visualize_results(true_labels, pred_labels, sentiments)
    save_results(results, OUTPUT_FILE)

    print("\n任务完成!")


if __name__ == "__main__":
    main()