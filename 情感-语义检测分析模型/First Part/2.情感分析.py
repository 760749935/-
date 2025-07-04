import csv
import ollama
import time
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# 配置参数
MODEL_NAME = "gemma3:12b-it-qat"
FILE_PATH = r"D:\Python\pythonProject3\情感-语义检测分析模型\First Part\social_media_posts.csv"
OUTPUT_FILE = "task2_results.csv"
DELAY_BETWEEN_REQUESTS = 1


# 设置中文字体
def set_chinese_font():
    try:
        # 查找系统中支持中文的字体
        font_path = None
        for font in font_manager.fontManager.ttflist:
            if 'SimHei' in font.name:
                font_path = font.fname
                break
            elif 'Microsoft YaHei' in font.name:
                font_path = font.fname
                break
            elif 'Arial Unicode MS' in font.name:
                font_path = font.fname
                break

        if font_path:
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
        else:
            # 如果找不到上述字体，尝试使用默认的宋体
            plt.rcParams['font.family'] = 'SimSun'
    except:
        # 如果所有尝试都失败，至少设置一个支持中文的字体族
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']


# 在程序开始时设置中文字体
set_chinese_font()


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
                if 'post_text' in row:
                    text = row['post_text'].strip()
                    if text:
                        data.append({
                            'text': text,
                            'label': row.get('label', '')
                        })
            return data
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return []


def analyze_sentiment(text):
    """分析文档语义情感"""
    prompt = f"""
    ## 任务说明
    你是一位情感分析专家，请对以下社交媒体内容进行情感分析：

    ## 内容
    "{text}"

    ## 分析维度
    1. 情感分类: 正面/负面/中性
    2. 情感强度: 弱/中等/强
    3. 情感关键词: 列举1-3个关键词
    4. 情感依据: 简要解释

    ## 输出格式
    - 情感分类: [结果]
    - 情感强度: [结果]
    - 情感关键词: [结果]
    - 情感依据: [结果]
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        result = response['message']['content'].strip()

        # 解析结果
        sentiment_data = {
            "sentiment": "未知",
            "intensity": "未知",
            "keywords": [],
            "evidence": ""
        }

        lines = result.split('\n')
        for line in lines:
            if "情感分类:" in line:
                sentiment_data["sentiment"] = line.split(":")[1].strip()
            elif "情感强度:" in line:
                sentiment_data["intensity"] = line.split(":")[1].strip()
            elif "情感关键词:" in line:
                keywords = line.split(":")[1].strip()
                sentiment_data["keywords"] = [kw.strip() for kw in keywords.split(",") if kw.strip()]
            elif "情感依据:" in line:
                sentiment_data["evidence"] = line.split(":")[1].strip()

        return sentiment_data
    except Exception as e:
        print(f"情感分析错误: {str(e)}")
        return {
            "sentiment": "错误",
            "intensity": "错误",
            "keywords": [],
            "evidence": str(e)
        }


def visualize_sentiment(sentiments):
    """可视化情感分布"""
    sentiment_counts = {}
    for sent in sentiments:
        s = sent["sentiment"]
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

    if sentiment_counts:
        labels = list(sentiment_counts.keys())
        sizes = list(sentiment_counts.values())

        # 确保有足够的颜色
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
        colors = colors[:len(labels)]

        plt.figure(figsize=(10, 7))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')
        plt.title('社交媒体内容情感分布')

        # 尝试保存图像，如果失败则尝试不同的方法
        try:
            plt.savefig('sentiment_distribution.png', bbox_inches='tight', dpi=300)
        except:
            # 如果仍然失败，尝试使用不同的后端
            plt.switch_backend('agg')
            plt.savefig('sentiment_distribution.png', bbox_inches='tight', dpi=300)

        print("情感分布图已保存为 sentiment_distribution.png")
    else:
        print("没有有效的情感数据用于可视化")


def main():
    print(f"加载数据集: {FILE_PATH}")
    # 加载数据集
    dataset = load_dataset(FILE_PATH)
    if not dataset:
        print("未加载到有效数据，程序退出")
        return

    print(f"开始分析 {len(dataset)} 条内容的情感...")

    # 存储结果
    results = []
    sentiments = []

    # 处理每条内容
    for i, item in enumerate(dataset, 1):
        print(f"\n分析第 {i}/{len(dataset)} 条内容...")
        print(f"内容: {item['text'][:50]}...")

        # 情感分析
        sentiment_data = analyze_sentiment(item['text'])
        sentiments.append(sentiment_data)
        print(f"情感分析结果: {sentiment_data['sentiment']} ({sentiment_data['intensity']})")
        print(f"关键词: {', '.join(sentiment_data['keywords'])}")

        # 存储结果
        results.append({
            "text": item['text'],
            "sentiment": sentiment_data['sentiment'],
            "intensity": sentiment_data['intensity'],
            "keywords": ", ".join(sentiment_data['keywords']),
            "evidence": sentiment_data['evidence'],
            "label": item.get('label', '')
        })

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # 可视化结果
    visualize_sentiment(sentiments)

    # 保存结果
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sentiment", "intensity", "keywords", "evidence", "label", "text"])
            for res in results:
                writer.writerow([
                    res["sentiment"],
                    res["intensity"],
                    res["keywords"],
                    res["evidence"],
                    res["label"],
                    res["text"]
                ])
        print(f"结果已保存到 {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")

    print("任务2完成!")


if __name__ == "__main__":
    main()