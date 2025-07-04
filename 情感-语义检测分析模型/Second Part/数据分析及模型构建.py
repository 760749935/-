import csv
import re
import nltk
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import ollama

# 设置NLTK数据存储路径
nltk.data.path.append(r"C:\Users\76074\nltk_data")
# 读取的文件数据
file_path = r"/情感-语义检测分析模型/Second Part\数据集.csv"


# ========== 添加的中文字体支持函数 ==========
def configure_matplotlib_chinese_support():
    """配置Matplotlib支持中文显示"""
    # 方法1: 尝试使用系统已知中文字体名称
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                     'STSong', 'LiHei Pro', 'AppleGothic']

    # 检查系统可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])

    # 寻找第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            return

    # 方法2: 尝试使用字体文件路径
    font_files = [
        r'C:\Windows\Fonts\simhei.ttf',  # Windows 黑体
        r'C:\Windows\Fonts\msyh.ttc',  # Windows 微软雅黑
        r'C:\Windows\Fonts\simsun.ttc',  # Windows 宋体
        '/System/Library/Fonts/PingFang.ttc',  # Mac 苹方
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux 文泉驿
    ]

    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                # 添加字体到字体管理器
                font_prop = fm.FontProperties(fname=font_file)
                fm.fontManager.addfont(font_file)

                # 设置为默认字体
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用字体文件: {font_file}")
                return
            except:
                continue

    # 方法3: 最后尝试使用IPAex字体（通常跨平台兼容）
    try:
        plt.rcParams['font.family'] = ['IPAexGothic']
        print("使用备用字体: IPAexGothic")
    except:
        print("警告: 无法配置中文字体支持，中文显示可能不正常")


def download_nltk(pkg):
    try:
        nltk.data.find(f"{'corpora' if pkg in ['stopwords', 'wordnet'] else 'tokenizers'}/{pkg}")
        print(f"{pkg} 已存在，跳过下载。")
    except LookupError:
        nltk.download(pkg)
        print(f"{pkg} 下载完成。")


try:
    download_nltk('punkt')
    download_nltk('stopwords')
    download_nltk('wordnet')
    print("NLTK数据包检测与下载完成！")
except Exception as e:
    print(f"下载NLTK数据包时出错: {e}")


def load_data(file_path, num_samples=10):
    """加载验证集数据,只保留英文文本,限制条数"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                content = row[1]
                # 判断文本是否为英文（英文字母比例大于60%）
                letters = re.findall(r'[A-Za-z]', content)
                ratio = len(letters) / max(len(content), 1)
                if ratio > 0.6:
                    texts.append(content)
                    if len(texts) >= num_samples:
                        break
    return texts


def preprocess_text(text):
    # 转小写并去除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # 按空格切分
    tokens = text.split()
    # 去停用词和短词
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens


def LLM_analysis_topic(topic_words):
    """使用大模型分析主题内容 - 使用ollama.chat()方法"""
    prompt = f"""你是一个主题分析专家，请分析以下关键词组成的主题代表什么内容，并简要概括主题内容（字数要求在50字以内）。
关键词：{', '.join(topic_words)}
"""

    try:
        # 使用ollama.chat方法调用模型
        response = ollama.chat(
            model='gemma3:12b-it-qat',
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={'temperature': 0.2}
        )

        # 获取模型响应
        result = response['message']['content'].strip()
        return result if result else "模型未返回有效结果"

    except Exception as e:
        print(f"调用模型出错: {e}")
        return f"分析失败: {str(e)}"


def main():
    # 0. 配置中文字体支持
    configure_matplotlib_chinese_support()

    # 1. 加载数据
    texts = load_data(file_path, num_samples=10)  # 限制10条
    print(f"加载数据完成，共{len(texts)}条文本")
    print("texts内容:", texts)

    # 2. 数据预处理
    processed_texts = [preprocess_text(text) for text in texts]

    # 3. 构建词典和语料库
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # 4. 训练LDA模型
    num_topics = 3
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=15
    )

    # 5. 可视化分析
    # 5.1 pyLDAvis可视化
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'pyLDAvis可视化结果.html')

    # 使用绝对路径打开HTML文件
    html_path = os.path.abspath('pyLDAvis可视化结果.html')
    webbrowser.open(f'file:///{html_path.replace("\\", "/")}')
    print(f"已打开可视化结果: {html_path}")

    # 5.2 生成词云
    plt.figure(figsize=(20, 4))
    for i in range(num_topics):
        plt.subplot(1, 5, i + 1)
        topic_words = dict(lda_model.show_topic(i, 20))
        wordcloud = WordCloud(
            width=400, height=400,
            background_color='white',
            font_path=None  # 使用系统默认字体
        ).generate_from_frequencies(topic_words)
        plt.imshow(wordcloud)
        plt.title(f'主题 {i + 1}')  # 使用中文标题
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('主题词云图.png')
    print("主题词云图已保存")

    # 5.3 热力图
    doc_topics = np.zeros((len(texts), num_topics))
    for i, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow)
        for topic_id, prob in topic_dist:
            doc_topics[i, topic_id] = prob

    plt.figure(figsize=(10, 8))
    sns.heatmap(doc_topics[:50],  # 只显示前50篇文档
                cmap='YlOrRd',
                xticklabels=[f'主题 {i + 1}' for i in range(num_topics)])
    plt.title('主题分布')  # 使用中文标题
    plt.savefig('主题分布结果.png')
    print("主题分布结果图已保存")

    # 6. 使用大模型分析主题
    print("\n主题内容分析部分")
    for i in range(num_topics):
        # 获取主题关键词
        topic_words = [w for w, _ in lda_model.show_topic(i, 10)]

        # 调用分析函数
        topic_analysis = LLM_analysis_topic(topic_words)

        # 打印结果
        print(f"\n主题 {i + 1}:")
        print(f"关键词: {', '.join(topic_words)}")
        print(f"主题解释: {topic_analysis}")
        print("-" * 70)  # 添加分隔线


if __name__ == "__main__":
    main()