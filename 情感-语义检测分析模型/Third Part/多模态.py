import requests
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from gensim import corpora, models as g_models
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import nltk
import re
import os
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
from torchvision import models as tv_models, transforms
from PIL import Image

# 确保停用词可用
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

OLLAMA_URL = "http://localhost:11434/api/generate"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行环境：{DEVICE}")


# 健康检查
def is_ollama_alive():
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        return r.status_code == 200
    except:
        return False


if not is_ollama_alive():
    print("[严重错误] 模型服务未启动，请确认 Ollama 是否运行在 http://localhost:11434")
    exit(1)


# 请求封装（连接失败最多重试3次）
def retry_request(prompt, retries=3, delay=5, fatal_on_fail=False):
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": "gemma3:12b-it-qat", "prompt": prompt, "stream": False},
                timeout=180
            )
            return response.json()['response'].strip()
        except Exception as e:
            print(f"[第 {attempt + 1} 次尝试] 请求失败：{e}")
            time.sleep(delay)
    if fatal_on_fail:
        print("[严重错误] 多次尝试后仍无法连接大模型，程序终止。")
        exit(1)
    return "默认响应"


print("一、数据加载阶段")
df = pd.read_csv("posts.txt", sep='\t')
df = df.head(30)  # 读取前30条数据
print(f"成功加载 {len(df)} 条记录")
print(f"数据字段: {df.columns.tolist()}")  # 显示所有字段名称

texts = df['post_text'].astype(str).tolist()
labels = df['label'].map({'fake': 0, 'real': 1}).tolist()

print("二、主题特征提取")


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    return text.lower()


tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
processed_texts = [[w for w in tokenizer.tokenize(clean_text(t)) if w.isalpha() and w not in stop_words] for t in texts]

dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]
lda_model = g_models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=5)


def get_topic_vector(bow):
    return [p for _, p in lda_model.get_document_topics(bow, minimum_probability=0)]


topic_vecs = torch.tensor([get_topic_vector(b) for b in corpus], dtype=torch.float32).to(DEVICE)

print("三、用户背景生成")
age_groups = ['young_adult', 'middle_aged', 'senior']
user_backgrounds = [random.choice(age_groups) for _ in texts]
CACHE_PATH = "用户背景.json"


def generate_background_desc(age_group):
    prompt = f"请为'{age_group}'年龄段的用户创建一个简短的生活背景描述（2-3句话）。"
    return retry_request(prompt)


background_descriptions = None
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            background_descriptions = json.load(f)
        if len(background_descriptions) != len(texts):
            print("缓存数据不匹配，重新生成")
            background_descriptions = None
    except:
        print("缓存读取异常，重新生成")
        background_descriptions = None

if background_descriptions is None:
    print("正在创建用户背景描述...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        background_descriptions = list(
            tqdm(executor.map(generate_background_desc, user_backgrounds), total=len(user_backgrounds)))
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(background_descriptions, f, ensure_ascii=False)
        print(f"已存储 {len(background_descriptions)} 个背景描述")

background_features = torch.tensor([[len(desc.split())] for desc in background_descriptions], dtype=torch.float32).to(
    DEVICE)

print("四、立场评分模拟")  # 修改为立场评分


def simulate_survey_response(args):
    news, background_desc = args
    prompt = (
        f"基于您的背景：{background_desc}\n"
        f"您对此新闻的立场评分是多少？（0-10分，10分表示完全支持/相信）\n"  # 修改为0-10分制
        f"新闻内容：{news}\n请仅提供数字。"
    )
    score_str = retry_request(prompt, retries=3, delay=5, fatal_on_fail=True)
    try:
        return [max(0, min(10, float(score_str)))]  # 修改为0-10分范围
    except:
        return [5.0]  # 默认值改为5.0（10分制的中立）


SURVEY_CACHE = "立场评分.json"  # 修改缓存文件名
survey_scores = None
if os.path.exists(SURVEY_CACHE):
    try:
        with open(SURVEY_CACHE, "r", encoding="utf-8") as f:
            survey_scores = json.load(f)
        if len(survey_scores) != len(texts):
            print("立场评分缓存不匹配，重新生成")  # 修改提示
            survey_scores = None
    except:
        print("立场评分缓存读取异常，重新生成")  # 修改提示
        survey_scores = None

if survey_scores is None:
    print("正在进行立场评分模拟...")  # 修改提示
    with ThreadPoolExecutor(max_workers=2) as executor:
        survey_scores = list(
            tqdm(executor.map(simulate_survey_response, zip(texts, background_descriptions)), total=len(texts)))
    with open(SURVEY_CACHE, "w", encoding="utf-8") as f:
        json.dump(survey_scores, f)
        print(f"已存储 {len(survey_scores)} 个立场评分结果")  # 修改提示

survey_features = torch.tensor(survey_scores, dtype=torch.float32).to(DEVICE)

print("五、情感特征分析")


def get_emotion_vector(text):
    prompt = (
        f"请评估以下文本的情感倾向：\n{text}\n"
        f"使用 [1,0,0] 表示积极，[0,1,0] 表示中性，[0,0,1] 表示消极。"
    )
    try:
        response = retry_request(prompt)
        vec = eval(response.strip())
        if isinstance(vec, list) and len(vec) == 3:
            return vec
    except:
        pass
    return [0, 1, 0]  # 默认中性


emo_vecs = torch.tensor([get_emotion_vector(text) for text in tqdm(texts, desc="情感分析进度")],
                        dtype=torch.float32).to(DEVICE)

print("六、视觉特征提取")
# 解决torchvision警告问题
image_model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
image_model.fc = nn.Identity()
image_model = image_model.to(DEVICE)
image_model.eval()

img_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

IMG_FOLDER = "images"
image_ids = df['image_id(s)'].astype(str).tolist()
print(f"图像标识列表: {image_ids}")


def extract_image_feature(image_id_str):
    # 处理多个图片ID的情况（用逗号分隔）
    image_ids = image_id_str.split(',')
    features = []

    for img_id in image_ids:
        img_id = img_id.strip()  # 去除空格
        # 尝试多种可能的图片路径格式
        path_jpg = os.path.join(IMG_FOLDER, f"{img_id}.jpg")
        path_png = os.path.join(IMG_FOLDER, f"{img_id}.png")
        path_no_ext = os.path.join(IMG_FOLDER, img_id)

        if os.path.exists(path_jpg):
            path = path_jpg
        elif os.path.exists(path_png):
            path = path_png
        elif os.path.exists(path_no_ext):
            path = path_no_ext
        else:
            print(f"注意：图片 {img_id} 未找到")
            continue

        try:
            img = Image.open(path).convert("RGB")
            img_tensor = img_preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = image_model(img_tensor).squeeze(0)
            features.append(feat)
        except Exception as e:
            print(f"图片处理异常 {path}: {str(e)}")

    # 如果有多个特征，取平均
    if features:
        return torch.stack(features).mean(dim=0)
    else:
        print(f"无可用图片: {image_id_str}")
        return torch.zeros(512)


image_features = torch.stack([extract_image_feature(i) for i in tqdm(image_ids, desc="视觉特征提取")], dim=0).to(DEVICE)

print("七、特征融合与模型训练")
fused_features = torch.cat([topic_vecs, emo_vecs, background_features, survey_features, image_features], dim=1)
target = torch.tensor(labels, dtype=torch.long).to(DEVICE)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    fused_features, target, range(len(fused_features)), test_size=0.2, random_state=42)

model = MLPClassifier(fused_features.shape[1]).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 模型训练
print("模型训练启动...")
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"训练轮次 {epoch + 1} 损失值: {loss.item():.4f}")

print("八、模型性能评估")
model.eval()
with torch.no_grad():
    y_pred = torch.argmax(model(X_test), dim=1)
print(classification_report(y_test.cpu(), y_pred.cpu()))
torch.save(model.state_dict(), "Model.pth")
print("模型参数已保存为 Model.pth")

print("九、用户群体分析")
test_user_backgrounds = [user_backgrounds[i] for i in test_indices]
results_df = pd.DataFrame({
    '真实标签': y_test.cpu().tolist(),
    '预测标签': y_pred.cpu().tolist(),
    '用户背景': test_user_backgrounds
})

for bg_group in set(test_user_backgrounds):
    group_df = results_df[results_df['用户背景'] == bg_group]
    if len(group_df) == 0:
        continue

    y_true_group = group_df['真实标签']
    y_pred_group = group_df['预测标签']

    # 处理小样本情况
    if len(y_true_group) > 0:
        acc = accuracy_score(y_true_group, y_pred_group)
        # 当只有一类样本时，使用macro平均避免错误
        if len(set(y_true_group)) == 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_group, y_pred_group, average='macro', zero_division=0
            )
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_group, y_pred_group, average='binary', zero_division=0
            )

        print(
            f"背景组：{bg_group} | 样本量：{len(group_df)} | 准确率: {acc:.3f} | 精确率: {precision:.3f} | 召回率: {recall:.3f} | F1值: {f1:.3f}")
    else:
        print(f"背景组：{bg_group} | 无测试样本")

# 打印未出现在测试集中的背景组
all_bg_groups = set(user_backgrounds)
test_bg_groups = set(test_user_backgrounds)
missing_bg = all_bg_groups - test_bg_groups
for bg in missing_bg:
    print(f"背景组：{bg} | 未包含在测试集中")

print("分析流程完成！")