import re, jieba
from gensim.models import Word2Vec

# 中文停用词集合，可以继续追加
stop_words = {
    "的", "了", "是", "有", "不", "在", "我", "你", "他", "她", "它",
    "这", "那", "就", "都", "而", "及", "和", "与", "也", "又", "或",
    "则", "乃", "之", "等", "某", "些", "么", "吗", "呢", "吧", "啊",
    "于", "由", "从", "向", "以", "为", "被", "把", "将", "可", "曾",
    "本", "此", "每", "各", "看", "来", "去", "说", "道", "遂", "亦",
    "若", "岂", "忽", "但见", "只见", "里面", "一时", "一面", "一边","“","”"
}

text = open("xiyouji.txt", encoding="utf-8").read()
# 先按中文标点粗略切成句子
sentences = re.split(r"[。！？；!?;]", text)
# 再用 jieba 把每个句子切成词 + 【新增停用词过滤】
tokenized_sentences = [
    [w for w in jieba.cut(s) if w.strip() and w not in stop_words]
    for s in sentences if s.strip()
]

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,    # 词向量维度
    window=15,           # 上下文窗口，左右各 5 词
    min_count=3,        # 过滤低频词，出现少于3次的丢掉
    sg=1,               # 1=Skip-Gram, 0=CBOW
    epochs=15,          # 改小，防止过拟合
    workers=1,
)
#打印最相似词语
res = model.wv.most_similar("悟空", topn=10)#topn 返回相似度排名前10位
print(res)
print("齐天大圣" in model.wv)
print("沙悟净" in model.wv)
print("孙悟空" in model.wv)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# 配置中文
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 挑选前50个词做可视化
words = list(model.wv.index_to_key[:50])
word_vecs = np.array([model.wv[w] for w in words])
# t-SNE降维到2维
tsne = TSNE(n_components=2, random_state=42)
vecs_2d = tsne.fit_transform(word_vecs)
# 绘图
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    x, y = vecs_2d[i]
    plt.scatter(x, y)
    plt.text(x, y, word, fontsize=9)
plt.title("西游记 Word2Vec 词向量可视化（停用词过滤）")
plt.show()

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 拿词向量
vec_wukong = model.wv["悟空"]
vec_tangseng = model.wv["师父"]
vec_taizong = model.wv["玉帝"]

# 手动计算
sim_wk_ts = cosine_similarity(vec_wukong, vec_tangseng)
sim_wk_tz = cosine_similarity(vec_wukong, vec_taizong)
print(f"悟空 vs 师父 手动余弦相似度：{sim_wk_ts:.4f}")
print(f"悟空 vs 玉帝 手动余弦相似度：{sim_wk_tz:.4f}")

# gensim自带函数计算
print(f"悟空 vs 师父 gensim相似度：{model.wv.similarity('悟空','师父'):.4f}")
print(f"悟空 vs 玉帝 gensim相似度：{model.wv.similarity('悟空','玉帝'):.4f}")

# top10相似词
print("\n和悟空最相似的10个词：")
print(model.wv.most_similar("悟空", topn=10))