import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer   # ① 导入
from snownlp import SnowNLP

COVER_DIR   = r'D:\bilitest\merged_data\covers'          # 封面图 .jpg 所在
CLEANED_CSV = r'D:\bilitest\cleaned_data\result_no0.csv'  # 清洗后的csv
SAVE_DIR    = r'D:\bilitest\features'        # 存放 *.npy
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取清洗后的表
df = pd.read_csv(CLEANED_CSV)

# 用 filename 列去掉后缀当标题
titles = df['filename'].str.replace('.jpg', '', regex=False).fillna('').tolist()

# 1. TF-IDF (1-2 gram, 最多 300 维)
tfidf = TfidfVectorizer(max_features=300,
                        ngram_range=(1, 2),
                        stop_words=['的','了','在','是','我','有','和','不','这','它'])
text_X = tfidf.fit_transform(titles).toarray()

# 2. 情感得分
sentiments = np.array([SnowNLP(t).sentiments for t in titles])  # 0~1
text_X = np.column_stack([text_X, sentiments])   # 301 维

np.save(os.path.join(SAVE_DIR, 'text_X.npy'), text_X)
print('text_X.shape =', text_X.shape)   # (N, 301)