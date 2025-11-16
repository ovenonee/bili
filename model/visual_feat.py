import os, glob, tqdm
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import sobel
from sklearn.preprocessing import StandardScaler

COVER_DIR   = r'D:\bilitest\merged_data\covers'          # 封面图 .jpg 所在
CLEANED_CSV = r'D:\bilitest\cleaned_data\result_no0.csv'  # 清洗后的csv
SAVE_DIR    = r'D:\bilitest\features'        # 存放 *.npy
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_visual_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_np = np.array(img) / 255.0          # 归一化到0-1

    # 1. 平均RGB
    avg_rgb = img_np.mean(axis=(0, 1))

    # 2. HSV 直方图（20bin × 3）
    hsv = rgb2hsv(img_np)
    h_hist = np.histogram(hsv[...,0], bins=20, range=(0,1), density=True)[0]
    s_hist = np.histogram(hsv[...,1], bins=20, range=(0,1), density=True)[0]
    v_hist = np.histogram(hsv[...,2], bins=20, range=(0,1), density=True)[0]
    hsv_feat = np.concatenate([h_hist, s_hist, v_hist])

    # 3. 亮度 / 对比度
    gray = np.dot(img_np, [0.299, 0.587, 0.114])
    brightness = gray.mean()
    contrast = gray.std()

    # 4. 边缘密度
    edge_map = sobel(gray)
    edge_density = edge_map.mean()

    return np.concatenate([avg_rgb, hsv_feat, [brightness, contrast, edge_density]])

# --- 批量提取 --------------------------------------------------
cover_paths = glob.glob(os.path.join(COVER_DIR, '*.jpg'))
feats = []
for p in tqdm.tqdm(cover_paths, desc='Visual'):
    feats.append(extract_visual_features(p))

visual_X = StandardScaler().fit_transform(np.array(feats))
np.save(os.path.join(SAVE_DIR, 'visual_X.npy'), visual_X)
print('visual_X.shape =', visual_X.shape)   # (N, 66)

