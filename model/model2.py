# modul2_fix.py
import os
os.environ['JOBLIB_TEMP_FOLDER'] = r'D:\temp_joblib'   # 避免中文路径
import pandas as pd
import numpy as np
from scipy.stats import boxcox, inv_boxcox
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score

# ---------- 1. 读数据 ----------
df       = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')
visual_X = np.load(r'D:\bilitest\features\visual_X.npy')
text_X   = np.load(r'D:\bilitest\features\text_X.npy')
X        = np.hstack([visual_X, text_X])

# ---------- 2. Box-Cox 变换（保留 lambda） ----------
y_bc, lam_p = boxcox(df['play_count'].values + 1)   # +1 防 0

# ---------- 3. 特征截断 ----------
base = lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, verbose=-1)
base.fit(X, y_bc)
imp_thresh = np.percentile(base.feature_importances_, 80)
X_sel = X[:, base.feature_importances_ > imp_thresh]
print('特征维度:', X_sel.shape[1])

# ---------- 4. 训练/测试集 + 保存索引 ----------
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_sel, y_bc, np.arange(len(df)), test_size=0.2, random_state=42)

# ---------- 5. 超参搜索 ----------
param = {
    'n_estimators': [800, 1000],
    'max_depth': [8, 10],
    'learning_rate': [0.02, 0.05],
    'reg_lambda': [0.1, 1]
}
rs = RandomizedSearchCV(
    lgb.LGBMRegressor(verbose=-1),
    param, n_iter=6, cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,                      # 若还报错可改 1
    random_state=42)
rs.fit(X_tr, y_tr)

# ---------- 6. 评估 + 还原 ----------
pred_bc = rs.best_estimator_.predict(X_te)
pred_play = inv_boxcox(pred_bc, lam_p) - 1   # 先逆变换再减 1
true_play = df.iloc[idx_te]['play_count'].values

r2 = r2_score(true_play, pred_play)
mape = np.mean(np.abs((true_play - pred_play) / true_play)) * 100
print('真实空间 R²', f'{r2:.3f}', 'MAPE', f'{mape:.1f}%')

# ---------- 7. 保存模型（可选） ----------
import joblib
joblib.dump(rs.best_estimator_, r'D:\bilitest\report\lgb_play_bc.pkl')
print('✅ 模型已保存')