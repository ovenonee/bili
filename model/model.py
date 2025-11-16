# regression_final.py
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

# ---------- 1. 路径 ----------
COVER_DIR   = r'D:\bilitest\merged_data\covers'          # 封面图 .jpg 所在
CLEANED_CSV = r'D:\bilitest\cleaned_data\result_no0.csv'  # 清洗后的csv
SAVE_DIR    = r'D:\bilitest\features'        # 存放 *.npy
FEAT_DIR    = r'D:\bilitest\features'
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 2. 读数据 ----------
df       = pd.read_csv(CLEANED_CSV)
visual_X = np.load(os.path.join(FEAT_DIR, 'visual_X.npy'))
text_X   = np.load(os.path.join(FEAT_DIR, 'text_X.npy'))
X        = np.hstack([visual_X, text_X])

# 平滑目标
y_play = np.log1p(df['play_count'])
y_like = np.log1p(df['like_count'])

# ---------- 3. 训练/测试 ----------
X_train, X_test, yp_tr, yp_te, yl_tr, yl_te = train_test_split(
    X, y_play, y_like, test_size=0.2, random_state=42)

# ---------- 4. 回归报告 ----------
def reg_report(model, X_tr, y_tr, X_te, y_te, name=''):
    X_tr_df = pd.DataFrame(X_tr, columns=[f'f{i}' for i in range(X_tr.shape[1])])
    X_te_df = pd.DataFrame(X_te, columns=X_tr_df.columns)

    model.fit(X_tr_df, y_tr)
    pred = model.predict(X_te_df)

    rmse = np.sqrt(mean_squared_error(y_te, pred))   # ✅ 兼容旧版
    r2   = r2_score(y_te, pred)
    print(f'{name}  RMSE={rmse:.3f}  R²={r2:.3f}')
    return model, pred

# 4.1 播放量
lgb_p, pred_p = reg_report(
    LGBMRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                  verbose=-1, reg_lambda=0.1),
    X_train, yp_tr, X_test, yp_te, 'Play')

# 4.2 点赞量
lgb_l, pred_l = reg_report(
    LGBMRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                  verbose=-1, reg_lambda=0.1),
    X_train, yl_tr, X_test, yl_te, 'Like')

# ---------- 5. 还原真值并算 MAPE ----------
play_pred = np.expm1(pred_p)
like_pred = np.expm1(pred_l)
play_true = np.expm1(yp_te)
like_true = np.expm1(yl_te)

mape_play = np.mean(np.abs((play_true - play_pred) / play_true)) * 100
mape_like = np.mean(np.abs((like_true - like_true) / like_true)) * 100
print(f'MAPE 播放={mape_play:.1f}%  点赞={mape_like:.1f}%')

# ---------- 6. SHAP 全局解释（播放量） ----------
explainer = shap.TreeExplainer(lgb_p)
shap_values = explainer.shap_values(pd.DataFrame(X_test, columns=[f'f{i}' for i in range(X_test.shape[1])]))

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=[f'f{i}' for i in range(X_test.shape[1])]),
                  show=False, max_display=20)
plt.title('图1 封面特征对播放量(log)的SHAP贡献度')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'shap_play_log.png'), dpi=300)
plt.close()

# ---------- 7. 保存模型 ----------
import joblib
joblib.dump(lgb_p, os.path.join(SAVE_DIR, 'lgb_play.pkl'))
joblib.dump(lgb_l, os.path.join(SAVE_DIR, 'lgb_like.pkl'))
print('✅ 回归完成，模型与 SHAP 图已保存至', SAVE_DIR)