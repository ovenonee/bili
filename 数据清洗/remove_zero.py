# align_covers_with_csv.py
import os
import pandas as pd

# ========== 用户路径 ==========
CSV_PATH   = r'D:\bilitest\cleaned_data\result_no0.csv'   # 清洗后的 csv
COVERS_DIR = r'D:\bilitest\merged_data\covers'        # 原始 covers 文件夹
# =============================

import pandas as pd

csv_path = r'D:\bilitest\cleaned_data\result.csv'
print('数据行数（不含表头）:', len(pd.read_csv(csv_path)))


df        = pd.read_csv(CSV_PATH)
keep_set  = set(df['filename'])          # 需要保留的 jpg 文件名（含 .jpg）
all_files = {f for f in os.listdir(COVERS_DIR) if f.lower().endswith(('.jpg', '.jpeg'))}

to_delete = all_files - keep_set
print(f'共 {len(all_files)} 张图，csv 需要 {len(keep_set)} 张')
print(f'待删除 {len(to_delete)} 张：', *list(to_delete)[:20], '...' if len(to_delete)>20 else '')

# 确认无误后取消下一行注释真正删除
for f in to_delete:
     os.remove(os.path.join(COVERS_DIR, f))


print('已删除多余封面，covers 文件夹现与 csv 完全对齐！')