import os
import pandas as pd
import shutil
from tqdm import tqdm
import fnmatch

def find_files(directory, pattern):
    """递归查找文件"""
    matches = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def merge_all_data(root_path='.', output_dir='merged_data', csv_pattern='video_data_final.csv'):
    """
    终极版合并脚本：强制从文件夹结构提取label
    
    参数:
        root_path: 根目录（包含多个类别子文件夹）
        output_dir: 输出目录
        csv_pattern: CSV文件名模式
    """
    
    print("🔍 正在搜索CSV文件...")
    csv_files = find_files(root_path, csv_pattern)
    
    if not csv_files:
        print(f"❌ 未找到任何匹配的CSV文件: {csv_pattern}")
        return
    
    print(f"✅ 找到 {len(csv_files)} 个CSV文件:")
    for i, path in enumerate(csv_files, 1):
        rel_path = os.path.relpath(path, root_path)
        # 提前计算label用于显示
        label = os.path.dirname(rel_path).split(os.sep)[0]
        print(f"   {i}. {rel_path}  (类别: {label})")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'covers'), exist_ok=True)
    
    global_counter = 0
    all_records = []
    
    # 处理每个CSV
    for csv_path in tqdm(csv_files, desc="处理CSV", ncols=80):
        try:
            df = pd.read_csv(csv_path)
            csv_dir = os.path.dirname(csv_path)
            
            # ==================== 核心：提取label ====================
            # 获取相对路径: "穿搭/video_data_final.csv" 或 "穿搭\video_data_final.csv"
            rel_path = os.path.relpath(csv_path, root_path)
            # 统一路径分隔符
            rel_path = rel_path.replace('\\', '/')
            # 提取第一级目录名作为label
            path_parts = rel_path.split('/')
            label = path_parts[0] if len(path_parts) > 1 else 'unknown'
            # ========================================================
            
            print(f"\n📁 正在处理: {rel_path}")
            print(f"   ├─ 强制使用label: '{label}'")
            print(f"   ├─ CSV原始列: {list(df.columns)}")
            
            # 查找covers文件夹
            covers_dir = os.path.join(csv_dir, 'covers')
            if not os.path.exists(covers_dir):
                print(f"   └─ ❌ 跳过: 未找到covers文件夹")
                continue
            
            img_count = len([f for f in os.listdir(covers_dir) if f.lower().endswith(('.jpg', '.png'))])
            print(f"   ├─ 图片数: {img_count}张")
            
            # 检测文件名列
            filename_col = next((col for col in ['filename', '文件名', 'file_name', '视频ID'] if col in df.columns), df.columns[0])
            print(f"   ├─ 文件名列: '{filename_col}'")
            
            # 按文件名排序
            df = df.sort_values(filename_col).reset_index(drop=True)
            
            processed = 0
            missing = 0
            
            for idx, row in df.iterrows():
                old_filename = str(row[filename_col])
                old_path = os.path.join(covers_dir, old_filename)
                
                if not os.path.exists(old_path):
                    missing += 1
                    continue
                
                # 复制重命名
                global_counter += 1
                new_filename = f"{global_counter}.jpg"
                new_path = os.path.join(output_dir, 'covers', new_filename)
                shutil.copy2(old_path, new_path)
                
                # ==================== 强制使用文件夹label ====================
                # 注意：这里直接传入label变量，而不是从row读取
                new_record = {
                    'filename': new_filename,
                    'play_count': row.get('play_count') or row.get('play count') or row.get('播放量') or 0,
                    'like_count': row.get('like_count') or row.get('like count') or row.get('点赞数') or 0,
                    'label': label,  # 强制使用文件夹名
                    'source_csv': rel_path,
                    'original_filename': old_filename
                }
                # ===========================================================
                
                all_records.append(new_record)
                processed += 1
            
            print(f"   └─ ✅ 成功: {processed}条 | 缺失: {missing}条")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue
    
    # 保存结果
    if all_records:
        merged_df = pd.DataFrame(all_records)
        
        # 显示label分布统计
        label_counts = merged_df['label'].value_counts()
        print(f"\n🏷️  Label提取统计:")
        for lbl, count in label_counts.items():
            print(f"      {lbl}: {count}条")
        
        # 最终列格式化
        final_column_mapping = {
            'filename': 'filename',
            'play_count': 'play_count',
            'like_count': 'like_'
            'count',
            'label': 'label'
        }
        
        final_df = merged_df[list(final_column_mapping.keys())].copy()
        final_df.rename(columns=final_column_mapping, inplace=True)
        
        output_csv = os.path.join(output_dir, 'merged_data.csv')
        final_df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"🎉 合并完成！")
        print(f"{'='*60}")
        print(f"📊 总计记录: {len(final_df)} 条")
        print(f"🖼️  图片数量: {len(os.listdir(os.path.join(output_dir, 'covers')))} 张")
        print(f"💾 CSV路径: {os.path.abspath(output_csv)}")
        print(f"📋 CSV列: {list(final_df.columns)}")
        print(f"{'='*60}")
        
        return final_df
    else:
        print("❌ 无数据可保存")
        return pd.DataFrame()

# ==================== 运行入口 ====================
if __name__ == "__main__":
    print("🚀 启动最终版合并脚本 (强制文件夹label)")
    print("="*60)
    
    # 交互式输入
    root = input("请输入根目录 (默认: ./): ").strip() or '.'
    output = input("请输入输出目录 (默认: merged_data): ").strip() or 'merged_data'
    
    # 自动检测模式
    possible_patterns = ['video_data_final.csv', 'video_data.csv', '*data.csv', '*.csv']
    detected_pattern = None
    
    for pattern in possible_patterns:
        test_files = find_files(root, pattern)
        if test_files:
            detected_pattern = pattern
            print(f"✅ 自动检测到CSV模式: {pattern} ({len(test_files)}个文件)")
            break
    
    if not detected_pattern:
        detected_pattern = 'video_data_final.csv'
        print(f"⚠️ 未检测到，使用默认: {detected_pattern}")
    
    pattern_input = input(f"确认CSV模式 (回车使用 {detected_pattern}): ").strip()
    csv_pattern = pattern_input or detected_pattern
    
    print("\n" + "="*60)
    print("📂 配置:")
    print(f"   根目录: {os.path.abspath(root)}")
    print(f"   输出目录: {os.path.abspath(output)}")
    print(f"   CSV模式: {csv_pattern}")
    print("="*60)
    
    # 执行合并
    merged = merge_all_data(root_path=root, output_dir=output, csv_pattern=csv_pattern)
    
    if not merged.empty:
        print(f"\n✅ 完成！数据在: {os.path.abspath(output)}")
        print(f"\n📊 前5行预览:")
        print(merged.head())