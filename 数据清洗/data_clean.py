import pandas as pd
import argparse
import sys
import os

def clean_data(df, output_path='cleaned_data.csv', save_file=True):
    """数据清洗函数 - 无CTR版本"""
    df_clean = df.copy()
    initial_total = len(df_clean)
    
    print("="*50)
    print("开始数据清洗...")
    print("="*50)
    
    # 检查必需列（CTR已移除）
    required_cols = ['play_count', 'like_count', 'filename']
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        print(f"❌ 错误：数据缺少必需的列: {', '.join(missing_cols)}")
        return None
    
    # ============ 跳过CTR规则 ============
    print("⚠️  未检测到CTR或click_count列，跳过CTR清洗规则")
    # =====================================
    
    # 规则2：删除播放量过低（<100）
    initial_len = len(df_clean)
    df_clean = df_clean[df_clean['play_count'] >= 100]
    print(f"🗑️  删除低播放量: {initial_len - len(df_clean)} 条")
    
    # 规则3：删除点赞数异常（>播放量×10）
    initial_len = len(df_clean)
    df_clean = df_clean[df_clean['like_count'] <= df_clean['play_count'] * 10]
    print(f"🗑️  删除异常点赞: {initial_len - len(df_clean)} 条")
    
    # 规则4：删除重复filename
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset='filename', keep='first')
    print(f"🗑️  删除重复文件: {initial_len - len(df_clean)} 条")
    
    # 规则5：删除缺失值
    initial_len = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"🗑️  删除缺失值: {initial_len - len(df_clean)} 条")
      # ===== 新增：去掉播放量为0或点赞量为0的行 =====
    initial_len = len(df_clean)
    df_clean = df_clean[(df_clean['play_count'] > 0) & (df_clean['like_count'] > 0)]
    print(f"🗑️  删除播放/点赞为0: {initial_len - len(df_clean)} 条")
    # ==========================================
    
    # 汇总统计
    print("\n" + "="*50)
    print(f"✅ 清洗后数据量: {len(df_clean)} 条")
    print(f"📊 保留率: {len(df_clean)/initial_total*100:.1f}%")
    print("="*50)
    
    # 保存数据
    if save_file:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 创建输出目录: {output_dir}")
        
        df_clean.to_csv(output_path, index=False)
        print(f"\n💾 数据已保存至: {os.path.abspath(output_path)}")
    
    return df_clean

def main():
    parser = argparse.ArgumentParser(description='数据清洗脚本（无CTR版本）')
    parser.add_argument('input_file', help='输入数据文件路径')
    parser.add_argument('-o', '--output', default='cleaned_data.csv', help='输出路径')
    parser.add_argument('--no-save', action='store_true', help='仅清洗不保存')
    parser.add_argument('--sheet', default=0, help='Excel工作表')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ 错误：输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    try:
        file_ext = os.path.splitext(args.input_file)[1].lower()
        print(f"📂 正在读取文件: {args.input_file}")
        
        if file_ext == '.csv':
            df = pd.read_csv(args.input_file)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(args.input_file, sheet_name=args.sheet)
        else:
            print(f"❌ 错误：不支持的文件格式 '{file_ext}'")
            sys.exit(1)
        
        print(f"📊 成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"📋 数据列名: {df.columns.tolist()}")
        
        df_cleaned = clean_data(df, output_path=args.output, save_file=not args.no_save)
        
        if df_cleaned is not None:
            print("\n✨ 数据清洗完成！")
        else:
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ 执行错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()