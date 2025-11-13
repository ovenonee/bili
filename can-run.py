# bili_massive_crawler.py
import requests
import time
import os
import pandas as pd
import urllib.parse
from tqdm import tqdm

def crawl_massive_data(target_total=10000, keywords_per_batch=5, pages_per_keyword=50):
    """
    大规模B站爬虫：自动循环关键词，支持断点续传，目标10,000条数据
    
    参数:
        target_total: 总目标数据量（默认10000）
        keywords_per_batch: 每批同时爬取的关键词数量（默认5个）
        pages_per_keyword: 每个关键词爬取的页数（每页20条，默认50页=1000条）
    """
    
    # ==================== 核心配置区（必须修改） ====================
    COOKIE = 'buvid3=B1EDF9ED-D91F-EBF8-48F0-1837F9A300FE47830infoc; b_nut=1762936547; i-wanna-go-back=-1; _uuid=8C54395C-CF9D-2179-310103-EBC918CEDD9748783infoc; FEED_LIVE_VERSION=V8'
    
    # 可扩展关键词池（建议准备20+个关键词循环使用）
    KEYWORD_POOL = [
        "美食", "旅行", "学习", "萌宠", "游戏", "科技", "健身", "音乐", 
        "舞蹈", "电影", "动漫", "搞笑", "手工", "摄影", "穿搭", "美妆",
        "汽车", "职场", "心理学", "历史", "法律", "育儿", "装修", "园艺"
    ]
    # ============================================================
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': COOKIE,
    }
    
    os.makedirs('covers', exist_ok=True)
    
    # 断点续传：加载已有数据
    data_file = 'video_data_massive.csv'
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        seen_urls = set(existing_df['cover_url'].tolist())
        all_data = existing_df.to_dict('records')
        start_count = len(existing_df)
        print(f"📂 加载已有数据: {start_count} 条")
    else:
        all_data = []
        seen_urls = set()
        start_count = 0
    
    # 计算需要爬取的量
    remaining = target_total - start_count
    if remaining <= 0:
        print(f"✅ 目标已达成！当前数据量: {start_count}")
        return pd.DataFrame(all_data)
    
    print(f"🎯 目标总量: {target_total} | 还需爬取: {remaining}")
    
    # 自动计算需要的关键词批次
    needed_batches = -(-remaining // (keywords_per_batch * pages_per_keyword * 20))  # 向上取整
    print(f"预计需要 {needed_batches} 个关键词批次")
    
    batch_count = 0
    
    # 主循环：按批次处理关键词
    for i in range(0, len(KEYWORD_POOL), keywords_per_batch):
        if len(all_data) >= target_total:
            break
            
        current_keywords = KEYWORD_POOL[i:i+keywords_per_batch]
        print(f"\n{'='*60}")
        print(f"📦 批次 {batch_count+1}/{needed_batches} | 关键词: {current_keywords}")
        print(f"{'='*60}")
        
        # 遍历当前批次的关键词
        for keyword in current_keywords:
            if len(all_data) >= target_total:
                print("🎉 已达到目标数量，提前结束")
                break
            
            print(f"\n🔍 正在爬取: '{keyword}'")
            
            # 为每个关键词创建独立进度条
            for page in tqdm(range(1, pages_per_keyword + 1), 
                           desc=f"{keyword:>6s}", 
                           ncols=80):
                
                success = False
                retry_count = 0
                
                while not success and retry_count < 3:
                    try:
                        # 关键修复：URL编码正确，无空格
                        encoded_keyword = urllib.parse.quote(keyword)
                        url = f"https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword={encoded_keyword}&page={page}"
                        
                        response = requests.get(url, headers=headers, timeout=15)
                        
                        # 反爬处理
                        if response.status_code == 412:
                            print(f"\n⚠️ 触发反爬，暂停60秒...")
                            time.sleep(60)
                            retry_count += 1
                            continue
                        elif response.status_code != 200:
                            print(f"\n❌ 状态码异常: {response.status_code}")
                            retry_count += 1
                            time.sleep(10)
                            continue
                        
                        json_data = response.json()
                        
                        if json_data.get('code') != 0:
                            print(f"\n❌ API错误: {json_data.get('message')}")
                            break
                        
                        videos = json_data['data'].get('result', [])
                        
                        for video in videos:
                            if len(all_data) >= target_total:
                                success = True
                                break
                            
                            try:
                                cover_url = video.get('pic', '')
                                if cover_url.startswith('//'):
                                    cover_url = 'https:' + cover_url
                                
                                if cover_url in seen_urls or not cover_url:
                                    continue
                                
                                title = video.get('title', '').replace('<em class="keyword">', '').replace('</em>', '')
                                
                                all_data.append({
                                    'keyword': keyword,
                                    'title': title,
                                    'cover_url': cover_url,
                                    'play_count': str(video.get('play', 0)),
                                    'like_count': str(video.get('like', 0))
                                })
                                
                                seen_urls.add(cover_url)
                                
                                # 每50条保存一次
                                if len(all_data) % 50 == 0:
                                    df_temp = pd.DataFrame(all_data)
                                    df_temp.to_csv(data_file, index=False, encoding='utf-8-sig')
                                    print(f"\n💾 自动保存: {len(all_data)} 条")
                                
                                # 下载封面
                                img_response = requests.get(cover_url, timeout=10, headers={'User-Agent': headers['User-Agent']})
                                with open(f'covers/{len(all_data)}.jpg', 'wb') as f:
                                    f.write(img_response.content)
                                
                                time.sleep(0.3)  # 请求间隔
                                
                            except Exception as e:
                                print(f"  解析失败: {e}")
                                continue
                        
                        success = True
                        time.sleep(1.5)  # 页面间隔
                        
                    except Exception as e:
                        print(f"\n  请求失败: {e}，重试 {retry_count+1}/3")
                        retry_count += 1
                        time.sleep(5)
        
        batch_count += 1
        
        # 批次间休息
        if len(all_data) < target_total:
            print(f"\n☕ 批次完成，休息30秒...")
            time.sleep(30)
    
    # 最终保存
    df_final = pd.DataFrame(all_data)
    df_final.to_csv(data_file, index=False, encoding='utf-8-sig')
    
    # 统计报告
    print(f"\n{'='*60}")
    print(f"🏁 爬取完成！")
    print(f"📊 总计数据: {len(df_final)} 条")
    print(f"🎯 目标完成度: {len(df_final)}/{target_total} ({len(df_final)/target_total*100:.1f}%)")
    print(f"📈 关键词分布:\n{df_final['keyword'].value_counts().head(10)}")
    print(f"💾 文件已保存: {data_file}")
    print(f"{'='*60}")
    
    return df_final

if __name__ == "__main__":
    # ==================== 参数配置 ====================
    # 目标：爬取10,000条数据
    # 策略：4个关键词 × 125页 = 10,000条（每页20条）
    
    print("🚀 启动大规模B站爬虫")
    print("="*60)
    
    # 快速开始：直接运行默认配置
    data = crawl_massive_data(
        target_total=10000,
        keywords_per_batch=4,      # 每批4个关键词
        pages_per_keyword=125      # 每个关键词125页
    )
    
    # 如需自定义，取消下面注释
    # data = crawl_massive_data(
    #     target_total=5000,       # 目标5000条
    #     keywords_per_batch=2,    # 每批2个关键词
    #     pages_per_keyword=50     # 每个关键词50页
    # )