# bili_final_crawler.py
import requests
import time
import os
import pandas as pd
import urllib.parse
import random
from tqdm import tqdm

# ==================== 全局配置区（用户需修改） ====================
# 从浏览器F12获取的完整Cookie（过期需更新）
COOKIE = 'buvid3=D5B97EBF-167B-3A7D-04B6-2C016C151A0331350infoc; b_nut=1753971030; i-wanna-go-back=-1; _uuid=CE910272B-9962-93EF-D4F7-C8D516E104C4E47670infoc; FEED_LIVE_VERSION=V8'

# 关键词池（24个关键词，可扩展）
KEYWORD_POOL = [
    "搞笑", "美食", "学习", "萌宠", "游戏", "科技", "健身", "音乐",
    "舞蹈", "电影", "动漫", "搞笑", "手工", "摄影", "穿搭", "美妆",
    "汽车", "职场", "心理学", "历史", "法律", "育儿", "装修", "园艺"
]
# ============================================================

def crawl_bilibili_videos(target_total=10000, pages_per_keyword=30):
    """
    B站大规模视频爬虫
    
    参数：
        target_total: 目标爬取数量（默认10000）
        pages_per_keyword: 每个关键词爬取页数（推荐30页）
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Cookie': COOKIE,
    }
    
    os.makedirs('covers', exist_ok=True)
    data_file = 'video_data_final.csv'
    
    # 断点续传
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        last_index = existing_df['filename'].astype(int).max() if len(existing_df) > 0 else 0
        all_data = existing_df.to_dict('records')
        print(f"📂 加载已有数据: {len(existing_df)} 条，最后编号: {last_index}.jpg")
    else:
        all_data = []
        last_index = 0
        print(f"🆕 新任务，目标: {target_total} 条")
    
    current_index = last_index
    if current_index >= target_total:
        print(f"✅ 目标已达成！当前: {current_index} 条")
        return pd.DataFrame(all_data)
    
    print(f"🎯 总量: {target_total} | 还需: {target_total - current_index}")
    
    # 主循环
    for keyword in KEYWORD_POOL:
        if current_index >= target_total:
            break
        
        print(f"\n{'='*50}")
        print(f"🔍 {keyword} | 进度: {current_index}/{target_total}")
        print(f"{'='*50}")
        
        for page in tqdm(range(1, pages_per_keyword + 1), desc=f"{keyword:>6s}", ncols=80):
            if current_index >= target_total:
                break
            
            # 页间延迟
            page_delay = random.uniform(5, 8)
            print(f"⏳ 页{page:>3d}等待: {page_delay:.1f}s")
            time.sleep(page_delay)
            
            try:
                url = f"https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword={urllib.parse.quote(keyword)}&page={page}"
                response = requests.get(url, headers=headers, timeout=20)
                
                if response.status_code == 412:
                    print(f"\n⚠️ 触发412反爬，暂停10分钟...")
                    time.sleep(600)
                    continue
                elif response.status_code != 200:
                    print(f"\n❌ 状态码: {response.status_code}")
                    time.sleep(10)
                    continue
                
                json_data = response.json()
                
                if json_data.get('code') != 0:
                    print(f"\n❌ API错误: {json_data.get('message')}")
                    break
                
                videos = json_data['data'].get('result', [])
                print(f"📦 本页返回 {len(videos)} 个视频")
                
                success_count = 0
                fail_count = 0
                
                for video in videos:
                    if current_index >= target_total:
                        break
                    
                    try:
                        # URL处理
                        pic_url = video.get('pic', '').strip()
                        if not pic_url:
                            fail_count += 1
                            continue
                        
                        if isinstance(pic_url, list):
                            pic_url = pic_url[0] if pic_url else ''
                        
                        if pic_url.startswith('//'):
                            cover_url = 'https:' + pic_url
                        elif pic_url.startswith('/bfs/'):
                            cover_url = 'https://i0.hdslb.com' + pic_url
                        elif pic_url.startswith('http://'):
                            cover_url = pic_url.replace('http://', 'https://')
                        else:
                            cover_url = pic_url
                        
                        if not cover_url.startswith('https://'):
                            fail_count += 1
                            continue
                        
                        # 下载
                        download_delay = random.uniform(1, 2)
                        time.sleep(download_delay)
                        
                        img_response = requests.get(cover_url, timeout=15, headers={'User-Agent': headers['User-Agent']})
                        
                        if img_response.status_code != 200:
                            fail_count += 1
                            continue
                        
                        if len(img_response.content) < 1024:
                            fail_count += 1
                            continue
                        
                        current_index += 1
                        filename = f"{current_index}.jpg"
                        with open(f'covers/{filename}', 'wb') as f:
                            f.write(img_response.content)
                        
                        # CSV精简
                        all_data.append({
                            'filename': filename,
                            'play_count': int(video.get('play', 0)),
                            'like_count': int(video.get('like', 0))
                        })
                        
                        success_count += 1
                        
                        # 每10条保存
                        if current_index % 10 == 0:
                            df_temp = pd.DataFrame(all_data)
                            df_temp.to_csv(data_file, index=False)
                            print(f"\n💾 自动保存: {current_index}/{target_total} ({current_index/target_total*100:.1f}%)")
                        
                    except Exception as e:
                        fail_count += 1
                        print(f"  ❌ {type(e).__name__}: {str(e)[:30]}")
                        continue
                
                print(f"📄 第{page}页: ✓{success_count} ✗{fail_count}")
                
            except Exception as e:
                print(f"\n❌ 页请求失败: {type(e).__name__}")
                time.sleep(10)
                continue
        
        if current_index < target_total:
            batch_delay = random.uniform(300, 400)
            print(f"\n☕ 休息 {batch_delay/60:.1f} 分钟...")
            time.sleep(batch_delay)
    
    # 最终强制保存
    if all_data:
        df_final = pd.DataFrame(all_data)
        df_final.to_csv(data_file, index=False)
        print(f"\n💾 最终CSV已保存: {os.path.abspath(data_file)}")
        print(f"📊 包含 {len(df_final)} 条数据")
    else:
        print("⚠️ 无数据可保存")
    
    return df_final

# ==================== 运行入口 ====================
if __name__ == "__main__":
    print("🚀 启动B站爬虫（完整修复版）")
    print("="*50)
    
    # 配置参数（用户可修改）
    TARGET = 10000      # 目标数量
    PAGES_PER_KEYWORD = 30  # 每个关键词页数
    
    # 预估信息
    estimated_total = len(KEYWORD_POOL) * PAGES_PER_KEYWORD * 16  # 按80%成功率估算
    print(f"📈 关键词数: {len(KEYWORD_POOL)}")
    print(f"📄 每关键词页数: {PAGES_PER_KEYWORD}")
    print(f"📊 预估总量: {estimated_total} 条")
    print(f"🎯 实际目标: {TARGET} 条")
    print("="*50)
    
    # 执行爬取
    data = crawl_bilibili_videos(
        target_total=TARGET,
        pages_per_keyword=PAGES_PER_KEYWORD
    )
    
    # 统计报告
    if not data.empty:
        print("\n" + "="*50)
        print("📈 爬取统计")
        print("="*50)
        print(f"✅ 成功: {len(data)} 条")
        print(f"💾 CSV路径: {os.path.abspath('video_data_final.csv')}")
        print(f"📁 封面路径: {os.path.abspath('covers/')}")
        print(f"📊 数据预览:\n{data.head()}")
        print("="*50)
        print("\n✅ 任务完成！")
