# bili_perfect_crawler.py
import requests
import time
import os
import pandas as pd
import urllib.parse
import random
from tqdm import tqdm

def crawl_perfect(target_total=10000, pages_per_keyword=20):
    """
    完美版B站爬虫：自动处理所有URL格式，彻底解决'cover_url'问题
    """
    
    # ==================== 核心配置区（必须修改） ====================
    COOKIE = 'buvid3=B1EDF9ED-D91F-EBF8-48F0-1837F9A300FE47830infoc; b_nut=1762936547; i-wanna-go-back=-1; _uuid=8C54395C-CF9D-2179-310103-EBC918CEDD9748783infoc; FEED_LIVE_VERSION=V8'
    KEYWORD_POOL = [
        "美食", "旅行", "学习", "萌宠", "游戏", "科技", "健身", "音乐",
        "舞蹈", "电影", "动漫", "搞笑", "手工", "摄影", "穿搭", "美妆",
        "汽车", "职场", "心理学", "历史", "法律", "育儿", "装修", "园艺"
    ]
    # ===================================================
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Cookie': COOKIE,
    }
    
    os.makedirs('covers', exist_ok=True)
    
    # 断点续传
    data_file = 'video_data_perfect.csv'
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        last_index = existing_df['filename'].astype(int).max() if len(existing_df) > 0 else 0
        all_data = existing_df.to_dict('records')
        print(f"📂 加载已有数据: {len(existing_df)} 条")
    else:
        all_data = []
        last_index = 0
        print(f"🆕 新任务，目标: {target_total} 条")
    
    current_index = last_index
    remaining = target_total - current_index
    
    if remaining <= 0:
        print(f"✅ 目标已达成！当前: {current_index} 条")
        return pd.DataFrame(all_data)
    
    print(f"🎯 目标: {target_total} | 还需: {remaining}")
    
    for keyword in KEYWORD_POOL:
        if current_index >= target_total:
            break
        
        print(f"\n{'='*50}")
        print(f"🔍 {keyword} | 进度: {current_index}/{target_total}")
        print(f"{'='*50}")
        
        for page in tqdm(range(1, pages_per_keyword + 1), desc=f"{keyword}", ncols=80):
            if current_index >= target_total:
                break
            
            # 页间延迟
            page_delay = random.uniform(5, 8)
            time.sleep(page_delay)
            
            try:
                url = f"https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword={urllib.parse.quote(keyword)}&page={page}"
                response = requests.get(url, headers=headers, timeout=20)
                
                if response.status_code == 412:
                    print(f"\n⚠️ 反爬，暂停10分钟...")
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
                
                success_count = 0
                fail_count = 0
                
                for video in videos:
                    if current_index >= target_total:
                        break
                    
                    try:
                        # ==================== 完美URL处理逻辑 ====================
                        # 1. 安全获取
                        pic_url = video.get('pic', '').strip()
                        
                        # 2. 空值检查
                        if not pic_url:
                            fail_count += 1
                            continue
                        
                        # 3. 统一转换为字符串
                        if isinstance(pic_url, list):
                            pic_url = pic_url[0] if pic_url else ''
                        elif not isinstance(pic_url, str):
                            pic_url = str(pic_url)
                        
                        # 4. 规范化处理（处理所有可能格式）
                        if pic_url.startswith('//'):
                            cover_url = 'https:' + pic_url
                        elif pic_url.startswith('/bfs/'):
                            cover_url = 'https://i0.hdslb.com' + pic_url
                        elif pic_url.startswith('http://'):
                            cover_url = pic_url.replace('http://', 'https://')
                        else:
                            cover_url = pic_url
                        
                        # 5. 最终验证
                        if not cover_url.startswith('https://'):
                            fail_count += 1
                            continue
                        
                        # ==================== 下载 ====================
                        download_delay = random.uniform(1, 2)
                        time.sleep(download_delay)
                        
                        img_response = requests.get(cover_url, timeout=15, headers={'User-Agent': headers['User-Agent']})
                        
                        if img_response.status_code != 200:
                            fail_count += 1
                            print(f"  ❌ 下载失败: {img_response.status_code}")
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
                        
                        # 进度显示
                        if current_index % 10 == 0:
                            print(f"  ✅ {current_index}/{target_total} ({current_index/target_total*100:.1f}%)")
                        
                    except Exception as e:
                        fail_count += 1
                        # 关键修复：不打印变量名
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
    
    # 最终保存
    df_final = pd.DataFrame(all_data)
    df_final.to_csv(data_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"🏁 完成！总数据: {len(df_final)} 条")
    print(f"CSV格式: filename,play_count,like_count")
    print(f"封面: covers/1.jpg ~ covers/{current_index}.jpg")
    print(f"{'='*50}")
    
    return df_final

if __name__ == "__main__":
    print("🚀 启动B站爬虫（完美修复版）")
    print("="*50)
    
    # 先测试100条
    data = crawl_perfect(
        target_total=10000,
        pages_per_keyword=30
    )