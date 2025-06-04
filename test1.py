#!/usr/bin/env python3
# scihub_downloader.py - 直接从Sci-Hub下载论文的工具

import requests
import time
import random
import os
import re
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse

# 可用的Sci-Hub镜像网址
SCIHUB_MIRRORS = [
    "https://sci-hub.ru",
    "https://sci-hub.st",
    "https://sci-hub.se",
    "https://sci-hub.ee", 
]

# 用户代理列表，避免被识别为机器人
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
]

def get_random_user_agent():
    """返回随机用户代理"""
    return random.choice(USER_AGENTS)

def clean_filename(title, doi):
    """生成合适的文件名"""
    if title:
        # 清理标题中的非法字符
        cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
        # 限制长度
        cleaned = cleaned.strip()[:100]
        # 替换空格
        cleaned = cleaned.replace(" ", "_")
    else:
        cleaned = "unknown"
    
    # 添加DOI（如果有）
    if doi:
        cleaned_doi = doi.replace("/", "_")
        return f"{cleaned}_{cleaned_doi}.pdf"
    else:
        return f"{cleaned}.pdf"

def find_pdf_link(html_content, base_url):
    """从Sci-Hub页面提取PDF下载链接"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 尝试多种方式找到PDF链接
        
        # 1. 查找embed标签
        embed = soup.find('embed')
        if embed and embed.get('src'):
            src = embed.get('src')
            if src.startswith('//'):
                return f"https:{src}"
            elif src.startswith('/'):
                return f"{base_url}{src}"
            return src
        
        # 2. 查找iframe标签
        iframe = soup.find('iframe')
        if iframe and iframe.get('src'):
            src = iframe.get('src')
            if src.startswith('//'):
                return f"https:{src}"
            elif src.startswith('/'):
                return f"{base_url}{src}"
            return src
        
        # 3. 查找下载按钮
        download_btn = soup.find('a', id='download')
        if download_btn and download_btn.get('href'):
            href = download_btn.get('href')
            if href.startswith('//'):
                return f"https:{href}"
            elif href.startswith('/'):
                return f"{base_url}{href}"
            return href
        
        # 4. 查找可能包含PDF关键字的链接
        pdf_links = soup.find_all('a', href=lambda href: href and ('pdf' in href.lower()))
        for link in pdf_links:
            href = link.get('href')
            if href.startswith('//'):
                return f"https:{href}"
            elif href.startswith('/'):
                return f"{base_url}{href}"
            return href
        
        # 5. 查找可能的下载区域
        download_div = soup.find('div', id='download')
        if download_div:
            buttons = download_div.find_all('button')
            for button in buttons:
                onclick = button.get('onclick', '')
                match = re.search(r"location.href='(.+?)'", onclick)
                if match:
                    href = match.group(1)
                    if href.startswith('//'):
                        return f"https:{href}"
                    elif href.startswith('/'):
                        return f"{base_url}{href}"
                    return href
        
        # 6. 提取页面中任何看起来像PDF链接的东西
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href and (href.endswith('.pdf') or '/pdf/' in href):
                if href.startswith('//'):
                    return f"https:{href}"
                elif href.startswith('/'):
                    return f"{base_url}{href}"
                return href
        
        return None
    except Exception as e:
        print(f"解析HTML时出错: {e}")
        return None

def download_from_scihub(doi, output_dir, use_mirror=None, delay=3):
    """从Sci-Hub下载指定DOI的论文"""
    if not doi:
        print("错误: 需要提供DOI")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定要使用的镜像
    mirrors = [use_mirror] if use_mirror else SCIHUB_MIRRORS
    
    for mirror in mirrors:
        try:
            # 构建Sci-Hub URL
            url = f"{mirror}/{doi}"
            print(f"尝试从 {url} 下载...")
            
            # 设置请求头
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # 获取Sci-Hub页面
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"无法访问 {mirror}, 状态码: {response.status_code}")
                continue
            
            # 提取PDF下载链接
            pdf_link = find_pdf_link(response.text, mirror)
            
            if not pdf_link:
                print(f"在 {mirror} 上找不到PDF下载链接")
                continue
            
            print(f"找到PDF链接: {pdf_link}")
            
            # 确定文件名
            # 尝试从页面提取标题
            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.title.string if soup.title else None
            if page_title and "sci-hub" in page_title.lower():
                # 移除"Sci-Hub | "前缀
                page_title = re.sub(r'^Sci-Hub\s*[|:]\s*', '', page_title, flags=re.IGNORECASE)
            
            filename = clean_filename(page_title, doi)
            output_path = os.path.join(output_dir, filename)
            
            # 下载PDF
            print(f"下载PDF到: {output_path}")
            pdf_headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'application/pdf,application/octet-stream',
                'Referer': url,
            }
            
            pdf_response = requests.get(pdf_link, headers=pdf_headers, timeout=60, stream=True)
            
            if pdf_response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 检查文件大小
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # 如果小于1KB可能不是有效PDF
                    print(f"警告: 下载的文件非常小 ({file_size} 字节)，可能不是有效PDF")
                    # 读取文件内容查看是否有错误信息
                    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(500)
                        print(f"文件内容预览: {content}")
                    return False
                
                print(f"✓ 成功下载论文! ({file_size} 字节)")
                return True
            else:
                print(f"下载PDF失败，状态码: {pdf_response.status_code}")
        
        except Exception as e:
            print(f"从 {mirror} 下载时出错: {e}")
        
        # 添加延迟，避免被封
        time.sleep(delay)
    
    print("❌ 所有镜像都失败了")
    return False

def batch_download(input_file, output_dir, mirror=None, delay=3, start_idx=0, batch_size=None):
    """批量下载DOI列表中的论文"""
    # 读取DOI列表
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dois = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取输入文件出错: {e}")
        return
    
    print(f"从文件中读取了 {len(dois)} 个DOI")
    
    # 应用起始索引和批量大小
    if start_idx > 0:
        dois = dois[start_idx:]
        print(f"从第 {start_idx+1} 个DOI开始下载")
    
    if batch_size:
        dois = dois[:batch_size]
        print(f"本次将下载 {len(dois)} 个DOI")
    
    # 统计
    success_count = 0
    fail_count = 0
    
    # 下载每个DOI
    for i, doi in enumerate(dois):
        print(f"\n处理 [{i+1}/{len(dois)}]: {doi}")
        
        success = download_from_scihub(doi, output_dir, use_mirror=mirror, delay=delay)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # 在DOI之间添加延迟
        if i < len(dois) - 1:
            delay_time = delay + random.uniform(1, 3)  # 添加随机延迟
            print(f"等待 {delay_time:.1f} 秒...")
            time.sleep(delay_time)
    
    # 输出统计信息
    print("\n===== 下载统计 =====")
    print(f"总计DOI: {len(dois)}")
    print(f"成功下载: {success_count}")
    print(f"下载失败: {fail_count}")

def main():
    parser = argparse.ArgumentParser(description="从Sci-Hub直接下载论文")
    parser.add_argument("--input", "-i", required=True, help="包含DOI列表的文本文件")
    parser.add_argument("--output", "-o", default="./downloads", help="下载论文的输出目录")
    parser.add_argument("--mirror", "-m", help="指定使用的Sci-Hub镜像，如https://sci-hub.ru")
    parser.add_argument("--delay", "-d", type=float, default=3.0, help="请求之间的延迟(秒)")
    parser.add_argument("--start", "-s", type=int, default=0, help="从第几个DOI开始下载(0表示从第一个开始)")
    parser.add_argument("--batch", "-b", type=int, help="本次下载的DOI数量")
    
    args = parser.parse_args()
    
    # 运行批量下载
    batch_download(args.input, args.output, args.mirror, args.delay, args.start, args.batch)

if __name__ == "__main__":
    main()

