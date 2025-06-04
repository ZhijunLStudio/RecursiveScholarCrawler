# #!/usr/bin/env python3
# # downloader.py - Paper downloading functionality

# import os
# import sys
# import argparse
# import logging
# import time
# from pathlib import Path
# import json
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import random

# # 导入但不立即使用配置
# import config
# from utils import load_json, save_json, get_timestamp_str, Locker
# from doi_helper import download as doi_download

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class PaperDownloader:
#     def __init__(self, output_dir, max_workers=4, retry_failed=False, delay_between=2):
#         """Initialize the paper downloader."""
#         self.output_dir = Path(output_dir)
#         # 下载目录从配置获取
#         paths = config.get_configured_paths()
#         self.download_dir = Path(paths["download_dir"])
#         self.download_dir.mkdir(parents=True, exist_ok=True)
        
#         # Processing settings
#         self.max_workers = max_workers
#         self.retry_failed = retry_failed
#         self.delay_between = delay_between
        
#         # Load state
#         self.download_queue = load_json(paths["download_queue_file"], [])
#         self.download_results = load_json(paths["download_results_file"], {})
        
#         self.lock = threading.Lock()
        
#     def save_state(self):
#         """Save current state to files."""
#         paths = config.get_configured_paths()
#         with self.lock:
#             save_json(self.download_queue, paths["download_queue_file"])
#             save_json(self.download_results, paths["download_results_file"])
    
#     def get_pending_downloads(self):
#         """Get list of pending downloads."""
#         with self.lock:
#             # If retry_failed is True, include failed downloads
#             if self.retry_failed:
#                 # All items that don't have a successful result
#                 pending = [item for item in self.download_queue 
#                           if item["title"] not in self.download_results or 
#                           not self.download_results[item["title"]].get("success")]
#             else:
#                 # Only items with status "pending"
#                 pending = [item for item in self.download_queue if item["status"] == "pending"]
                
#             return pending
    
#     def download_paper(self, paper_item):
#         """Download a single paper and update results."""
#         title = paper_item["title"]
        
#         # Mark as in-progress
#         with self.lock:
#             for item in self.download_queue:
#                 if item["title"] == title:
#                     item["status"] = "downloading"
#                     item["download_attempt_time"] = get_timestamp_str()
#             self.save_state()
        
#         logger.info(f"Downloading: {title}")
        
#         try:
#             # Use doi_helper to download the paper
#             result = doi_download(title, output_directory=str(self.download_dir))
            
#             # Update download results
#             with self.lock:
#                 self.download_results[title] = {
#                     "success": result.get("success", False),
#                     "status": result.get("status", "unknown"),
#                     "file_path": result.get("file_path", ""),
#                     "doi": result.get("doi", ""),
#                     "timestamp": get_timestamp_str(),
#                     "original_request": paper_item
#                 }
                
#                 # Update queue item status
#                 for item in self.download_queue:
#                     if item["title"] == title:
#                         item["status"] = "completed" if result.get("success") else "failed"
#                         item["download_result"] = result.get("status", "unknown")
                
#                 self.save_state()
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error downloading '{title}': {e}", exc_info=True)
            
#             # Update on failure
#             with self.lock:
#                 self.download_results[title] = {
#                     "success": False,
#                     "status": f"error: {str(e)}",
#                     "timestamp": get_timestamp_str(),
#                     "original_request": paper_item
#                 }
                
#                 # Update queue item status
#                 for item in self.download_queue:
#                     if item["title"] == title:
#                         item["status"] = "failed"
#                         item["download_result"] = f"error: {str(e)}"
                
#                 self.save_state()
            
#             return {"success": False, "status": f"error: {str(e)}"}
    
#     def download_all_papers(self):
#         """Download all pending papers in parallel."""
#         pending_downloads = self.get_pending_downloads()
        
#         if not pending_downloads:
#             logger.info("No papers to download")
#             return
        
#         logger.info(f"Starting download of {len(pending_downloads)} papers with {self.max_workers} workers")
        
#         # Statistics
#         start_time = time.time()
#         success_count = 0
#         fail_count = 0
        
#         try:
#             # 使用分批处理以便看到进度
#             batch_size = min(20, len(pending_downloads))  # 每批最多20个
#             total_batches = (len(pending_downloads) + batch_size - 1) // batch_size
            
#             for batch_num in range(total_batches):
#                 start_idx = batch_num * batch_size
#                 end_idx = min((batch_num + 1) * batch_size, len(pending_downloads))
#                 current_batch = pending_downloads[start_idx:end_idx]
                
#                 logger.info(f"Processing batch {batch_num+1}/{total_batches} with {len(current_batch)} papers")
                
#                 # 为这一批创建线程池
#                 with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#                     futures = []
                    
#                     # 提交本批次的任务
#                     for item in current_batch:
#                         futures.append(executor.submit(self.download_paper, item))
                    
#                     # 处理结果
#                     for future in futures:
#                         try:
#                             result = future.result(timeout=300)  # 5分钟超时
#                             if result.get("success"):
#                                 success_count += 1
#                                 logger.info(f"Successfully downloaded: {result.get('file_path', 'unknown')}")
#                             else:
#                                 fail_count += 1
#                                 logger.warning(f"Download failed: {result.get('status', 'unknown error')}")
#                         except concurrent.futures.TimeoutError:
#                             fail_count += 1
#                             logger.error(f"Download worker timed out after 5 minutes")
#                         except Exception as e:
#                             fail_count += 1
#                             logger.error(f"Download worker failed: {e}")
                
#                 # 显示批次完成的统计
#                 batch_end_time = time.time()
#                 batch_runtime = batch_end_time - start_time
#                 logger.info(f"Batch {batch_num+1}/{total_batches} completed. Current stats - Success: {success_count}, Failed: {fail_count}")
                
#                 # 批次间隔
#                 if batch_num < total_batches - 1:
#                     logger.info(f"Waiting {self.delay_between}s before starting next batch...")
#                     time.sleep(self.delay_between)
            
#             # Final stats
#             end_time = time.time()
#             runtime = end_time - start_time
#             hours, remainder = divmod(runtime, 3600)
#             minutes, seconds = divmod(remainder, 60)
#             runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
#             logger.info("=" * 50)
#             logger.info(f"Download complete. Runtime: {runtime_str}")
#             logger.info(f"Success: {success_count}, Failed: {fail_count}")
#             logger.info("=" * 50)
            
#         except KeyboardInterrupt:
#             logger.info("Download interrupted by user. Saving state.")
        
#         # Final save
#         self.save_state()


# # 在 downloader.py 中修改 main 函数
# def main():
#     # Define command line arguments
#     parser = argparse.ArgumentParser(description='Academic Paper Downloader')
#     parser.add_argument('--output-dir', required=True, help='Directory to store output files')
#     parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel downloads')
#     parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed downloads')
#     parser.add_argument('--delay', type=float, default=2.0, help='Delay between downloads in seconds')
    
#     args = parser.parse_args()
    
#     # 对于下载器，我们使用临时目录作为输入目录 (不传空字符串)
#     temp_input_dir = os.path.join(args.output_dir, "temp_input")
#     config.configure_paths(temp_input_dir, args.output_dir)
#     paths = config.get_configured_paths()
#     logger.info(f"配置下载路径: 输出={paths['output_dir']}, 下载={paths['download_dir']}")
    
#     # Initialize and run downloader
#     try:
#         downloader = PaperDownloader(
#             output_dir=args.output_dir,
#             max_workers=args.max_workers,
#             retry_failed=args.retry_failed,
#             delay_between=args.delay
#         )
        
#         logger.info("Starting paper downloads...")
#         downloader.download_all_papers()
        
#     except KeyboardInterrupt:
#         logger.info("Downloader stopped by user")
#     except Exception as e:
#         logger.error(f"Downloader stopped due to error: {e}", exc_info=True)


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# debug_download.py - 用于调试下载问题的脚本

import os
import sys
import time
import json
import argparse
import logging
import datetime
import requests
import traceback
from bs4 import BeautifulSoup
import re

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,  # 使用DEBUG级别获取最详细的日志
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 配置超时和请求头
TIMEOUT = 15  # 15秒超时，避免长时间等待
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

def get_timestamp_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_headers():
    return {'User-Agent': USER_AGENTS[0]}

def load_download_queue(file_path):
    """加载下载队列以进行调试"""
    try:
        logger.info(f"正在从 {file_path} 加载下载队列...")
        if not os.path.exists(file_path):
            logger.error(f"找不到文件: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"成功加载 {len(data)} 个下载项")
            return data
    except Exception as e:
        logger.error(f"加载下载队列时出错: {e}")
        logger.debug(traceback.format_exc())
        return []

def debug_get_doi(title):
    """调试DOI查找功能"""
    logger.info(f"正在查找标题的DOI: {title}")
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": title,
            "rows": 1,
            "sort": "score",
        }
        logger.debug(f"正在发送请求到 {url} 参数: {params}")
        
        # 添加超时和异常捕获
        start_time = time.time()
        response = requests.get(url, params=params, headers=get_headers(), timeout=TIMEOUT)
        elapsed = time.time() - start_time
        
        logger.debug(f"CrossRef请求完成，耗时: {elapsed:.2f}秒，状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                doi = items[0].get("DOI")
                if doi:
                    logger.info(f"找到DOI: {doi}")
                    return doi
            logger.warning(f"未找到DOI")
            return None
    except requests.exceptions.Timeout:
        logger.error(f"CrossRef请求超时 ({TIMEOUT}秒)")
    except requests.exceptions.RequestException as e:
        logger.error(f"CrossRef请求错误: {e}")
    except Exception as e:
        logger.error(f"查找DOI时出错: {e}")
        logger.debug(traceback.format_exc())
    
    return None

def debug_sci_hub_page(doi):
    """调试Sci-Hub页面访问"""
    logger.info(f"正在访问Sci-Hub页面，DOI: {doi}")
    try:
        base_url = "https://sci-hub.se/"
        sci_hub_url = base_url + doi
        logger.debug(f"正在请求: {sci_hub_url}")
        
        # 添加超时和异常捕获
        start_time = time.time()
        response = requests.get(sci_hub_url, headers=get_headers(), timeout=TIMEOUT)
        elapsed = time.time() - start_time
        
        logger.debug(f"Sci-Hub请求完成，耗时: {elapsed:.2f}秒，状态码: {response.status_code}")
        
        if response.status_code == 200:
            logger.info("成功获取Sci-Hub页面")
            return response.text
        else:
            logger.warning(f"Sci-Hub返回状态码 {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        logger.error(f"Sci-Hub请求超时 ({TIMEOUT}秒)")
    except requests.exceptions.RequestException as e:
        logger.error(f"Sci-Hub请求错误: {e}")
    except Exception as e:
        logger.error(f"访问Sci-Hub时出错: {e}")
        logger.debug(traceback.format_exc())
    
    return None

def extract_download_link(html):
    """调试从Sci-Hub页面提取下载链接"""
    if not html:
        logger.warning("没有HTML内容可供解析")
        return None
        
    try:
        logger.debug("正在解析HTML...")
        soup = BeautifulSoup(html, "html.parser")
        
        # 查找嵌入标签
        logger.debug("正在搜索embed标签...")
        embed_tag = soup.find("embed", src=True)
        if embed_tag:
            link = embed_tag.get("src")
            logger.info(f"从embed标签找到链接: {link}")
            if link.startswith('/'):
                link = "https://sci-hub.se" + link
            return link
            
        # 查找iframe标签
        logger.debug("正在搜索iframe标签...")
        iframe_tag = soup.find("iframe", src=True)
        if iframe_tag:
            link = iframe_tag.get("src")
            logger.info(f"从iframe标签找到链接: {link}")
            if link.startswith('//'):
                link = "https:" + link
            elif link.startswith('/'):
                link = "https://sci-hub.se" + link
            return link
            
        # 查找a标签可能的PDF链接
        logger.debug("正在搜索a标签...")
        pdf_links = soup.find_all("a", href=lambda href: href and (href.endswith(".pdf") or "pdf" in href))
        if pdf_links:
            link = pdf_links[0].get("href")
            logger.info(f"从a标签找到链接: {link}")
            return link
            
        # 尝试找div#buttons中的内容
        logger.debug("正在搜索div#buttons...")
        buttons_div = soup.find("div", id="buttons")
        if buttons_div:
            logger.debug(f"找到buttons div: {buttons_div}")
            
        logger.warning("未找到任何下载链接")
        # 将整个HTML内容记录出来，以便于分析
        with open("debug_sci_hub_response.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.debug("已将完整HTML保存到debug_sci_hub_response.html")
        
        return None
    except Exception as e:
        logger.error(f"解析HTML时出错: {e}")
        logger.debug(traceback.format_exc())
        return None

def debug_paper_download(item, index, total):
    """测试单个论文下载流程"""
    logger.info(f"===== 正在调试论文 {index}/{total}: {item.get('title')} =====")
    
    title = item.get("title")
    if not title:
        logger.error("找不到论文标题，跳过")
        return False
        
    # 步骤1: 查找DOI
    logger.info("[步骤1] 查找DOI")
    doi = debug_get_doi(title)
    if not doi:
        logger.error("无法获取DOI，跳过后续步骤")
        return False
        
    # 步骤2: 获取Sci-Hub页面
    logger.info("[步骤2] 获取Sci-Hub页面")
    html = debug_sci_hub_page(doi)
    if not html:
        logger.error("无法获取Sci-Hub页面，跳过后续步骤")
        return False
        
    # 步骤3: 提取下载链接
    logger.info("[步骤3] 提取下载链接")
    download_link = extract_download_link(html)
    if not download_link:
        logger.error("无法提取下载链接，跳过后续步骤")
        return False
        
    # 步骤4: 获取文件（不实际下载，仅测试链接）
    logger.info("[步骤4] 测试下载链接")
    try:
        # 只发送HEAD请求，不下载实际内容
        logger.debug(f"正在测试链接: {download_link}")
        response = requests.head(download_link, headers=get_headers(), timeout=TIMEOUT)
        logger.info(f"链接测试结果: 状态码 {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', 'unknown')
            logger.info(f"内容类型: {content_type}, 文件大小: {content_length} 字节")
            return True
        else:
            logger.warning(f"下载链接返回非200状态码: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        logger.error(f"下载链接请求超时 ({TIMEOUT}秒)")
    except requests.exceptions.RequestException as e:
        logger.error(f"下载链接请求错误: {e}")
    except Exception as e:
        logger.error(f"测试下载链接时出错: {e}")
        logger.debug(traceback.format_exc())
        
    return False

def main():
    parser = argparse.ArgumentParser(description='调试下载器问题')
    parser.add_argument('--output-dir', required=True, help='包含download_queue.json的输出目录')
    parser.add_argument('--start', type=int, default=0, help='开始测试的项目索引')
    parser.add_argument('--count', type=int, default=5, help='要测试的项目数量')
    
    args = parser.parse_args()
    
    queue_file = os.path.join(args.output_dir, "download_queue.json")
    logger.info(f"使用队列文件: {queue_file}")
    
    queue_items = load_download_queue(queue_file)
    if not queue_items:
        logger.error("下载队列为空或无法加载")
        return
        
    # 只处理指定范围的项目
    start_idx = args.start
    end_idx = min(start_idx + args.count, len(queue_items))
    items_to_process = queue_items[start_idx:end_idx]
    
    logger.info(f"将测试 {len(items_to_process)} 个项目，从索引 {start_idx} 到 {end_idx-1}")
    
    # 依次测试每个项目
    results = []
    for i, item in enumerate(items_to_process):
        try:
            success = debug_paper_download(item, i+1, len(items_to_process))
            results.append({
                "index": start_idx + i,
                "title": item.get("title"),
                "success": success
            })
            # 简单的间隔，避免被封IP
            if i < len(items_to_process) - 1:
                time.sleep(3)
        except Exception as e:
            logger.error(f"处理项目时发生未捕获的异常: {e}")
            logger.debug(traceback.format_exc())
    
    # 输出汇总结果
    logger.info("\n===== 测试结果汇总 =====")
    successful = [r for r in results if r["success"]]
    logger.info(f"总共测试了 {len(results)} 个项目，成功: {len(successful)}, 失败: {len(results)-len(successful)}")
    
    if len(successful) < len(results):
        logger.info("\n失败的项目:")
        for r in results:
            if not r["success"]:
                logger.info(f"  - 索引 {r['index']}: {r['title']}")

if __name__ == "__main__":
    main()
