#!/usr/bin/env python3
# downloader.py - Paper downloading functionality

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import random

# 导入但不立即使用配置
import config
from utils import load_json, save_json, get_timestamp_str, Locker
from doi_helper import download as doi_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperDownloader:
    def __init__(self, output_dir, max_workers=4, retry_failed=False, delay_between=2):
        """Initialize the paper downloader."""
        self.output_dir = Path(output_dir)
        # 下载目录从配置获取
        paths = config.get_configured_paths()
        self.download_dir = Path(paths["download_dir"])
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing settings
        self.max_workers = max_workers
        self.retry_failed = retry_failed
        self.delay_between = delay_between
        
        # Load state
        self.download_queue = load_json(paths["download_queue_file"], [])
        self.download_results = load_json(paths["download_results_file"], {})
        
        self.lock = threading.Lock()
        
    def save_state(self):
        """Save current state to files."""
        paths = config.get_configured_paths()
        with self.lock:
            save_json(self.download_queue, paths["download_queue_file"])
            save_json(self.download_results, paths["download_results_file"])
    
    def get_pending_downloads(self):
        """Get list of pending downloads."""
        with self.lock:
            # If retry_failed is True, include failed downloads
            if self.retry_failed:
                # All items that don't have a successful result
                pending = [item for item in self.download_queue 
                          if item["title"] not in self.download_results or 
                          not self.download_results[item["title"]].get("success")]
            else:
                # Only items with status "pending"
                pending = [item for item in self.download_queue if item["status"] == "pending"]
                
            return pending
    
    def download_paper(self, paper_item):
        """Download a single paper and update results."""
        title = paper_item["title"]
        
        # Mark as in-progress
        with self.lock:
            for item in self.download_queue:
                if item["title"] == title:
                    item["status"] = "downloading"
                    item["download_attempt_time"] = get_timestamp_str()
            self.save_state()
        
        logger.info(f"Downloading: {title}")
        
        try:
            # Use doi_helper to download the paper
            result = doi_download(title, output_directory=str(self.download_dir))
            
            # Update download results
            with self.lock:
                self.download_results[title] = {
                    "success": result.get("success", False),
                    "status": result.get("status", "unknown"),
                    "file_path": result.get("file_path", ""),
                    "doi": result.get("doi", ""),
                    "timestamp": get_timestamp_str(),
                    "original_request": paper_item
                }
                
                # Update queue item status
                for item in self.download_queue:
                    if item["title"] == title:
                        item["status"] = "completed" if result.get("success") else "failed"
                        item["download_result"] = result.get("status", "unknown")
                
                self.save_state()
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading '{title}': {e}", exc_info=True)
            
            # Update on failure
            with self.lock:
                self.download_results[title] = {
                    "success": False,
                    "status": f"error: {str(e)}",
                    "timestamp": get_timestamp_str(),
                    "original_request": paper_item
                }
                
                # Update queue item status
                for item in self.download_queue:
                    if item["title"] == title:
                        item["status"] = "failed"
                        item["download_result"] = f"error: {str(e)}"
                
                self.save_state()
            
            return {"success": False, "status": f"error: {str(e)}"}
    
    def download_all_papers(self):
        """Download all pending papers in parallel."""
        pending_downloads = self.get_pending_downloads()
        
        if not pending_downloads:
            logger.info("No papers to download")
            return
        
        logger.info(f"Starting download of {len(pending_downloads)} papers with {self.max_workers} workers")
        
        # Statistics
        start_time = time.time()
        success_count = 0
        fail_count = 0
        
        try:
            # We'll use our own loop instead of ThreadPoolExecutor.map to control timing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                # Submit all tasks
                for item in pending_downloads:
                    futures.append(executor.submit(self.download_paper, item))
                    # Add a small delay to avoid hammering servers
                    time.sleep(self.delay_between)
                
                # Process results as they complete
                for future in futures:
                    try:
                        result = future.result()
                        if result.get("success"):
                            success_count += 1
                            logger.info(f"Successfully downloaded: {result.get('file_path', 'unknown')}")
                        else:
                            fail_count += 1
                            logger.warning(f"Download failed: {result.get('status', 'unknown error')}")
                    except Exception as e:
                        fail_count += 1
                        logger.error(f"Download worker failed: {e}")
            
            # Final stats
            end_time = time.time()
            runtime = end_time - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            logger.info("=" * 50)
            logger.info(f"Download complete. Runtime: {runtime_str}")
            logger.info(f"Success: {success_count}, Failed: {fail_count}")
            logger.info("=" * 50)
            
        except KeyboardInterrupt:
            logger.info("Download interrupted by user. Saving state.")
        
        # Final save
        self.save_state()

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Academic Paper Downloader')
    parser.add_argument('--output-dir', required=True, help='Directory to store output files')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel downloads')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed downloads')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between downloads in seconds')
    
    args = parser.parse_args()
    
    # 最重要：先配置路径，这要在任何其他操作之前
    # 设置空的输入目录，因为下载器只需要输出目录
    config.configure_paths("", args.output_dir)
    paths = config.get_configured_paths()
    logger.info(f"配置下载路径: 输出={paths['output_dir']}, 下载={paths['download_dir']}")
    
    # Initialize and run downloader
    try:
        downloader = PaperDownloader(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            retry_failed=args.retry_failed,
            delay_between=args.delay
        )
        
        logger.info("Starting paper downloads...")
        downloader.download_all_papers()
        
    except KeyboardInterrupt:
        logger.info("Downloader stopped by user")
    except Exception as e:
        logger.error(f"Downloader stopped due to error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
