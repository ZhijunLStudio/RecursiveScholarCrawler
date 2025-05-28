#!/usr/bin/env python3
# parser.py - Paper parsing functionality using LLM

import os
import sys
import argparse
import logging
import time
import signal
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime

# 导入但不立即使用配置
import config
from utils import (load_json, save_json, get_timestamp_str, extract_text_with_limit, 
                 extract_text_by_ratio, Locker)
from llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperParser:
    def __init__(self, input_dir, output_dir, api_base, api_key, model,
                 max_workers=4, max_depth=-1, text_ratio=None, save_interval=30, **llm_params):
        """Initialize the paper parser."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM parameters
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.llm_params = llm_params
        self.text_ratio = text_ratio
        
        # Processing settings
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.save_interval = save_interval
        
        # Initialize LLM service
        self.llm_service = LLMService(
            api_key=api_key,
            api_base=api_base,
            model=model,
            **llm_params
        )
        
        # Initialize state
        self.load_state()
        self.lock = threading.Lock()
        
        # 设置自动保存定时器
        self.last_save_time = time.time()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 当前正在处理的文件
        self.current_processing = {}
        
    def signal_handler(self, sig, frame):
        """处理中断信号，保存状态后退出"""
        logger.info("接收到中断信号，保存状态后退出...")
        self.save_state()
        self.save_processing_state()
        sys.exit(0)
        
    def load_state(self):
        """Load processing state from files."""
        # 从配置获取路径
        paths = config.get_configured_paths()
        
        # Load paper details
        self.paper_details = load_json(paths["paper_details_file"], {})
        
        # Load or initialize stats
        self.stats = load_json(paths["stats_file"], {
            "processed_count": 0,
            "found_references": 0,
            "start_time": get_timestamp_str(),
            "last_update": get_timestamp_str(),
            "status": "initializing",
            "max_depth": self.max_depth
        })
        
        # Load or initialize progress tracker
        self.progress = load_json(paths["progress_file"], {
            "processed_files": [],
            "failed_files": [],
            "pending_files": [],
            "in_progress_files": [],
            "total_files": 0
        })
        
        # Initialize download queue
        self.download_queue = load_json(paths["download_queue_file"], [])
        
        # 加载中间处理状态
        self.load_processing_state()
        
    def load_processing_state(self):
        """加载中断时的处理状态"""
        paths = config.get_configured_paths()
        processing_state = load_json(paths["processing_state_file"], {
            "in_progress": {},
            "last_update": get_timestamp_str()
        })
        
        self.current_processing = processing_state.get("in_progress", {})
        
        # 检查是否有上次未完成的处理
        if self.current_processing:
            logger.info(f"检测到上次有未完成的处理: {len(self.current_processing)} 个文件")
            # 将未完成的文件重新加入待处理队列
            for file_path, state in self.current_processing.items():
                if file_path not in self.progress["processed_files"] and file_path not in self.progress["failed_files"]:
                    if file_path not in self.progress["pending_files"]:
                        self.progress["pending_files"].append(file_path)
                    if file_path not in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].append(file_path)
                    logger.info(f"将未完成的文件重新加入队列: {file_path} (深度: {state.get('depth', 0)})")
    
    def save_processing_state(self):
        """保存当前处理状态"""
        paths = config.get_configured_paths()
        processing_state = {
            "in_progress": self.current_processing,
            "last_update": get_timestamp_str()
        }
        save_json(processing_state, paths["processing_state_file"])
        
    def save_state(self):
        """Save current state to files."""
        paths = config.get_configured_paths()
        with self.lock:
            save_json(self.paper_details, paths["paper_details_file"])
            save_json(self.stats, paths["stats_file"])
            save_json(self.progress, paths["progress_file"])
            save_json(self.download_queue, paths["download_queue_file"])
            self.save_processing_state()
    
    def check_autosave(self):
        """检查是否需要自动保存"""
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            logger.info(f"自动保存状态 (间隔: {self.save_interval}秒)")
            self.save_state()
            self.last_save_time = current_time
    
    def update_stats(self, key, increment=1):
        """Update a stats counter."""
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + increment
            self.stats["last_update"] = get_timestamp_str()
    
    def scan_input_directory(self):
        """Scan input directory for PDF files and update pending list."""
        pdf_files = list(self.input_dir.glob("**/*.pdf"))
        
        # First pass: get all file paths
        all_files = set(str(path) for path in pdf_files)
        processed = set(self.progress["processed_files"])
        failed = set(self.progress["failed_files"])
        
        # Identify new or pending files
        pending = list(all_files - processed - failed)
        
        # 检查上次中断的文件
        in_progress = [path for path in pending if path in self.current_processing]
        if in_progress:
            logger.info(f"发现 {len(in_progress)} 个上次中断的文件, 将优先处理")
        
        # Update progress tracker
        self.progress["pending_files"] = pending
        self.progress["in_progress_files"] = in_progress
        self.progress["total_files"] = len(all_files)
        
        logger.info(f"Found {len(pending)} new/pending PDF files out of {len(all_files)} total files")
        return in_progress + [p for p in pending if p not in in_progress]
    
    def process_paper(self, pdf_path, current_depth=0):
        """Process a single paper using LLM and update state."""
        pdf_path = str(pdf_path)
        paper_id = os.path.basename(pdf_path)
        
        # 设置当前正在处理的文件状态
        with self.lock:
            self.current_processing[pdf_path] = {
                "start_time": get_timestamp_str(),
                "depth": current_depth,
                "status": "processing",
                "paper_id": paper_id
            }
            
            if pdf_path not in self.progress["in_progress_files"]:
                self.progress["in_progress_files"].append(pdf_path)
                
            self.save_processing_state()
        
        # Check if already processed
        if paper_id in self.paper_details:
            logger.info(f"Paper already processed: {paper_id}")
            
            # 清理处理状态
            with self.lock:
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                self.save_processing_state()
                
            return self.paper_details[paper_id]
        
        logger.info(f"[Depth {current_depth}] Processing paper: {paper_id}")
        
        # Check if we've reached max depth
        if self.max_depth != -1 and current_depth > self.max_depth:
            logger.info(f"Reached maximum depth ({self.max_depth}) for {paper_id}, skipping further processing")
            
            with self.lock:
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                
                self.progress["processed_files"].append(pdf_path)
                if pdf_path in self.progress["pending_files"]:
                    self.progress["pending_files"].remove(pdf_path)
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                    
            self.save_state()
            return {"depth_limit_reached": True, "path": pdf_path}
        
        try:
            # 更新处理状态
            with self.lock:
                self.current_processing[pdf_path]["status"] = "extracting_text"
                self.save_processing_state()
            
            # Extract text from PDF
            if self.text_ratio:
                head_ratio, tail_ratio = self.text_ratio
                paper_text = extract_text_by_ratio(pdf_path, head_ratio, tail_ratio)
                logger.info(f"[Depth {current_depth}] Extracted text using ratio {head_ratio}%,{tail_ratio}% from {paper_id}")
            else:
                paper_text = extract_text_with_limit(pdf_path)
                logger.info(f"[Depth {current_depth}] Extracted full text from {paper_id}")
            
            if not paper_text:
                logger.error(f"[Depth {current_depth}] Failed to extract text from {paper_id}")
                
                with self.lock:
                    self.progress["failed_files"].append(pdf_path)
                    if pdf_path in self.progress["pending_files"]:
                        self.progress["pending_files"].remove(pdf_path)
                    if pdf_path in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].remove(pdf_path)
                    if pdf_path in self.current_processing:
                        del self.current_processing[pdf_path]
                        
                self.save_state()
                return {"error": "Text extraction failed", "path": pdf_path}
            
            # 更新处理状态
            with self.lock:
                self.current_processing[pdf_path]["status"] = "querying_llm"
                self.current_processing[pdf_path]["text_extraction_time"] = get_timestamp_str()
                self.save_processing_state()
            
            # 检查是否需要自动保存
            self.check_autosave()
            
            # Query LLM to extract information
            paper_info = self.llm_service.extract_paper_info(paper_text)
            
            # Check for LLM failure
            if not paper_info or (paper_info.get("llm_usage", {}).get("success") is False and len(paper_info) == 1):
                logger.error(f"[Depth {current_depth}] LLM extraction failed for {paper_id}")
                
                with self.lock:
                    self.progress["failed_files"].append(pdf_path)
                    if pdf_path in self.progress["pending_files"]:
                        self.progress["pending_files"].remove(pdf_path)
                    if pdf_path in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].remove(pdf_path)
                    if pdf_path in self.current_processing:
                        del self.current_processing[pdf_path]
                        
                self.save_state()
                return {"error": "LLM extraction failed", "path": pdf_path}
            
            # 更新处理状态
            with self.lock:
                self.current_processing[pdf_path]["status"] = "processing_references"
                self.current_processing[pdf_path]["llm_completion_time"] = get_timestamp_str()
                self.save_processing_state()
            
            # Add metadata
            paper_info["source_pdf"] = pdf_path
            paper_info["processing_timestamp"] = get_timestamp_str()
            paper_info["processing_depth"] = current_depth
            
            # 检查是否需要自动保存
            self.check_autosave()
            
            # Process references for download queue
            references = paper_info.get("references", [])
            
            for ref in references:
                # Add to download queue if title exists and depth is not exceeded
                if ref.get("ref_title"):
                    should_queue = True
                    if self.max_depth != -1 and current_depth + 1 > self.max_depth:
                        ref["depth_limit_reached"] = True
                        should_queue = False
                    
                    if should_queue:
                        with self.lock:
                            # Check if this reference is already in the queue
                            title = ref.get("ref_title")
                            exists = any(item["title"] == title for item in self.download_queue)
                            
                            if not exists:
                                self.download_queue.append({
                                    "title": title,
                                    "authors": ref.get("ref_authors", []),
                                    "year": ref.get("ref_year", ""),
                                    "venue": ref.get("ref_venue", ""),
                                    "source_paper": paper_id,
                                    "source_depth": current_depth,
                                    "target_depth": current_depth + 1,
                                    "queued_at": get_timestamp_str(),
                                    "status": "pending"
                                })
            
            # Update statistics
            self.update_stats("processed_count")
            self.update_stats("found_references", len(references))
            
            # Add to paper details and update progress
            with self.lock:
                self.paper_details[paper_id] = paper_info
                self.progress["processed_files"].append(pdf_path)
                if pdf_path in self.progress["pending_files"]:
                    self.progress["pending_files"].remove(pdf_path)
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
            
            # Save state
            self.save_state()
            return paper_info
            
        except Exception as e:
            logger.error(f"[Depth {current_depth}] Error processing {paper_id}: {e}", exc_info=True)
            
            with self.lock:
                self.progress["failed_files"].append(pdf_path)
                if pdf_path in self.progress["pending_files"]:
                    self.progress["pending_files"].remove(pdf_path)
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                    
            self.save_state()
            return {"error": str(e), "path": pdf_path}
    
    def process_all_papers(self):
        """Process all pending papers in parallel."""
        pending_files = self.scan_input_directory()
        
        if not pending_files:
            logger.info("No new papers to process")
            return
        
        logger.info(f"Starting processing of {len(pending_files)} papers with {self.max_workers} workers")
        self.stats["status"] = "processing"
        self.stats["max_depth"] = self.max_depth
        self.save_state()
        
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_paper, pdf_path, 0) for pdf_path in pending_files]
                
                # Process as they complete
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        logger.info(f"Completed {i+1}/{len(futures)}: {result.get('title', 'Unknown title')}")
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                
            # Update final stats
            end_time = time.time()
            runtime = end_time - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            self.stats["status"] = "completed"
            self.stats["end_time"] = get_timestamp_str()
            self.stats["runtime"] = runtime_str
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Saving state.")
            self.stats["status"] = "interrupted"
            self.stats["end_time"] = get_timestamp_str()
        finally:
            # Save final state
            self.save_state()
            
            # Print summary
            logger.info("=" * 50)
            logger.info(f"Processing complete. Processed {self.stats.get('processed_count', 0)} papers.")
            logger.info(f"Found {self.stats.get('found_references', 0)} references.")
            logger.info(f"Added {len(self.download_queue)} items to download queue.")
            logger.info("=" * 50)

def validate_api_url(url):
    """Validate and fix API URL if needed."""
    if not url.startswith('http'):
        return 'https://' + url.lstrip('://')
    return url

def parse_text_ratio(ratio_str):
    """Parse the text extraction ratio parameter."""
    if not ratio_str:
        return None
    
    try:
        head, tail = map(int, ratio_str.split(','))
        if head < 0 or tail < 0 or head + tail > 100:
            return None
        return (head, tail)
    except ValueError:
        return None

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Academic Paper Parser using LLM')
    parser.add_argument('--input-dir', required=True, help='Directory containing input PDF files')
    parser.add_argument('--output-dir', required=True, help='Directory to store output files')
    parser.add_argument('--api-base', required=True, help='LLM API base URL')
    parser.add_argument('--api-key', default="EMPTY", help='API key for the LLM service')
    parser.add_argument('--model', required=True, help='LLM model name')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--max-depth', type=int, default=-1, help='Maximum recursion depth (-1 for unlimited)')
    parser.add_argument('--text-ratio', default=None, help='Text extraction ratio as "head,tail" (e.g., "10,30")')
    parser.add_argument('--save-interval', type=int, default=30, help='Auto-save interval in seconds')
    parser.add_argument('--temperature', type=float, help='LLM temperature parameter')
    parser.add_argument('--top-p', type=float, help='LLM top-p parameter')
    parser.add_argument('--top-k', type=int, help='LLM top-k parameter')
    parser.add_argument('--presence-penalty', type=float, help='LLM presence penalty parameter')
    parser.add_argument('--max-tokens', type=int, help='LLM max tokens parameter')
    
    args = parser.parse_args()
    
    # 最重要：先配置路径，这要在任何其他操作之前
    config.configure_paths(args.input_dir, args.output_dir)
    paths = config.get_configured_paths()
    logger.info(f"配置路径: 输入={paths['input_dir']}, 输出={paths['output_dir']}")
    
    # Fix API URL if needed
    api_base = validate_api_url(args.api_base)
    
    # Parse text ratio parameter
    text_ratio = parse_text_ratio(args.text_ratio)
    if args.text_ratio and text_ratio is None:
        logger.warning(f"Invalid text ratio '{args.text_ratio}'. Format should be 'head,tail' (e.g., '10,30'). Using full text instead.")
    
    # Collect LLM parameters
    llm_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'presence_penalty': args.presence_penalty,
        'max_tokens': args.max_tokens
    }
    # Remove None values
    llm_params = {k: v for k, v in llm_params.items() if v is not None}
    
    # Initialize and run parser
    try:
        paper_parser = PaperParser(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            api_base=api_base,
            api_key=args.api_key,
            model=args.model,
            max_workers=args.max_workers,
            max_depth=args.max_depth,
            text_ratio=text_ratio,
            save_interval=args.save_interval,
            **llm_params
        )
        
        logger.info(f"Starting paper parsing with max depth: {args.max_depth}...")
        paper_parser.process_all_papers()
        
    except KeyboardInterrupt:
        logger.info("Parser stopped by user")
    except Exception as e:
        logger.error(f"Parser stopped due to error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
