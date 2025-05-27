#!/usr/bin/env python3
# main.py - Academic Paper Reference Crawler main entry point

import argparse
import logging
import sys
import os
import time
from pathlib import Path

# Import from src directory
from src.paper_crawler import PaperCrawler

# Configure logging with proper file handling
def setup_logging():
    """Set up logging with both file and console output."""
    log_file = os.path.abspath("paper_crawler.log")
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=False),
            logging.StreamHandler()
        ]
    )
    
    # Ensure the file handler flushes properly
    file_handler = next((h for h in logging.getLogger().handlers 
                        if isinstance(h, logging.FileHandler)), None)
    if file_handler:
        file_handler.flush()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Validate and fix API URL if needed
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
    # Set up logging
    logger = setup_logging()
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Academic Paper Reference Crawler')
    parser.add_argument('--input-dir', required=True, help='Directory containing input PDF files')
    parser.add_argument('--output-dir', default='./output', help='Directory to store output files')
    parser.add_argument('--api-base', required=True, help='LLM API base URL')
    parser.add_argument('--api-key', default="EMPTY", help='API key for the LLM service')
    parser.add_argument('--model', required=True, help='LLM model name')
    parser.add_argument('--max-depth', type=int, default=-1, help='Maximum recursion depth (-1 for unlimited)')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of parallel workers')
    parser.add_argument('--text-ratio', default=None, help='Text extraction ratio as "head,tail" (e.g., "10,30" for first 10% and last 30%)')
    parser.add_argument('--temperature', type=float, help='LLM temperature parameter')
    parser.add_argument('--top-p', type=float, help='LLM top-p parameter')
    parser.add_argument('--top-k', type=int, help='LLM top-k parameter')
    parser.add_argument('--presence-penalty', type=float, help='LLM presence penalty parameter')
    parser.add_argument('--max-tokens', type=int, help='LLM max tokens parameter')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Fix API URL if needed
    api_base = validate_api_url(args.api_base)
    if api_base != args.api_base:
        logger.info(f"Fixed API base URL: {api_base}")
    
    # Parse text ratio parameter
    text_ratio = parse_text_ratio(args.text_ratio)
    if args.text_ratio and text_ratio is None:
        logger.warning(f"Invalid text ratio '{args.text_ratio}'. Format should be 'head,tail' (e.g., '10,30'). Using full text instead.")
    elif text_ratio:
        head, tail = text_ratio
        logger.info(f"Using text extraction ratio: first {head}% and last {tail}% of papers")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display startup information
    logger.info("=" * 50)
    logger.info("RecursiveScholarCrawler")
    logger.info("=" * 50)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"API Base: {api_base}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max recursion depth: {args.max_depth}")
    logger.info(f"Max workers: {args.max_workers}")
    if text_ratio:
        logger.info(f"Text extraction ratio: {text_ratio[0]}% head, {text_ratio[1]}% tail")
    else:
        logger.info("Text extraction: full text")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Initialize and run crawler
    try:
        crawler = PaperCrawler(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            api_base=api_base,
            api_key=args.api_key,
            model=args.model,
            max_recursion_depth=args.max_depth,
            max_workers=args.max_workers,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            text_ratio=text_ratio
        )
        
        logger.info("Starting crawler...")
        crawler.crawl()
        
    except KeyboardInterrupt:
        logger.info("Crawler stopped by user")
    except Exception as e:
        logger.error(f"Crawler stopped due to error: {e}", exc_info=True)
    finally:
        # Calculate total runtime
        end_time = time.time()
        runtime = end_time - start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        logger.info("=" * 50)
        logger.info(f"Crawler finished. Total runtime: {runtime_str}")
        logger.info("=" * 50)
        
        # Ensure all logs are flushed to disk
        for handler in logging.getLogger().handlers:
            handler.flush()
        logging.shutdown()

if __name__ == "__main__":
    main()
