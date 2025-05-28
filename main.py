#!/usr/bin/env python3
# main.py - Combined entry point with options for parsing or downloading

import argparse
import logging
import sys
import os
from pathlib import Path

# Import modules
import parser
import downloader
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR

def setup_logging():
    """Set up logging with console output only."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

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
    
    # Define top-level command line arguments
    parser = argparse.ArgumentParser(description='Academic Paper Processing Tool')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Parse mode arguments
    parse_parser = subparsers.add_parser('parse', help='Parse papers using LLM')
    parse_parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR, help='Directory containing input PDF files')
    parse_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory to store output files')
    parse_parser.add_argument('--api-base', required=True, help='LLM API base URL')
    parse_parser.add_argument('--api-key', default="EMPTY", help='API key for the LLM service')
    parse_parser.add_argument('--model', required=True, help='LLM model name')
    parse_parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parse_parser.add_argument('--max-depth', type=int, default=-1, help='Maximum recursion depth (-1 for unlimited)')
    parse_parser.add_argument('--text-ratio', default=None, help='Text extraction ratio as "head,tail" (e.g., "10,30")')
    parse_parser.add_argument('--temperature', type=float, help='LLM temperature parameter')
    parse_parser.add_argument('--top-p', type=float, help='LLM top-p parameter')
    parse_parser.add_argument('--top-k', type=int, help='LLM top-k parameter')
    parse_parser.add_argument('--presence-penalty', type=float, help='LLM presence penalty parameter')
    parse_parser.add_argument('--max-tokens', type=int, help='LLM max tokens parameter')
    
    # Download mode arguments
    download_parser = subparsers.add_parser('download', help='Download papers from queue')
    download_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Base directory for all outputs')
    download_parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel downloads')
    download_parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed downloads')
    download_parser.add_argument('--delay', type=float, default=2.0, help='Delay between downloads in seconds')
    
    args = parser.parse_args()
    
    if args.mode == 'parse':
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
        
        # Run parser
        try:
            paper_parser = parser.PaperParser(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                api_base=api_base,
                api_key=args.api_key,
                model=args.model,
                max_workers=args.max_workers,
                max_depth=args.max_depth,  # 添加最大深度参数
                text_ratio=text_ratio,
                **llm_params
            )
            
            logger.info(f"Starting paper parsing with max depth: {args.max_depth}...")
            paper_parser.process_all_papers()
            
        except KeyboardInterrupt:
            logger.info("Parser stopped by user")
        except Exception as e:
            logger.error(f"Parser stopped due to error: {e}", exc_info=True)
    
    elif args.mode == 'download':
        # Run downloader
        try:
            paper_downloader = downloader.PaperDownloader(
                output_dir=args.output_dir,
                max_workers=args.max_workers,
                retry_failed=args.retry_failed,
                delay_between=args.delay
            )
            
            logger.info("Starting paper downloads...")
            paper_downloader.download_all_papers()
            
        except KeyboardInterrupt:
            logger.info("Downloader stopped by user")
        except Exception as e:
            logger.error(f"Downloader stopped due to error: {e}", exc_info=True)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
