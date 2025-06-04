#!/usr/bin/env python3
# retry_failed_downloads.py - Script to retry downloading papers that failed initially

import argparse
import logging
import json
import os
import time
from pathlib import Path
from src.paper_downloader import retry_failed_downloads

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Retry downloading papers that failed previously")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory containing results files")
    parser.add_argument("--download-dir", "-d", help="Directory to save downloaded papers (defaults to output-dir/downloads)")
    parser.add_argument("--delay", "-t", type=float, default=5, help="Delay between downloads in seconds (default: 5)")
    parser.add_argument("--max-retries", "-r", type=int, default=3, help="Maximum retry attempts per paper (default: 3)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir) if args.download_dir else output_dir / "downloads"
    
    # Ensure directories exist
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to download results file
    download_results_file = output_dir / "download_results.json"
    
    if not download_results_file.exists():
        logger.error(f"Download results file not found: {download_results_file}")
        return
    
    # Load download results
    with open(download_results_file, 'r', encoding='utf-8') as f:
        download_results = json.load(f)
    
    # Filter failed downloads
    failed_downloads = [item for item in download_results if item.get('download_success') is False]
    
    if not failed_downloads:
        logger.info("No failed downloads found to retry")
        return
    
    logger.info(f"Found {len(failed_downloads)} failed downloads to retry")
    
    # Retry downloads
    still_failed, successful = retry_failed_downloads(
        failed_downloads,
        str(download_dir),
        delay=args.delay,
        max_retries=args.max_retries
    )
    
    # Update download results with retry information
    for item in download_results:
        # For each item that was failed but now succeeded, update its status
        for success in successful:
            if (item.get('download_success') is False and 
                item.get('doi') == success.get('doi')):
                item.update({
                    'download_success': True,
                    'download_status': 'complete',
                    'local_path': success.get('local_path'),
                    'retry_count': success.get('retry_count'),
                    'retry_time': success.get('retry_time'),
                    'retry_success': True
                })
    
    # Save updated results
    with open(download_results_file, 'w', encoding='utf-8') as f:
        json.dump(download_results, f, indent=2, ensure_ascii=False)
    
    # Save still failed downloads to a separate file for further analysis
    if still_failed:
        still_failed_file = output_dir / "still_failed_downloads.json"
        with open(still_failed_file, 'w', encoding='utf-8') as f:
            json.dump(still_failed, f, indent=2, ensure_ascii=False)
        
        # Also save just the DOIs
        still_failed_dois_file = output_dir / "still_failed_dois.txt"
        with open(still_failed_dois_file, 'w', encoding='utf-8') as f:
            for item in still_failed:
                if item.get('doi'):
                    f.write(f"{item['doi']}\n")
    
    # Print summary
    success_count = len(successful)
    logger.info("=" * 50)
    logger.info(f"Retry complete: {success_count} out of {len(failed_downloads)} papers successfully downloaded")
    logger.info(f"Still failed: {len(still_failed)}")
    if still_failed:
        logger.info(f"Failed papers list saved to: {still_failed_file}")
        logger.info(f"Failed DOIs saved to: {still_failed_dois_file}")
    logger.info(f"Updated download results saved to: {download_results_file}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
