# recover_queue.py
import os
import json
import argparse
import logging
from pathlib import Path

# 设置简单的日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recover_download_queue(results_file, queue_file, force_overwrite=False):
    """
    Recovers the download queue from the download results log.
    
    This script reads 'download_results.json' and rebuilds 'download_queue.json'
    with all items that were not successfully downloaded.

    Args:
        results_file (str): Path to the download_results.json file.
        queue_file (str): Path to the download_queue.json file to be created/updated.
        force_overwrite (bool): If True, it will completely overwrite the existing queue file.
                                If False, it will only add missing failed items.
    """
    # 1. 检查源文件是否存在
    if not os.path.exists(results_file):
        logging.error(f"Source file not found: {results_file}")
        logging.error("Cannot recover queue. Please run the main parser first to generate a results file.")
        return

    # 2. 读取 download_results.json
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            download_results = json.load(f)
        logging.info(f"Successfully loaded {len(download_results)} items from {results_file}.")
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error reading {results_file}: {e}")
        return

    # 3. 识别所有未成功的下载项
    #    "download_success" is not True 包括了 False, None, 或者键不存在的情况
    failed_items = [item for item in download_results if item.get('download_success') is not True]
    
    if not failed_items:
        logging.info("No failed or incomplete downloads found in results. The queue does not need recovery.")
        return
    
    logging.info(f"Found {len(failed_items)} items to potentially re-queue.")

    # 4. 加载现有的下载队列（如果存在）
    existing_queue = []
    if os.path.exists(queue_file) and not force_overwrite:
        try:
            with open(queue_file, 'r', encoding='utf-8') as f:
                existing_queue = json.load(f)
            logging.info(f"Loaded {len(existing_queue)} items from existing queue file {queue_file}.")
        except (json.JSONDecodeError, IOError):
            logging.warning(f"Could not read existing queue file {queue_file}. It will be overwritten.")
            existing_queue = []
    
    # 5. 合并或覆盖队列
    if force_overwrite:
        final_queue = failed_items
        logging.info("Forcing overwrite. The new queue will contain only failed items from the results.")
    else:
        # 只添加不存在于现有队列中的失败项
        # 使用 DOI 或 Title 作为唯一标识符
        existing_identifiers = {item.get('doi') or item.get('title') for item in existing_queue}
        items_to_add = [
            item for item in failed_items 
            if (item.get('doi') or item.get('title')) not in existing_identifiers
        ]
        
        final_queue = existing_queue + items_to_add
        logging.info(f"Adding {len(items_to_add)} missing failed items to the existing queue.")

    # 6. 保存新的 download_queue.json
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(queue_file), exist_ok=True)
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(final_queue, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully recovered and saved {len(final_queue)} items to {queue_file}.")
    except IOError as e:
        logging.error(f"Failed to write to {queue_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recovers 'download_queue.json' from 'download_results.json'.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--output-dir', 
        required=True, 
        help="The main output directory where state files (e.g., download_results.json) are stored."
    )
    parser.add_argument(
        '--subdir',
        default=None,
        help="Optional subdirectory within the output directory, if you used one in the main parser."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force overwrite of 'download_queue.json'. By default, it only appends missing items."
    )
    
    args = parser.parse_args()
    
    # 构建文件路径
    base_output_dir = Path(args.output_dir)
    if args.subdir:
        actual_output_dir = base_output_dir / args.subdir
    else:
        actual_output_dir = base_output_dir
        
    results_file = actual_output_dir / "download_results.json"
    queue_file = actual_output_dir / "download_queue.json"
    
    logging.info(f"Attempting to recover queue for output directory: {actual_output_dir}")
    logging.info(f"Source (results) file: {results_file}")
    logging.info(f"Target (queue) file:   {queue_file}")
    
    recover_download_queue(str(results_file), str(queue_file), args.force)

if __name__ == "__main__":
    main()