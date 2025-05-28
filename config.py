# config.py - 动态设置路径配置

import os
from pathlib import Path

# 用于存储实际路径的变量 - 将从命令行参数设置
INPUT_DIR = None
OUTPUT_DIR = None
DOWNLOAD_DIR = None

# 状态文件路径 - 将在设置输出目录时更新
PAPER_DETAILS_FILE = None
STATS_FILE = None
DOWNLOAD_QUEUE_FILE = None
DOWNLOAD_RESULTS_FILE = None
PROGRESS_FILE = None
PROCESSING_STATE_FILE = None

# 初始化状态 - 尚未配置
PATHS_CONFIGURED = False

def configure_paths(input_dir, output_dir):
    """配置所有路径 - 必须在使用任何路径前调用"""
    global INPUT_DIR, OUTPUT_DIR, DOWNLOAD_DIR
    global PAPER_DETAILS_FILE, STATS_FILE, DOWNLOAD_QUEUE_FILE, DOWNLOAD_RESULTS_FILE
    global PROGRESS_FILE, PROCESSING_STATE_FILE, PATHS_CONFIGURED
    
    # 设置主要目录
    INPUT_DIR = input_dir
    OUTPUT_DIR = output_dir
    DOWNLOAD_DIR = str(Path(output_dir) / "downloads")
    
    # 确保目录存在
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # 设置文件路径
    PAPER_DETAILS_FILE = str(Path(OUTPUT_DIR) / "paper_details.json")
    STATS_FILE = str(Path(OUTPUT_DIR) / "crawler_stats.json")
    DOWNLOAD_QUEUE_FILE = str(Path(OUTPUT_DIR) / "download_queue.json")
    DOWNLOAD_RESULTS_FILE = str(Path(OUTPUT_DIR) / "download_results.json")
    PROGRESS_FILE = str(Path(OUTPUT_DIR) / "processing_progress.json")
    PROCESSING_STATE_FILE = str(Path(OUTPUT_DIR) / "processing_state.json")
    
    # 标记为已配置
    PATHS_CONFIGURED = True
    
    return {
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "download_dir": DOWNLOAD_DIR,
        "paper_details_file": PAPER_DETAILS_FILE,
        "stats_file": STATS_FILE,
        "download_queue_file": DOWNLOAD_QUEUE_FILE,
        "download_results_file": DOWNLOAD_RESULTS_FILE,
        "progress_file": PROGRESS_FILE,
        "processing_state_file": PROCESSING_STATE_FILE
    }

# 确保路径已配置的装饰器
def requires_configured_paths(func):
    """装饰器，确保在访问路径前已配置路径"""
    def wrapper(*args, **kwargs):
        if not PATHS_CONFIGURED:
            raise RuntimeError("必须先调用 configure_paths() 设置路径后才能访问")
        return func(*args, **kwargs)
    return wrapper

# 检查路径配置状态
@requires_configured_paths
def get_configured_paths():
    """获取当前配置的所有路径"""
    return {
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "download_dir": DOWNLOAD_DIR,
        "paper_details_file": PAPER_DETAILS_FILE,
        "stats_file": STATS_FILE,
        "download_queue_file": DOWNLOAD_QUEUE_FILE,
        "download_results_file": DOWNLOAD_RESULTS_FILE,
        "progress_file": PROGRESS_FILE,
        "processing_state_file": PROCESSING_STATE_FILE
    }

# 默认LLM提示词保持不变
DEFAULT_PROMPT = """
Extract the following information from this academic paper as JSON:

- title: The paper's full title
- authors: List of authors' names
- affiliations: List of author affiliations
- abstract: The paper's abstract
- references: List of references with the following details for each reference:
  - ref_title: Full title of the referenced paper (IMPORTANT: provide complete titles)
  - ref_authors: List of authors of the referenced paper
  - ref_year: Publication year
  - ref_venue: Publication venue (journal/conference)

IMPORTANT INSTRUCTIONS:
1. Pay special attention to the References or Bibliography section at the end of the paper
2. Extract ALL references you can find, even if the paper text is truncated
3. For each reference, ensure you capture the complete title - this is crucial
4. Return ONLY valid JSON with no additional text or formatting
5. If you cannot find certain information, use empty strings or arrays rather than omitting fields

PAPER TEXT:
{paper_text}
"""
