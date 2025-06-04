#!/usr/bin/env python3
# export_queue.py - 将下载队列导出为DoiHunter可用的文本文件

import json
import os
import argparse
from pathlib import Path

def export_queue_to_txt(output_dir, export_file="papers_to_download.txt"):
    """将下载队列从JSON导出到文本文件"""
    # 设置路径
    json_file = Path(output_dir) / "download_queue.json"
    output_file = Path(output_dir) / export_file
    
    print(f"正在从 {json_file} 导出下载队列...")
    
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            queue_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取下载队列文件: {e}")
        return False
    
    # 提取DOI或标题
    download_list = []
    for item in queue_data:
        # 如果有DOI则使用DOI，否则使用标题
        if "doi" in item and item["doi"]:
            download_list.append(item["doi"])
        elif "title" in item and item["title"]:
            download_list.append(item["title"])
    
    # 保存到文本文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(download_list))
        
        print(f"成功导出 {len(download_list)} 篇论文到 {output_file}")
        return True
    except Exception as e:
        print(f"错误: 保存文本文件时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将下载队列导出为DoiHunter可用的文本文件")
    parser.add_argument("--output-dir", required=True, help="包含download_queue.json的目录")
    parser.add_argument("--export-file", default="papers_to_download.txt", help="导出文件的名称")
    
    args = parser.parse_args()
    
    # 导出队列
    success = export_queue_to_txt(args.output_dir, args.export_file)
    
    if success:
        print("\n如何使用DoiHunter下载论文:")
        print("1. 首先安装DoiHunter: pip install doi_hunter")
        print(f"2. 运行命令: python -m doi_hunter {args.export_file} --batch_size=5")
        print("\n注意: 建议使用较小的batch_size以避免被封锁")

if __name__ == "__main__":
    main()
