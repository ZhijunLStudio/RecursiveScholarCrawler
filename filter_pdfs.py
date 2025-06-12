#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import pikepdf

def is_valid_pdf(path):
    """
    尝试打开 PDF，能正常打开即视为“状态正常”
    """
    try:
        # pikepdf.open 在打开损坏文件时会抛出 PdfError
        with pikepdf.open(path):
            return True
    except pikepdf.PdfError:
        return False

def filter_pdfs(src_dir, dst_dir):
    """
    遍历 src_dir 下所有 .pdf 文件，
    将能正常打开的复制到 dst_dir（不保留目录结构）
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.lower().endswith('.pdf'):
                continue
            src_path = os.path.join(root, fname)
            if is_valid_pdf(src_path):
                dst_path = os.path.join(dst_dir, fname)
                # 若担心同名冲突，可以在这里改名或加前缀
                shutil.copy2(src_path, dst_path)
                print(f"[OK]   {src_path} → {dst_path}")
            else:
                print(f"[跳过] {src_path}（无法打开或已损坏）")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="筛选可打开的 PDF 并复制到新文件夹"
    )
    parser.add_argument(
        "-s", "--src",
        required=True,
        help="源文件夹路径，脚本会遍历这个目录下所有 PDF"
    )
    parser.add_argument(
        "-d", "--dst",
        required=True,
        help="目标文件夹路径，符合条件的 PDF 将被复制到这里"
    )
    args = parser.parse_args()

    filter_pdfs(args.src, args.dst)
