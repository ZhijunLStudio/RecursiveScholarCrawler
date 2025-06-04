#!/usr/bin/env python3
# batch_title_to_doi.py - 批量从文本文件中的论文标题查询DOI

import requests
import time
import json
import re
import os
import argparse
from difflib import SequenceMatcher
from pathlib import Path

def normalize_title(title):
    """标准化论文标题以提高匹配率"""
    # 移除特殊字符，保留字母、数字和空格
    clean_title = re.sub(r'[^\w\s]', ' ', title)
    # 转小写并移除多余空格
    clean_title = ' '.join(clean_title.lower().split())
    return clean_title

def similarity_score(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()

def get_doi_from_title(title, api="crossref"):
    """使用API从标题查询DOI"""
    print(f"查询: {title}")
    
    # 设置请求头
    headers = {
        'User-Agent': 'DOIFinder/1.0 (mailto:your-email@example.com)',
        'Accept': 'application/json'
    }
    
    if api == "crossref":
        url = "https://api.crossref.org/works"
        params = {
            "query": title,
            "rows": 5,
            "sort": "score"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                
                # 如果找到结果
                if items:
                    # 找到相似度最高的匹配
                    best_match = None
                    best_score = 0
                    
                    for item in items:
                        item_title = item.get("title", [""])[0] if item.get("title") else ""
                        if not item_title:
                            continue
                            
                        score = similarity_score(title, item_title)
                        
                        # 只考虑相似度超过阈值的结果
                        if score > best_score and score > 0.6:
                            best_score = score
                            best_match = {
                                "doi": item.get("DOI", ""),
                                "title": item_title,
                                "score": score,
                                "publisher": item.get("publisher", ""),
                                "type": item.get("type", ""),
                                "container-title": item.get("container-title", [""])[0] if item.get("container-title") else ""
                            }
                    
                    if best_match:
                        return best_match
            
            # 如果没有找到或者API调用出错
            return {"doi": None, "error": f"无法找到匹配的DOI (状态码: {response.status_code})"}
                
        except Exception as e:
            return {"doi": None, "error": str(e)}
    
    return {"doi": None, "error": "不支持的API"}

def process_titles_from_file(input_file, delay=1.0):
    """从文件中读取论文标题，查询DOI并保存结果到同级目录"""
    # 解析路径
    input_path = Path(input_file)
    input_dir = input_path.parent
    
    # 输出文件路径
    output_json = input_dir / "paper_dois_results.json"
    output_txt = input_dir / "paper_dois.txt"
    output_map = input_dir / "paper_title_doi_map.json"
    
    print(f"读取文件: {input_file}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            titles = [line.strip() for line in f if line.strip()]
        
        print(f"找到 {len(titles)} 个论文标题")
    except Exception as e:
        print(f"无法读取文件: {e}")
        return
    
    # 处理每个标题
    results = []
    title_doi_map = {}
    
    for i, title in enumerate(titles):
        try:
            # 查询DOI
            result = get_doi_from_title(title)
            result["original_title"] = title
            results.append(result)
            
            # 添加到标题-DOI映射
            if result.get("doi"):
                title_doi_map[title] = result["doi"]
            
            # 打印结果
            if result.get("doi"):
                print(f"✓ [{i+1}/{len(titles)}] 找到DOI: {result['doi']} (相似度: {result['score']:.2f})")
                print(f"  标题: {result['title']}")
            else:
                print(f"✗ [{i+1}/{len(titles)}] 未找到DOI: {result.get('error', '未知错误')}")
            
            # 添加延迟，避免API速率限制
            if i < len(titles) - 1:
                time.sleep(delay)
            
            # 每5个请求保存一次中间结果
            if (i + 1) % 5 == 0 or i == len(titles) - 1:
                # 保存详细结果
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                # 保存纯DOI列表
                dois_only = [r.get("doi") for r in results if r.get("doi")]
                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(dois_only))
                
                # 保存标题-DOI映射
                with open(output_map, "w", encoding="utf-8") as f:
                    json.dump(title_doi_map, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"处理标题时出错 [{title}]: {e}")
    
    # 统计信息
    found = sum(1 for r in results if r.get("doi"))
    print("\n===== 结果统计 =====")
    print(f"总计标题: {len(results)}")
    print(f"找到DOI: {found}")
    print(f"未找到DOI: {len(results) - found}")
    print(f"详细结果已保存至: {output_json}")
    print(f"纯DOI列表已保存至: {output_txt}")
    print(f"标题-DOI映射已保存至: {output_map}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="批量从论文标题查询DOI")
    parser.add_argument("--input", "-i", required=True, help="包含论文标题的文本文件路径")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="请求之间的延迟(秒)")
    
    args = parser.parse_args()
    
    # 处理标题
    process_titles_from_file(args.input, args.delay)

if __name__ == "__main__":
    main()
