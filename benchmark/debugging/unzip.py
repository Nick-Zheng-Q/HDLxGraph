import gzip
import json
import os
from pathlib import Path
from typing import List, Dict

def read_json_files(directory: str, 
                    encoding: str = 'utf-8', 
                    ignore_errors: bool = True) -> List[Dict]:
    data = []
    json_extensions = ('.json', '.json.gz')  # 支持.json和.json.gz
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() not in json_extensions:
            continue
        
        try:
            if file_path.suffix.lower() == '.gz':
                import gzip
                with gzip.open(file_path, 'rt', encoding=encoding) as f:
                    file_data = json.load(f)
            else:
                with open(file_path, 'r', encoding=encoding) as f:
                    file_data = json.load(f)
            # 处理JSON Lines格式（每行一个JSON）
            if isinstance(file_data, list):
                data.extend(file_data)
            else:
                data.append(file_data)
                
        except Exception as e:
            if not ignore_errors:
                raise
            print(f"跳过错误文件 {file_path}: {str(e)}")
    
    return data

def read_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:  # 使用gzip.open打开.gz文件，并指定模式为't'表示文本模式
        data = json.load(f)
        # data = json.load(f)  # 使用json.load读取并解析JSON数据
    return data

def get_bug_info(json_data):
    filename = json_data.get("filename")
    before = json_data.get("before")
    after = json_data.get("after")
    commit_message = json_data.get("commit_message")
    pull_number = json_data.get("pull_number")
    pull_title = json_data.get("pull_title")
    pull_body = json_data.get("pull_body")
    
    return before, after, pull_body, pull_title

def filter_benchmark(json_data):
    commit_sha = json_data.get("commit_sha")

    return commit_sha

