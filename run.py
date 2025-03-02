import argparse
import subprocess
import sys
import glob, os
from generation.generation import main_generation
from rag.rag import RAG_Agent

from benchmark.debugging.evaluation import main_debugging
benchmark_path = "/home/qin00162/Workspace/pingqing2024/HDLxGraph/benchmark"
dataset_path = "/home/qin00162/Workspace/pingqing2024/HDLxGraph/dataset"

def load_documents(directory):
    """
    从目录中加载所有文件
    :param directory: 文件目录路径
    :return: Dict of {filename: content}
    """
    documents = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.v', '.sv')):  # 只读取Verilog/SystemVerilog文件
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 使用相对路径作为键
                rel_path = os.path.relpath(filepath, directory)
                documents[rel_path] = content
    return documents

def code_search(model, RAG):
    rag_agent = RAG_Agent(RAG)
    documents = None
    rag_agent.build_database(documents)
    print(f"Executing code search with model {model} and RAG {RAG}...")

def code_debugging(model, RAG):
    rag_agent = RAG_Agent(RAG)
    documents_path = f"{dataset_path}/debugging/database"
    documents = load_documents(documents_path)
    rag_agent.build_database(documents)
    main_debugging(model, rag_agent, main_generation)

def code_completion(model, RAG):
    rag_agent = RAG_Agent(RAG)
    documents_path = f"{dataset_path}/completion"
    documents = load_documents(documents_path)
    rag_agent.build_database(documents)
    # verilog-eval
    task = "spec-to-rtl"
    command = f"""
    conda init bash
    conda activate codex
    mkdir -p build/
    ../configure --with-task={task} --with-model={model} --with-examples=0 --with-samples=2
    make
    """
    # 使用 bash 来执行这些命令
    result = subprocess.run(command, shell=True, cwd=f"{benchmark_path}/completion/verilog-eval", executable="/bin/bash")
    if result.returncode != 0:
        print("执行过程中出现错误。")
        sys.exit(result.returncode)

    def process_single_test(base_path: str) -> None:
        """处理单个测试用例"""
        # 读取系统提示和完整提示
        with open(f"{base_path}_systemprompt.txt", 'r') as f:
            system_prompt = f.read()
        with open(f"{base_path}_fullprompt.txt", 'r') as f:
            full_prompt = f.read()
            
        # 调用模型生成响应
        response = main_generation(model, system_prompt, full_prompt)
        
        # 保存响应
        with open(f"{base_path}_response.txt", 'w') as f:
            f.write(response)
            
    def process_directory(directory: str) -> None:
        """处理目录中的所有测试用例"""
        # 查找所有以_fullprompt.txt结尾的文件
        prompt_files = glob.glob(os.path.join(directory, "*_fullprompt.txt"))
        
        for prompt_file in prompt_files:
            # 获取基础文件名（去掉_fullprompt.txt）
            base_path = prompt_file[:-14]  # len("_fullprompt.txt") = 14
            print(f"Processing: {os.path.basename(base_path)}")
            
            try:
                process_single_test(base_path)
                print(f"✓ Completed: {os.path.basename(base_path)}")
            except Exception as e:
                print(f"✗ Failed: {os.path.basename(base_path)}")
                print(f"  Error: {str(e)}")
                
    process_directory(f"{benchmark_path}/verilog-eval/build")

def select_task(task, model, RAG):
    tasks = {
        'code_search': code_search,
        'code_debugging': code_debugging,
        'code_generation': code_completion,
    }
    
    if task in tasks:
        tasks[task](model, RAG)
    else:
        print(f"Task '{task}' not recognized!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Execution Script")
    parser.add_argument('task', choices=['code_search', 'code_debugging', 'code_completion'], help="The task to execute")
    parser.add_argument('model', choices=['starcoder2-7b','claude', 'qwen'], help="The model to use")
    parser.add_argument('RAG', choices=['no-rag','bm25', 'similarity', 'HDLxGraph'], help="The RAG to use")
    
    args = parser.parse_args()

    select_task(args.task, args.model, args.RAG)

