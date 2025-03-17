import argparse
import subprocess
import sys
import glob, os
import re
from generation.generation import main_generation
from rag.rag import RAG_Agent
from mrr import MRREvaluator, load_processed_data, load_signal_data

from benchmark.debugging.evaluation import main_debugging
benchmark_path = "/home/pingqing2024/HDLxGraph/benchmark"
dataset_path = "/home/pingqing2024/HDLxGraph/dataset"

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
                documents[filepath] = content
    return documents

def build_data(model, RAG):
    rag_agent = RAG_Agent(RAG, main_generation, model)
    documents_path = f"{benchmark_path}/search/files"
    documents = load_documents(documents_path)
    rag_agent.build_database(documents)

def code_search(model, RAG):
    rag_agent = RAG_Agent(RAG, main_generation, model)
    documents_path = f"{benchmark_path}/search/files"
    documents = load_documents(documents_path)
    rag_agent.build_database(documents)
    print(f"Executing code search with model {model} and RAG {RAG}...")
    test_queries = []
    ground_truth = {}
    processed_data_dir = f"{benchmark_path}/search/query_and_truth"
    print(f"从处理后的数据目录加载查询和地面真相: {processed_data_dir}")
    # test_queries, ground_truth = load_processed_data(processed_data_dir)
    test_queries, ground_truth = load_signal_data(processed_data_dir)
    print(f"加载了 {len(test_queries)} 个测试查询")
    evaluator = MRREvaluator(ground_truth)
    HDLxGraph_results = evaluator.evaluate_queries(
        test_queries,
        lambda query: rag_agent.retrieve(query, k=10)
    )
    print("\n评估结果:")
    print(f"HDLxGraph MRR: {HDLxGraph_results['mrr']:.4f}")
    print("\nHDLxGraph 每个查询的倒数排名:")
    for query, rr in HDLxGraph_results['query_results'].items():
        print(f"  {query[:50]}...: {rr:.4f}")

def code_debugging(model, RAG):
    rag_agent = RAG_Agent(RAG, main_generation, model)
    documents_path = f"{dataset_path}/debugging/database"
    documents = load_documents(documents_path)
    rag_agent.build_database(documents)
    main_debugging(model, rag_agent, main_generation)

def code_completion(model, RAG):
    rag_agent = RAG_Agent(RAG, main_generation, model)
    documents_path = f"{dataset_path}/completion/database"
    documents = load_documents(documents_path)
    rag_agent.build_database(documents)
    # verilog-eval

    def process_single_test(base_path: str) -> None:
        """处理单个测试用例"""
        # 读取系统提示和完整提示
        with open(f"{base_path}_systemprompt.txt", 'r') as f:
            system_prompt = f.read()
        with open(f"{base_path}_fullprompt.txt", 'r') as f:
            full_prompt = f.read()
        retrieve = rag_agent.retrieve(full_prompt)
        full_prompt = """
        Please write your code as:
        [BEGIN]
            your code here
        [DONE]
        """ + full_prompt 
        retrieve = """
        The following is the code retrieved from the database for you as a reference
        """ + retrieve
            
        # 调用模型生成响应
        response = main_generation(model, system_prompt, full_prompt)
        
        # 保存响应
        with open(f"{base_path}_response.txt", 'w') as f:
            print(f"Write {base_path}_response.txt")
            f.write(response)
            
    def process_directory(directory: str) -> None:
        """处理目录中的所有测试用例"""
        
        # 查找所有以_fullprompt.txt结尾的文件
        prompt_files = glob.glob(os.path.join(directory, "**/*_fullprompt.txt"), recursive=True)

        for prompt_file in prompt_files:
        # 精准提取 base_path
            base_path = re.sub(r"_fullprompt\.txt$", "", prompt_file)  # 移除 _fullprompt.txt
            base_path = base_path.rstrip("_")  # 去除可能存在的结尾下划线
            
            print(f"Processing: {os.path.basename(base_path)}")

            sv_path = f"{base_path}.sv"
            try:
                os.remove(sv_path)
            except FileNotFoundError:
                pass  # 文件不存在时的静默处理
            compile_path = f"{base_path}-sv-iv-test.log"
            try:
                os.remove(compile_path)
            except FileNotFoundError:
                pass  # 文件不存在时的静默处理
            try:
                # 生成系统提示路径（强制单下划线）
                system_prompt_path = f"{base_path}_systemprompt.txt"
                
                # 验证路径是否存在
                if not os.path.exists(system_prompt_path):
                    raise FileNotFoundError(f"系统提示文件缺失: {system_prompt_path}")
                
                process_single_test(base_path)
                print(f"✓ Completed: {os.path.basename(base_path)}")
            except Exception as e:
                print(f"✗ Failed: {os.path.basename(base_path)}")
                print(f"  Error: {str(e)}")
                
    process_directory(f"{benchmark_path}/completion/verilog-eval/build")
    
    command = f"""
    SHELL=/bin/bash make
    """
    result = subprocess.run(command, shell=True, cwd=f"{benchmark_path}/completion/verilog-eval/build", executable="/bin/bash")

def select_task(task, model, RAG):
    tasks = {
        'code_search': code_search,
        'code_debugging': code_debugging,
        'code_completion': code_completion,
    }
    
    if task in tasks:
        tasks[task](model, RAG)
    else:
        print(f"Task '{task}' not recognized!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Execution Script")
    parser.add_argument('task', choices=['code_search', 'code_debugging', 'code_completion'], help="The task to execute")
    parser.add_argument('model', choices=['llama', 'starcoder2-7b','claude', 'qwen'], help="The model to use")
    parser.add_argument('RAG', choices=['no-rag','bm25', 'similarity', 'HDLxGraph'], help="The RAG to use")
    
    args = parser.parse_args()

    # build_data(args.model, args.RAG)
    select_task(args.task, args.model, args.RAG)

