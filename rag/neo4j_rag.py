from py2neo import Graph
from transformers import AutoTokenizer, AutoModel
import torch
import traceback
import re
import json
import numpy as np
from pyverilog.vparser.parser import ParseError

def extract_json_from_response(response):
    """从 LLM 回复中提取 JSON 内容"""
    # 匹配包含在 { } 内且包含指定字段的 JSON
    pattern = r'\{[^{]*"module_desc"[^}]*"block_desc"[^}]*"signal_desc"[^}]*\}'
    match = re.search(pattern, response)
    
    if not match:
        return None
        
    try:
        # 解析提取的 JSON 字符串
        json_str = match.group()
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

class Neo4jRAG:
    def __init__(self, generate_func, uri="http://localhost:7474", 
                 user="neo4j", 
                 password="neo4j",
                 ):
        """Initialize Neo4j RAG retriever"""
        self.uri = uri
        self.user = user
        self.password = password
        self.graph = None
        
        # 初始化相同的编码模型
        checkpoint = "Salesforce/codet5p-110m-embedding"
        device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        self.generate_func = generate_func
        
        # Connect to Neo4j
        self.connect()
        # self.graph.delete_all()

    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.graph = Graph(self.uri, auth=(self.user, self.password))
            # self.graph = Graph(self.uri, user=self.user, password=self.password)
            print("Successfully connected to Neo4j database")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {str(e)}")
            self.graph = None

    # def store_verilog(self, file_path):
    #     pass
    def store_verilog(self, file_path):
        """Store Verilog file into Neo4j database"""
        from .test_neo4j import store_verilog_graph
        for document in file_path:
            print(f"document: {document}")
            try:
                store_verilog_graph(document)
            except ParseError as e:
                print(f"Pyverilog解析错误: {str(e)}")
                traceback.print_exc()  # 打印完整的堆栈跟踪
                continue
            except Exception as e:
                print(f"处理文件时出错: {str(e)}")
                print("跳过此文件...")
                continue

    def compute_similarity(self, query_embedding, target_embedding):
        """Compute cosine similarity between embeddings"""
        return torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(target_embedding),
            dim=0
        ).item()

    def search_module(self, description, top_k=10):
        """基于语义描述搜索模块"""
        # 对查询文本进行编码
        query_embedding = self.encode_code(description).tolist()
        
        # 使用存储的embedding进行搜索
        query = """
        MATCH (m:Module)
        WITH m, gds.similarity.cosine(m.code_embedding, $query_embedding) AS similarity
        RETURN m.name as name, m.code as code, similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        results = list(self.graph.run(query, 
                                    query_embedding=query_embedding,
                                    top_k=top_k).data())
        
        return results

    def search_block(self, description, top_k=10):
        """基于语义描述搜索代码块"""
        query_embedding = self.encode_code(description).tolist()
        
        query = """
        MATCH (b:Block)
        WITH b, gds.similarity.cosine(b.code_embedding, $query_embedding) AS similarity
        RETURN b.id as id, b.type as type, b.code as block_code, similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        return list(self.graph.run(query, 
                                 query_embedding=query_embedding,
                                 top_k=top_k).data())

    def search_signal(self, description, top_k=10):
        """基于语义描述搜索信号"""
        query_embedding = self.encode_code(description).tolist()
        
        query = """
        MATCH (s:Block)
        WHERE s.context_embedding IS NOT NULL
        WITH s, gds.similarity.cosine(s.context_embedding, $query_embedding) AS similarity
        RETURN s.name as signal_name, s.context as context, similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        return list(self.graph.run(query,
                                 query_embedding=query_embedding,
                                 top_k=top_k).data())

    def encode_code(self, code_text):
        """使用与test_neo4j相同的编码方法"""
        inputs = self.tokenizer(code_text, return_tensors="pt")["input_ids"].to("cuda")
        embedding = self.model(inputs)[0]
        return embedding

    def search_module_block(self, module_desc, block_desc, top_k=10):
        """基于语义和图关系搜索特定模块中的代码块"""
        # 先进行模块搜索
        module_query_embedding = self.encode_code(module_desc).tolist()
        
        # 使用图关系和语义相似度联合查询
        query = """
        MATCH (m:Module)-[:CONTAINS]->(b:Block)
WHERE EXISTS(m.code_embedding) AND EXISTS(b.code_embedding)  // 过滤空嵌入
WITH m, b,
     gds.similarity.cosine(
       m.code_embedding, 
       coalesce($module_embedding, [])  // 处理空参数
     ) AS module_score,
     gds.similarity.cosine(
       b.code_embedding, 
       coalesce($block_embedding, [])   // 处理空参数
     ) AS block_score
WHERE module_score IS NOT NULL AND block_score IS NOT NULL  // 过滤无效计算
WITH m, b, 
     (module_score + block_score)/2 AS combined_score
RETURN b.code AS block_code,
       combined_score AS similarity  // 确保别名与预期字段名一致
ORDER BY similarity DESC
LIMIT $top_k

        """
        
        results = list(self.graph.run(
            query,
            module_embedding=module_query_embedding,
            block_embedding=self.encode_code(block_desc).tolist(),
            top_k=top_k
        ).data())
        
        return results

    def search_module_signal(self, module_desc, signal_desc, top_k=10):
        """基于语义和图关系搜索特定模块中的信号"""
        module_embedding=self.encode_code(module_desc).tolist()
        signal_embedding=self.encode_code(signal_desc).tolist()
        query = """
        MATCH (m:Module)-[:CONTAINS]->(b:Block)
WHERE EXISTS(m.code_embedding) AND EXISTS(b.code_embedding)
WITH m, b,
     gds.similarity.cosine(m.code_embedding, $module_embedding) AS module_score,
     gds.similarity.cosine(b.code_embedding, $block_embedding) AS block_score
WHERE module_score IS NOT NULL AND block_score IS NOT NULL
WITH m, b, 
     (module_score + block_score)/2 AS block_combined_score
ORDER BY block_combined_score DESC
LIMIT 5  // 确保限制在聚合前应用

// 关联信号并计算加权分数
MATCH (b)-[:CONTAINS]->(s:Signal)
WITH s, 
     SUM(block_combined_score) AS total_score,
     COUNT(DISTINCT b) AS block_count
WHERE block_count > 0  // 防止除以零
WITH s, 
     (total_score / 5) * (block_count / 5.0) AS signal_score  // 明确分母为浮点
RETURN s.name AS signal_name,
       signal_score AS similarity  // 统一字段名
ORDER BY similarity DESC
LIMIT $top_k

        """
        
        results = list(self.graph.run(
            query,
            module_embedding=module_embedding,
            block_embedding=signal_embedding,
            top_k=top_k
        ).data())
        
        return results

    def search(self, model, description, top_k=10):
        """统一的搜索接口，使用 LLM 分析查询并进行多维度搜索
        
        Args:
            description (str): 用户的搜索描述
            top_k (int): 每种类型返回的最大结果数
        """
        system_message="""You are a Hardware Description Language search assistant.
Your role is to analyze user search queries and break them down into the following aspects:
1. Module-related description
2. Code block-related description
3. Signal-related description

For each aspect, you need to provide clear descriptive text. If an aspect is not explicitly mentioned in the user's query, return an empty string.

Please return results in the following JSON format:
{
    "module_desc": "module description",
    "block_desc": "code block description",
    "signal_desc": "signal description"
}

Examples:
1. Query: "Find a counter module with reset functionality"
   Response: {
       "module_desc": "counter module with reset functionality",
       "block_desc": "",
       "signal_desc": ""
   }

2. Query: "Look for clock-triggered counting logic in reset counter"
   Response: {
       "module_desc": "reset counter module",
       "block_desc": "clock-triggered counting logic",
       "signal_desc": ""
   }

3. Query: "Find reset signal in counter module"
   Response: {
       "module_desc": "counter module",
       "block_desc": "",
       "signal_desc": "reset signal"
   }

Focus on technical aspects and HDL-specific terminology in your descriptions."""

        # response = self.generate_func(model, system_message, description)
        # search_params = extract_json_from_response(response)
        search_params = {
            "module_desc": description,
            "block_desc":description
        }
        # if search_params is None:
        #     return ""
        # # 执行搜索
        results = []
        # 
        # # 如果同时指定了模块和其他描述，使用组合搜索
        # if search_params["module_desc"] and (search_params["block_desc"] or search_params["signal_desc"]):
        #     if search_params["block_desc"]:
        #         block_results = self.search_module_block(
        #             search_params["module_desc"],
        #             search_params["block_desc"],
        #             top_k
        #         )
        #         results.append(("模块代码块搜索结果", "block", block_results))
        #         
        #     if search_params["signal_desc"]:
        #         signal_results = self.search_module_signal(
        #             search_params["module_desc"],
        #             search_params["signal_desc"],
        #             top_k
        #         )
        #         results.append(("模块信号搜索结果", "signal", signal_results))
        # else:
        #     # 单独搜索
        #     if search_params["module_desc"]:
        #         module_results = self.search_module(search_params["module_desc"], top_k)
        #         results.append(("模块搜索结果", "module", module_results))
        #         
        if search_params["block_desc"]:
            block_results = self.search_block(search_params["block_desc"], top_k)
            results.append(("代码块搜索结果", "block", block_results))
        #         
        #     if search_params["signal_desc"]:
        #         signal_results = self.search_signal(search_params["signal_desc"], top_k)
        #         results.append(("信号搜索结果", "signal", signal_results))
        
        return self.format_search_results(results)

    def format_search_results(self, results):
        """格式化搜索结果
        
        Args:
            results: [(title, type, result_list), ...]
        """
        if not results:
            return "未找到匹配结果"
        
        output = []
        for title, result_type, result_list in results:
            output.append(f"\n=== {title} ===")
            
            for i, item in enumerate(result_list, 1):
                output.append(f"\n结果 {i} (相似度: {item['similarity']:.3f}):")
                
                if result_type == "module":
                    output.append(f"模块名称: {item['name']}")
                    code = item['code'][:200] + "..." if len(item['code']) > 200 else item['code']
                    output.append(f"代码片段:\n{code}")
                    
                elif result_type == "block":
                    output.append(f"代码内容:\n{item['block_code']}")
                    
                elif result_type == "signal":
                    output.append(f"信号名称: {item['signal_name']}")
                
                output.append("-" * 50)
        
        return "\n".join(output)

