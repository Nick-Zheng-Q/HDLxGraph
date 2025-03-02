from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from sklearn.neighbors import NearestNeighbors

checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cpu"  # 或 "cpu" 根据您的硬件选择
class PureRAG:
    def __init__(self, documents):
        """
        Initialize CodeT5+ retriever
        :param documents: Dict of {filename: content} or List of text documents
        """
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        # 支持字典或列表输入
        self.documents = documents if isinstance(documents, dict) else {f"doc_{i}": doc for i, doc in enumerate(documents)}
        self.chunks = {}  # 存储每个文件的代码块
        self.chunk_line_indices = {}  # 存储每个文件的行索引
        self.chunk_file_mapping = []  # 存储chunk对应的文件来源
        self.retriever = None

    def split_code_into_chunks(self, code_snippet, filename, max_lines=5):
        lines = code_snippet.splitlines()
        chunks = []
        chunk_line_indices = []
        for i in range(0, len(lines), max_lines):
            chunk = '\n'.join(lines[i:i + max_lines])
            chunks.append(chunk)
            chunk_line_indices.append((i, min(i + max_lines, len(lines))))
        return chunks, chunk_line_indices

    # 生成代码嵌入
    def generate_embedding(self, code_snippet):
        inputs = self.tokenizer.encode(code_snippet, return_tensors="pt").to(device)
        embedding = self.model(inputs)[0]
        return embedding

    def build_database(self):
        all_embeddings = []
        self.chunks = {}
        self.chunk_line_indices = {}
        self.chunk_file_mapping = []
        
        # 处理每个文件
        for filename, content in self.documents.items():
            chunks, line_indices = self.split_code_into_chunks(content, filename)
            self.chunks[filename] = chunks
            self.chunk_line_indices[filename] = line_indices
            
            # 为每个chunk生成embedding
            for chunk in chunks:
                embedding = self.generate_embedding(chunk)
                all_embeddings.append(embedding)
                self.chunk_file_mapping.append(filename)
        
        # 构建检索器
        embeddings_array = np.array([emb.cpu().detach().numpy() for emb in all_embeddings])
        self.retriever = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.retriever.fit(embeddings_array)

    def search(self, query, k=1):
        """
        检索相关代码片段
        :param query: 查询文本
        :param k: 返回的相关片段数量
        :return: List of (filename, code_snippet, line_range)
        """
        query_inputs = self.tokenizer.encode(query, return_tensors="pt").to(device)
        query_embedding = self.model(query_inputs)[0].cpu().detach().numpy()
        query_embedding = query_embedding.reshape(1, -1)
        
        # 获取k个最相关的chunks
        distances, indices = self.retriever.kneighbors(query_embedding, n_neighbors=k)
        
        results = []
        for idx in indices[0]:
            filename = self.chunk_file_mapping[idx]
            # 计算当前文件在idx之前的chunk数量
            prev_chunks = 0
            for i in range(idx):
                if self.chunk_file_mapping[i] == filename:
                    prev_chunks += 1
                    
            # 使用累计的chunk数作为索引
            chunk_idx = prev_chunks
            
            # 添加安全检查
            if chunk_idx >= len(self.chunk_line_indices[filename]):
                continue
                
            start, end = self.chunk_line_indices[filename][chunk_idx]
            code_snippet = self.chunks[filename][chunk_idx]
            results.append({
                'filename': filename,
                'code': code_snippet,
                'line_range': (start, end)
            })
            
        return results

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

if __name__ == "__main__":
    import os
    
    # 从目录加载文档
    doc_dir = r"d:\LLM4RV\verilog-eval\test_verilog"
    documents = load_documents(doc_dir)
    print(f"加载了 {len(documents)} 个文件")
    
    # 初始化RAG
    rag = PureRAG(documents)
    rag.build_database()
    
    # 测试不同的查询
    test_queries = [
        "如何实现一个计数器",
        "移位寄存器的实现",
        "状态机示例",
        "always块的使用方法",
        "时序逻辑电路"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"查询: {query}")
        print("="*50)
        
        results = rag.search(query, k=2)
        for result in results:
            print(f"\n文件: {result['filename']}")
            print(f"行范围: {result['line_range']}")
            print("代码:")
            print(result['code'])

