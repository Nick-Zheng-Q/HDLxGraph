from rag import BM25Retriever
from rag import PureRAG
from rag import main_generation
from rag.neo4j_rag import Neo4jRAG

class RAG_Agent:
    def __init__(self, rag_type, generate_func, model):
        """
        初始化RAG代理
        :param rag_type: RAG类型 ("no-rag", "bm25", "similarity", "HDLxGraph")
        """
        self.rag_type = rag_type
        self.rag_agent = None
        self.model = model
        self.generate_func = generate_func

    def build_database(self, documents):
        """
        构建检索数据库
        :param documents: 文档集合，可以是字典或列表
        """
        if self.rag_type == "no-rag":
            return
        elif self.rag_type == "bm25":
            self.rag_agent = BM25Retriever(documents)
        elif self.rag_type == "similarity":
            self.rag_agent = PureRAG(documents)
            self.rag_agent.build_database()
        elif self.rag_type == "HDLxGraph":
            self.rag_agent = Neo4jRAG(self.generate_func)
            # self.rag_agent.store_verilog(documents)

    def retrieve(self, prompt, k=2):
        """
        检索相关内容
        :param prompt: 查询文本
        :param k: 返回结果数量
        :return: 检索结果
        """
        if self.rag_type == "no-rag":
            return ""
        elif self.rag_type == "similarity":
            results = self.rag_agent.search(prompt, k=k)
            # 格式化结果
            # retrieved_content = []
            # for result in results:
            #     retrieved_content.append(
            #         f"File: {result['filename']}\n"
            #         f"Lines {result['line_range'][0]}-{result['line_range'][1]}:\n"
            #         f"{result['code']}\n"
            #     )
            retrieved_content = [result['code'] for result in results]
            return "\n".join(retrieved_content)
        elif self.rag_type == "HDLxGraph":
            return self.rag_agent.search(self.model, prompt)
        else:
            return self.rag_agent.search(prompt)

