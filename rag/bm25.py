from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import os

class BM25Retriever:
    def __init__(self, documents, k1=1.2, b=0.75, preprocess=True):
        """
        Initialize BM25 retriever
        :param documents: Dict of {filename: content} or List of documents
        :param k1: BM25 k1 parameter (controls term frequency saturation)
        :param b: BM25 b parameter (controls document length normalization)
        :param preprocess: Whether to apply text preprocessing
        """
        self._check_nltk_resources()
        self.preprocess_enabled = preprocess
        
        # 支持字典输入
        if isinstance(documents, dict):
            self.filenames = list(documents.keys())
            self.raw_documents = list(documents.values())
        else:
            self.filenames = [f"doc_{i}" for i in range(len(documents))]
            self.raw_documents = documents
            
        # 存储每个文档的行信息
        self.line_indices = {}
        for i, doc in enumerate(self.raw_documents):
            lines = doc.splitlines()
            self.line_indices[self.filenames[i]] = [
                (j, j + 1) for j in range(len(lines))
            ]
        
        self.tokenized_corpus = self._preprocess_docs(self.raw_documents) if preprocess else [word_tokenize(doc) for doc in self.raw_documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    def _check_nltk_resources(self):
        """Verify and download required NLTK resources"""
        required_resources = [
            'punkt',
            'stopwords',
            'wordnet',
            'omw-1.4',  # Open Multilingual Wordnet
            'punkt_tab'  # 添加punkt_tab资源
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
                
        # 特别检查tokenizers/punkt
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading additional punkt resources")
            nltk.download('punkt', quiet=True)

    def _preprocess(self, text):
        """Text cleaning pipeline"""
        tokens = word_tokenize(text.lower())
        return [
            token for token in tokens
            if token not in stopwords.words('english')
            and token not in string.punctuation
            and len(token) > 2
        ]

    def _preprocess_docs(self, docs):
        """Batch process documents"""
        return [self._preprocess(doc) for doc in docs]

    def search(self, query, top_k=1):
        """
        Search for most relevant documents
        :param query: Search query string
        :param top_k: Number of results to return
        :return: List of dicts with filename, code, and line_range
        """
        tokens = self._preprocess(query) if self.preprocess_enabled else word_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        # 获取排序后的索引
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in ranked_indices:
            filename = self.filenames[idx]
            code = self.raw_documents[idx]
            # 获取整个文件的行范围
            line_range = (0, len(code.splitlines()))
            
            results.append({
                'filename': filename,
                'code': code,
                'line_range': line_range
            })
            
        return results

# Usage Example
if __name__ == "__main__":
    import os
    
    # 从目录加载文档
    def load_documents(directory):
        documents = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.v', '.sv')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    rel_path = os.path.relpath(filepath, directory)
                    documents[rel_path] = content
        return documents

    # 测试代码
    doc_dir = r"d:\LLM4RV\verilog-eval\test_verilog"
    documents = load_documents(doc_dir)
    print(f"加载了 {len(documents)} 个文件")
    
    # 初始化检索器
    retriever = BM25Retriever(documents)
    
    # 测试查询
    test_queries = [
        "如何实现一个计数器",
        "移位寄存器的实现",
        "状态机示例"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"查询: {query}")
        print("="*50)
        
        results = retriever.search(query, top_k=2)
        for result in results:
            print(f"\n文件: {result['filename']}")
            print(f"行范围: {result['line_range']}")
            print("代码:")
            print(result['code'])

