from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

documents = SimpleDirectoryReader("data").load_data()
# 初始化本地嵌入模型（避免API依赖）
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文轻量级嵌入模型
    device="cpu"  # 使用CPU运行
)
# 创建索引（只测试嵌入模型功能，不进行查询）
index = VectorStoreIndex(
    documents,
    embed_model=embed_model,  # 嵌入模型，用于将文本转换为向量
    similarity_top_k=5,        # 返回最相似的节点数量，默认5
    distance_metric="cosine"   # 相似度度量方式，常用cosine或euclidean
)

# 保存索引到本地
index.storage_context.persist(persist_dir="llama_index_storage")

print("索引创建成功！")
print(f"文档数量: {len(documents)}")
print("索引已保存到本地目录: llama_index_storage")