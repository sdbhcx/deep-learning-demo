from dotenv import load_dotenv, find_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings

load_dotenv(find_dotenv())

embedding_model = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("GLM_API_KEY"),
)
# 加载本地FAISS索引，由于是本地创建的安全文件，可以允许反序列化
# 初始数据
# initial_texts = ["我的爱好是旅游", "我喜欢读书", "我热爱编程"]
# initial_metadata = [{"category": "hobby"}, {"category": "hobby"}, {"category": "skill"}]

# # 创建新的FAISS索引
# vector_store = FAISS.from_texts(
#     texts=initial_texts,
#     embedding=embedding_model,
#     metadatas=initial_metadata
# )

# 保存索引到本地（可选）
# vector_store.save_local("faiss_index")
# 使用相对路径加载项目根目录下的faiss_index文件夹
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

def add_data_to_index(data, metadata):
    vector_store.add_texts([data], metadata=[metadata])

def search_similar(query):
    results = vector_store.similarity_search(query, k=2)
    for result in results:
        print(f"Content: {result.page_content}\nMetadata: {result.metadata}\n")
    return results

add_data_to_index("爱吃水果", {"category": "hobby"})
search_similar("喜欢哪些东西")