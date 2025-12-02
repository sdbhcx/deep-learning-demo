import os
import sys
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

from langchain_community.chat_models import ChatZhipuAI 
from langchain_core.prompts import PromptTemplate

# 获取API密钥
api_key = os.environ.get("GLM_API_KEY")


# 初始化智谱AI的Chat模型
# 智谱AI模型名称通常为glm-4系列
llm = ChatZhipuAI(
    model="glm-4",  # 智谱AI的标准模型名称
    temperature=0.7,
    api_key=api_key
)

loan_template = PromptTemplate(
    input_variables=["user_input"],  # input_variables应该是列表
    template="用户询问贷款问题：{user_input}，请简述贷款流程。"
)
material_template = PromptTemplate(
    input_variables=["response"],  # input_variables应该是列表
    template="请根据贷款流程结果：{response}，请说明贷款需要的材料。"
)

# 使用LangChain 1.x推荐的Runnable接口
# 直接将prompt和llm连接起来形成链
chain = loan_template | llm | material_template | llm

user_input = "我想申请贷款"
print(f"\n用户输入: {user_input}")
print("正在获取回复...")

try:
    # 调用链并获取响应
    response = chain.invoke({"user_input": user_input})
    # 输出响应内容
    print(f"\n模型回复: {response.content}")
except Exception as e:
    print(f"\n调用过程中发生错误: {e}")
    print("\n可能的原因:")
    print("1. API密钥无效或权限不足")
    print("2. 网络连接问题")
    print("3. 模型名称不正确")
    print("4. 智谱AI服务暂时不可用")