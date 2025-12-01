#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化函数调用示例 - 改进版NLG.py

本文件演示如何将原始NLG.py中的手动函数调用改进为使用FunctionCaller类的自动化方式。
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class FunctionCaller:
    """
    函数调用自动化处理器
    
    这个类提供了一个更自动化的方式来处理OpenAI的工具调用，通过注册函数，
    自动解析参数，执行函数，并处理结果。
    """
    
    def __init__(self):
        """初始化函数调用器，创建函数注册表"""
        self.function_registry: Dict[str, Any] = {}
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    
    def register_function(self, name: str, func: Any, description: str, parameters: Dict) -> None:
        """
        注册一个函数及其元数据到函数注册表
        
        Args:
            name: 函数名称，需要与OpenAI工具定义中的name一致
            func: 要注册的函数对象
            description: 函数描述
            parameters: 函数参数定义
        """
        self.function_registry[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        生成OpenAI工具调用所需的工具定义列表
        
        Returns:
            工具定义列表
        """
        tools = []
        for name, info in self.function_registry.items():
            tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"]
                }
            }
            tools.append(tool)
        return tools
    
    def execute_function(self, function_name: str, args: Dict) -> Any:
        """
        执行注册的函数
        
        Args:
            function_name: 函数名称
            args: 函数参数字典
            
        Returns:
            函数执行结果
            
        Raises:
            ValueError: 如果函数未注册
        """
        if function_name not in self.function_registry:
            raise ValueError(f"函数 '{function_name}' 未注册")
        
        func = self.function_registry[function_name]["function"]
        return func(**args)
    
    def get_completion(self, messages: List[Dict], model: str = "gpt-3.5-turbo-1106") -> Any:
        """
        获取模型完成结果，自动处理函数调用
        
        Args:
            messages: 消息列表
            model: 使用的模型
            
        Returns:
            模型响应
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            tools=self.get_tool_definitions()
        )
        return response.choices[0].message
    
    def process_response(self, messages: List[Dict]) -> List[Dict]:
        """
        处理模型响应，自动执行函数调用并更新消息列表
        
        Args:
            messages: 消息列表
            
        Returns:
            更新后的消息列表
        """
        # 获取初始响应
        response = self.get_completion(messages)
        messages.append(response)
        
        # 处理工具调用
        while response.tool_calls:
            tool_call = response.tool_calls[0]  # 简化处理，只处理第一个工具调用
            function_name = tool_call.function.name
            
            # 解析参数
            try:
                arguments = json.loads(tool_call.function.arguments)
                print(f"====调用函数: {function_name} ====")
                print(f"参数: {arguments}")
                
                # 执行函数
                result = self.execute_function(function_name, arguments)
                print(f"函数执行结果: {result}")
                
                # 将工具执行结果添加到消息列表
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(result)
                })
                
                # 获取下一轮响应
                response = self.get_completion(messages)
                messages.append(response)
                
            except json.JSONDecodeError as e:
                print(f"解析函数参数失败: {str(e)}")
                break
            except Exception as e:
                print(f"执行函数失败: {str(e)}")
                break
        
        return messages

# 创建数据库并插入示例数据
def setup_database():
    """
    设置内存数据库并插入示例数据
    
    Returns:
        数据库连接和游标
    """
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # 创建orders表
    database_schema = """
    CREATE TABLE orders (
        id INT PRIMARY KEY NOT NULL, 
        customer_id INT NOT NULL, 
        product_id STR NOT NULL, 
        price DECIMAL(10,2) NOT NULL, 
        status INT NOT NULL, 
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
        pay_time TIMESTAMP
    );
    """
    cursor.execute(database_schema)
    
    # 插入模拟数据
    mock_data = [
        (1, 1001, 'TSHIRT_1', 50.00, 0, '2023-10-12 10:00:00', None),
        (2, 1001, 'TSHIRT_2', 75.50, 1, '2023-10-16 11:00:00', '2023-08-16 12:00:00'),
        (3, 1002, 'SHOES_X2', 25.25, 2, '2023-10-17 12:30:00', '2023-08-17 13:00:00'),
        (4, 1003, 'HAT_Z112', 60.75, 1, '2023-10-20 14:00:00', '2023-08-20 15:00:00'),
        (5, 1002, 'WATCH_X001', 90.00, 0, '2023-10-28 16:00:00', None)
    ]
    
    for record in mock_data:
        cursor.execute('''
        INSERT INTO orders (id, customer_id, product_id, price, status, create_time, pay_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', record)
    
    conn.commit()
    return conn, cursor

# 定义ask_database函数
def create_ask_database_function(cursor):
    """
    创建ask_database函数，使用指定的数据库游标
    
    Args:
        cursor: 数据库游标
        
    Returns:
        ask_database函数
    """
    def ask_database(query: str) -> List[Any]:
        """
        执行SQL查询
        
        Args:
            query: SQL查询语句
            
        Returns:
            查询结果列表
        """
        cursor.execute(query)
        return cursor.fetchall()
    
    return ask_database

# 自动生成SQL查询的函数
def create_auto_sql_function():
    """
    创建自动SQL生成函数
    
    Returns:
        auto_generate_sql函数
    """
    def auto_generate_sql(question: str, database_schema: str) -> str:
        """
        根据问题和数据库模式自动生成SQL查询
        
        Args:
            question: 用户问题
            database_schema: 数据库模式描述
            
        Returns:
            生成的SQL查询语句
        """
        # 这里可以使用另一个LLM调用来生成SQL
        # 简化版本直接返回示例SQL
        if "销售额" in question:
            return "SELECT SUM(price) FROM orders WHERE status = 1"  # 已支付订单总额
        elif "用户" in question and "消费" in question:
            return "SELECT customer_id, SUM(price) FROM orders WHERE status = 1 GROUP BY customer_id ORDER BY SUM(price) DESC LIMIT 1"
        else:
            return "SELECT * FROM orders LIMIT 10"
    
    return auto_generate_sql

def run_automated_demo():
    """
    运行自动化函数调用演示
    """
    print("自动化函数调用演示")
    print("=" * 50)
    
    # 1. 设置数据库
    conn, cursor = setup_database()
    
    # 2. 创建函数调用器
    function_caller = FunctionCaller()
    
    # 3. 创建并注册函数
    ask_database = create_ask_database_function(cursor)
    auto_generate_sql = create_auto_sql_function()
    
    # 数据库模式字符串
    database_schema_string = """
    CREATE TABLE orders (
        id INT PRIMARY KEY NOT NULL, 
        customer_id INT NOT NULL, 
        product_id STR NOT NULL, 
        price DECIMAL(10,2) NOT NULL, 
        status INT NOT NULL, 
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
        pay_time TIMESTAMP
    );
    """
    
    # 注册ask_database函数
    function_caller.register_function(
        name="ask_database",
        func=ask_database,
        description="使用此函数回答有关业务的用户问题。输出应该是一个完全形成的SQL查询。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"SQL查询提取信息以回答用户问题。\nSQL应该使用以下数据库模式编写：\n{database_schema_string}\n查询应该以纯文本形式返回，而不是JSON格式。\n查询应该只包含SQLite支持的语法。"
                }
            },
            "required": ["query"]
        }
    )
    
    # 注册auto_generate_sql函数
    function_caller.register_function(
        name="auto_generate_sql",
        func=auto_generate_sql,
        description="根据用户问题和数据库模式自动生成SQL查询",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的问题"
                },
                "database_schema": {
                    "type": "string",
                    "description": "数据库模式描述"
                }
            },
            "required": ["question", "database_schema"]
        }
    )
    
    # 4. 定义用户查询
    prompts = [
        "上个月的销售额是多少？",
        "哪个用户消费最高？消费了多少？"
    ]
    
    for prompt in prompts:
        print(f"\n\n====处理用户问题: {prompt} ====")
        
        # 5. 初始化消息列表
        messages = [
            {"role": "system", "content": f"基于 orders 表回答用户问题。数据库模式：{database_schema_string}"},
            {"role": "user", "content": prompt}
        ]
        
        # 6. 自动处理函数调用
        messages = function_caller.process_response(messages)
        
        # 7. 获取最终回复
        final_response = messages[-1]
        if hasattr(final_response, 'content') and final_response.content:
            print("\n====最终回复====")
            print(final_response.content)
    
    # 关闭数据库连接
    conn.close()
    
    print("\n" + "=" * 50)
    print("自动化函数调用演示完成")
    print("\n相比原始手动实现的改进：")
    print("1. 使用函数注册表集中管理可调用函数")
    print("2. 自动生成工具定义，减少重复代码")
    print("3. 自动处理函数调用流程，包括参数解析、函数执行和结果处理")
    print("4. 支持多个函数的注册和调用")
    print("5. 异常处理更加完善")
    print("6. 代码结构更清晰，便于维护和扩展")

if __name__ == "__main__":
    run_automated_demo()
