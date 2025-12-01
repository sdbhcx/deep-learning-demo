#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
函数调用自动化实现演示

本文件演示如何将NLG.py中的手动函数调用改进为更自动化的方式，包括：
1. 函数注册表模式
2. 自动参数解析
3. 动态函数调用
4. 错误处理和重试机制
"""

import os
import json
import time
from typing import Dict, Callable, Any, List, Optional
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
        self.function_registry: Dict[str, Callable] = {}
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    
    def register_function(self, name: str, func: Callable) -> None:
        """
        注册一个函数到函数注册表
        
        Args:
            name: 函数名称，需要与OpenAI工具定义中的name一致
            func: 要注册的函数对象
        """
        self.function_registry[name] = func
    
    def create_function_schema(self, name: str, description: str, parameters: Dict) -> Dict:
        """
        创建OpenAI工具调用所需的函数schema
        
        Args:
            name: 函数名称
            description: 函数描述
            parameters: 函数参数定义
            
        Returns:
            格式化的函数schema字典
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
    
    def call_function(self, function_name: str, args: Dict) -> Any:
        """
        根据函数名和参数调用已注册的函数
        
        Args:
            function_name: 要调用的函数名称
            args: 函数参数字典
            
        Returns:
            函数执行结果
            
        Raises:
            ValueError: 如果函数未注册
        """
        if function_name not in self.function_registry:
            raise ValueError(f"函数 {function_name} 未注册")
        
        return self.function_registry[function_name](**args)
    
    def process_tool_call(self, messages: List[Dict], tool_call: Dict, max_retries: int = 3) -> List[Dict]:
        """
        处理工具调用，执行函数并将结果添加到消息列表
        
        Args:
            messages: 消息列表
            tool_call: 工具调用信息
            max_retries: 最大重试次数
            
        Returns:
            更新后的消息列表
        """
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        print(f"====调用函数: {function_name} ====")
        print(f"参数: {arguments}")
        
        # 带重试机制的函数调用
        retry_count = 0
        while retry_count <= max_retries:
            try:
                result = self.call_function(function_name, arguments)
                print(f"函数执行结果: {result}")
                break
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"函数调用失败，已达最大重试次数: {str(e)}")
                    result = f"Error: {str(e)}"
                else:
                    wait_time = 2 ** retry_count  # 指数退避
                    print(f"函数调用失败，{wait_time}秒后重试 ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
        
        # 将工具执行结果添加到消息列表
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": str(result)
        })
        
        return messages
    
    def get_completion_with_function_calling(
        self, 
        messages: List[Dict], 
        tools: List[Dict], 
        max_tool_calls: int = 5,
        model: str = "gpt-3.5-turbo-1106"
    ) -> Dict:
        """
        获取完成结果，自动处理函数调用
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tool_calls: 最大工具调用次数，防止无限循环
            model: 使用的模型
            
        Returns:
            最终的模型响应
        """
        # 创建一个消息副本，避免修改原始消息
        current_messages = messages.copy()
        tool_call_count = 0
        
        while tool_call_count < max_tool_calls:
            # 获取模型响应
            response = self.client.chat.completions.create(
                model=model,
                messages=current_messages,
                temperature=0,
                tools=tools
            )
            
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if message.tool_calls:
                tool_call_count += 1
                print(f"\n====检测到工具调用 #{tool_call_count} ====")
                # 处理第一个工具调用（简单实现，只处理第一个）
                tool_call = message.tool_calls[0]
                # 将模型生成的消息添加到消息列表
                current_messages.append(message)
                # 处理工具调用
                current_messages = self.process_tool_call(current_messages, tool_call)
                # 获取最终回复
                response = self.client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    temperature=0
                )
                message = response.choices[0].message
            
            # 如果没有工具调用，返回结果
            if not message.tool_calls:
                return message
        
        raise Exception(f"达到最大工具调用次数 ({max_tool_calls})")

# 模拟数据库查询函数
def ask_database(query: str) -> str:
    """
    模拟数据库查询函数
    
    Args:
        query: SQL查询语句
        
    Returns:
        查询结果字符串
    """
    print(f"执行SQL: {query}")
    # 这里应该是实际的数据库查询
    # 出于演示目的，返回模拟结果
    return "[(1, '模拟查询结果')]"

# 创建一个更灵活的数据库查询函数，支持更多操作
def database_operation(operation: str, query: str, params: Optional[Dict] = None) -> Dict:
    """
    更灵活的数据库操作函数
    
    Args:
        operation: 操作类型 (query, update, insert, delete)
        query: SQL语句
        params: 可选参数字典
        
    Returns:
        包含操作结果的字典
    """
    print(f"执行数据库操作: {operation}")
    print(f"SQL: {query}")
    print(f"参数: {params or '无'}")
    
    # 实际应用中应根据operation执行不同的数据库操作
    return {
        "success": True,
        "operation": operation,
        "affected_rows": 1,
        "data": [(1, "模拟数据")] if operation == "query" else None
    }

# 使用装饰器简化函数注册
def register_with_function_caller(function_caller: FunctionCaller, name: Optional[str] = None):
    """
    用于注册函数到FunctionCaller的装饰器
    
    Args:
        function_caller: FunctionCaller实例
        name: 注册的函数名，默认为函数本身的名称
        
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        register_name = name or func.__name__
        function_caller.register_function(register_name, func)
        return func
    return decorator

def run_demo():
    """运行函数调用自动化演示"""
    print("函数调用自动化实现演示")
    print("=" * 50)
    
    # 1. 创建FunctionCaller实例
    function_caller = FunctionCaller()
    
    # 2. 注册函数
    function_caller.register_function("ask_database", ask_database)
    function_caller.register_function("database_operation", database_operation)
    
    # 使用装饰器注册函数
    @register_with_function_caller(function_caller, "greet")
    def greet_user(name: str, greeting: Optional[str] = "Hello") -> str:
        return f"{greeting}, {name}!"
    
    # 3. 定义工具schema
    tools = [
        function_caller.create_function_schema(
            name="ask_database",
            description="查询数据库",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL查询语句"
                    }
                },
                "required": ["query"]
            }
        ),
        function_caller.create_function_schema(
            name="greet",
            description="向用户问好",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "用户名"
                    },
                    "greeting": {
                        "type": "string",
                        "description": "问候语"
                    }
                },
                "required": ["name"]
            }
        )
    ]
    
    # 4. 定义消息
    messages = [
        {"role": "system", "content": "你是一个助手，根据用户的问题决定是否调用函数。"},
        {"role": "user", "content": "查询所有用户数据"}
    ]
    
    # 5. 自动处理函数调用
    try:
        response = function_caller.get_completion_with_function_calling(messages, tools)
        print("\n====最终回复====")
        print(response.content)
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    print("\n" + "=" * 50)
    print("函数调用自动化演示完成")
    print("\n实现特点：")
    print("1. 函数注册表模式，集中管理可调用函数")
    print("2. 自动参数解析和函数调用")
    print("3. 内置重试机制，提高稳定性")
    print("4. 装饰器支持，简化函数注册")
    print("5. 灵活的工具schema生成")
    print("6. 支持多轮函数调用处理")

if __name__ == "__main__":
    run_demo()
