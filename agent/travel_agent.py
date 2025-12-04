import os
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID
from langchain.memory import ConversationTokenBufferMemory
from langchain.tools.render import render_text_description
from langchain.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatZhipuAI
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

def search_train_ticket(
    origin: str,
    destination: str,
    date: str,
    departure_time_start: str,
    departure_time_end: str,
) -> List[dict[str, str]]:
    """
    搜索火车票
    :param origin: 出发站
    :param destination: 到达站
    :param date: 日期
    :param departure_time_start: 出发时间开始
    :param departure_time_end: 出发时间结束
    :return: 火车票列表
    """
    # 模拟搜索结果
    return [
        {
            "train_number": "G123",
            "origin": "北京",
            "destination": "上海",
            "date": "2023-12-01",
            "departure_time": "08:00",
            "arrival_time": "12:00",
            "price": "500元",
            "seat_type": "商务座"
        },
        {
            "train_number": "G124",
            "origin": "北京",
            "destination": "上海",
            "date": "2023-12-01",
            "departure_time": "12:00",
            "arrival_time": "16:00",
            "price": "500元",
            "seat_type": "商务座"
        },
    ]

def purchase_train_ticket(train_number: str) -> dict:
    """购买火车票"""
    return {
        "result": "success",
        "message": "购买成功",
        "data": {
            "train_number": "G1234",
            "seat_type": "商务座",
            "seat_number": "7-17A"
        }
    }

search_train_ticket_tool = StructuredTool.from_function(
    func=search_train_ticket,
    name="查询火车票",
    description="查询指定日期可用的火车票。",
)
purchase_train_ticket_tool = StructuredTool.from_function(
    func=purchase_train_ticket,
    name="购买火车票",
    description="购买火车票。会返回购买结果(result)，和座位号(seat_number)",
)
finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)

tools = [search_train_ticket_tool, purchase_train_ticket_tool, finish_placeholder]

prompt_text = """  
你是强大的AI火车票助手，可以使用工具与指令查询并购买火车票  

你的任务是：  
{task_description}  

你可以使用以下工具或指令，它们又称为动作或actions：  
{tools}  

当前的任务执行记录：  
{memory}  

按照以下格式输出：  

任务：你收到的需要执行的任务  
思考：观察你的任务和执行记录，并思考你下一步应该采取的行动  
然后，根据以下格式说明，输出你选择执行的动作/工具：  
{format_instructions}  
"""

final_prompt = """
你的任务是:
{task_description}
{memory}
你已经完成任务。
现在请根据上述结果简要总结出你的最终答案。
直接给出答案。不用再解释或分析你的思考过程。
"""

class Action(BaseModel):
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工具或指令参数，由参数名称和参数值组成")

class MyPrintHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
    def on_llm_new_token(
      self,
      token: str,
      *,
      chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
      run_id: str,
      parent_run_id: Optional[UUID] = None,
      **kwargs: Any,
    ) -> Any:
        end = ""
        content = token + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return token
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> Any:
        end = "\n"
        content = end
        sys.stdout.write(content)
        sys.stdout.flush()
        return response

class MyAgent:
    def __init__(
        self,
        llm:BaseChatModel = ChatZhipuAI(
            model="glm-4",  # 智谱AI的标准模型名称
            temperature=0.7,
            api_key=os.environ.get("GLM_API_KEY"),
        ),
        tools: Optional[List[StructuredTool]] = None,
        prompt: str = "",
        final_prompt: str = "",
        max_thought_steps: Optional[int] = 10,
    ):
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.final_prompt = PromptTemplate.from_template(final_prompt)
        self.max_thought_steps = max_thought_steps
        self.output_parser = PydanticOutputParser[Action](pydantic_object=Action)
        self.prompt = self.__init_prompt(prompt)
        self.llm_chain = self.prompt | self.llm | self.output_parser
        self.verbose_printer = MyPrintHandler()
    
    @staticmethod
    def __chinese_friendly(string) -> str:
        lines = string.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('{') and line.endswith('}'):
                try:
                    lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
                except:
                    pass
        return '\n'.join(lines)
    
    def __init_prompt(self, prompt: str) -> PromptTemplate:
        """
        初始化提示模板，填充固定参数
        :param prompt: 原始提示模板字符串
        :return: 初始化后的PromptTemplate对象
        """
        return PromptTemplate.from_template(prompt).partial(
            tools=render_text_description(self.tools),
            format_instructions=self.__chinese_friendly(
                self.output_parser.get_format_instructions()
            )
        )

    def run(self, task_description):
        thought_step_count = 0
        agent_memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=4000)
        agent_memory.save_context({"input": "\ninit"}, {"output": "\n开始"})
        while thought_step_count < self.max_thought_steps:
            print(f">>>Round: {thought_step_count}<<<<")
            action, response = self.__step(task_description, agent_memory)
            if action.name == "FINISH":
                break
            observation = self.__exec_action(action)
            print(f"---\nObservation:\n{observation}")
            self.__update_memory(agent_memory, response, observation)
            thought_step_count += 1
        if thought_step_count >= self.max_thought_steps:
            reply = "任务未完成！"
        else:
            final_chain = self.final_prompt | self.llm | StrOutputParser()
            reply = final_chain.invoke({"task_description": task_description, "memory": agent_memory})
        return reply

    def __step(self, task_description, memory) -> Tuple[Action, str]:
        # 由于llm_chain已经包含output_parser，stream会直接返回Action对象
        # 我们需要分离原始响应和解析后的动作
        raw_response = ""
        action = None
        
        # 重新构建只包含prompt和llm的链来获取原始文本响应
        text_chain = self.prompt | self.llm
        
        # 先获取原始文本响应
        for chunk in text_chain.stream({"task_description": task_description, "memory": memory}, config={"callbacks": [self.verbose_printer]}):
            raw_response += chunk.content
        
        # 然后使用output_parser解析原始响应
        try:
            action = self.output_parser.parse(raw_response)
        except Exception as e:
            print(f"解析错误: {e}")
            print(f"原始响应: {raw_response}")
            # 如果解析失败，返回一个默认的FINISH动作
            action = Action(name="FINISH", args=None)
        
        return action, raw_response

    def __exec_action(self, action: Action) -> str:
        """
        执行动作，返回观测结果
        :param action: 要执行的动作
        :return: 动作执行的观测结果（字符串格式）
        """
        if action.name == "FINISH":
            return "任务完成！"
        elif action.name in [tool.name for tool in self.tools]:
            tool = next(tool for tool in self.tools if tool.name == action.name)
            result = tool.run(action.args)
            # 将工具返回的字典或列表转换为JSON字符串
            if isinstance(result, (dict, list)):
                import json
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        else:
            return f"未知动作：{action.name}"

    @staticmethod
    def __update_memory(memory: ConversationTokenBufferMemory, response: str, observation: str):
        """
        更新代理记忆，添加新的交互记录
        :param memory: 代理记忆对象
        :param response: 模型生成的响应
        :param observation: 执行动作后的观测结果
        """
        memory.save_context({"input": response}, {"output": observation})

if __name__ == "__main__":
    agent = MyAgent(
        tools=tools,
        prompt=prompt_text,
        final_prompt=final_prompt
    )
    task_description = "帮我买2023年12月1日早上去上海的火车票"
    reply = agent.run(task_description)
    print(reply)