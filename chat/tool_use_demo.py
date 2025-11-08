import os
import re
from typing import TypedDict, Annotated, List, Dict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

# 配置 OpenAI API
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1.6-250615")




# 定义状态结构
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    steps_taken: int
    final_answer:str


# 定义实际可用的工具
@tool
def calculate(expression: str) -> str:
    """进行简单的数学计算，只支持加减乘除四则运算。

    参数:
    - expression: 数学表达式字符串，例如 "2 + 3 * 4"
    """
    # 安全地计算简单的数学表达式，只允许数字和四则运算符
    try:
        # 验证表达式只包含数字和四则运算符
        if not re.match(r'^[\d\s+\-*/.()]+$', expression):
            return "表达式只能包含数字和加减乘除运算符"

        # 使用eval计算表达式（在受控环境下）
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 创建工具集合
tools = [calculate]


# 创建工具节点
tool_node = ToolNode(tools)


# 定义节点函数
def think_node(state: AgentState) -> AgentState:
    """思考节点：分析问题并决定是否使用工具"""
    # 构建提示
    prompt = ChatPromptTemplate.from_template(
        """
        你是一个助手，可以使用工具来帮助回答问题。
        请分析当前问题，决定是否需要使用工具。如果需要，请选择合适的工具和参数。

        可用工具：
        - calculate(expression): 进行数学计算
        """)

    # 让LLM决定是否使用工具
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt.format()),
        ]+state.messages
    )

    new_state = state.model_copy()
    new_state.messages = [response]
    new_state.steps_taken = state.steps_taken + 1

    return new_state


# 路由器：决定下一个节点
def router(state: AgentState) -> str:
    """路由器：根据LLM的响应决定是使用工具还是直接回答"""
    # 检查是否达到最大步骤数
    if state.steps_taken >= 5:
        return "answer"

    # 获取最后一条消息
    last_message = state.messages[-1]

    # 检查是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 没有工具调用，直接回答
    return "answer"


def answer_node(state: AgentState) -> AgentState:
    """回答节点：生成最终答案"""
    # 获取最后一条消息
    last_message = state.messages[-1]

    new_state = state.model_copy()

    # 如果已经有内容，直接使用
    if hasattr(last_message, "content") and last_message.content:
        new_state.final_answer = last_message.content
    else:
        # 否则生成一个答案
        prompt = ChatPromptTemplate.from_template(
            "根据已有信息，回答用户问题：{question}"
        )

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": state["question"]})
        new_state.final_answer = answer

    return new_state


# 创建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("think", think_node)
workflow.add_node("tools", tool_node)
workflow.add_node("answer", answer_node)

# 添加边
workflow.add_edge(START, "think")
workflow.add_edge("tools", "think")  # 工具调用后回到思考节点
workflow.add_edge("answer", END)

# 添加条件边
workflow.add_conditional_edges(
    "think",
    router,
    {
        "tools": "tools",
        "answer": "answer"
    }
)

# 编译图
app = workflow.compile()

# 运行演示
if __name__ == "__main__":
    print("基于LangGraph的工具调用Demo")
    print("=" * 50)

    # 示例问题
    example_question = "计算 3 + 5 * 2 - 4 / 2 的结果"

    # 运行示例
    print(f"\n示例 : {example_question}")

    # 初始状态
    initial_state = AgentState(
        steps_taken=0,
        messages=[HumanMessage(content=example_question)],
        final_answer=""
    )

    # 运行图
    result = app.invoke(initial_state)

    # 打印结果
    print(f"最终答案: {result['final_answer']}")
    print(f"步骤数: {result['steps_taken']}")
    print("-" * 50)
