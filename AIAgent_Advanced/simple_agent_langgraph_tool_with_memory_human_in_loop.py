# Check point라는 것을 이용하여 대화의 맥락을 기억하도록 할 것
# thread_id라는 것으로 구분되어, 각각의 checkpoint마다 별도로 맥락을 관리할 수 있음!
# checkpointer는 state를 주기적으로 확인하고, state를 저장


import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

import json
from langchain_core.messages import ToolMessage

from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver # 해당 library는 langgraph와 별도로 따로 추가 설치를 해야 사용 가능 > pip install langgraph-checkpoint-sqlite



load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-4o-mini"

model = ChatOpenAI(api_key=openai_key, model=llm_name)

# 기본적인 chatbot 만들기!

# 그래프에 저장할 데이터 타입을 명시한 state 만들기 !!!
class State(TypedDict):
    # Messages 는 "list"의 데이터 타입을 갖고 있음, 'add_messages'함수를 이용하여 새로운 데이터가 들어오면 해당 함수를 이용하여 list에 저장
    # annotation은 이 state에 값이 들어올 때 어떻게 처리할 지를 알려주는 역할을 수행
    # 아래의 보기에서는 messages의 list에 새로 들어온 메시지를 더하는 형태로 동작
    # 기존의 대화에 새 대화가 계속 추가되므로, agent는 사용자의 기존 대화 내역을 모두 알고 이에 대해 답변할 수 있게 됨
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 도구 생성
tool = TavilySearchResults(max_results=2)
tools = [tool]
# 생성된 도구는 넘길 수 있어야 함

# rest = tool.invoke("질문 거리")
# print(rest)

# 모델에 도구를 전달하는 과정을 아래의 한 줄로 처리할 수 있음!
# OpenAI와 같은 LLM은 add_tools로
# ChatOpenAI와 같은 Chat model은 bind_tools로 처리해야 하는 것을 제외하고는 동일함
model_with_tools = model.bind_tools(tools)

# res = model_with_tools.invoke("")
# print(res)
# 확인을 위해 잠시 처리한 것

# 이제 저 도구 달린 모델을 graph에 넣어야 함
# 도구에 대한 설명도 같이 집어넣어주어야 어떻게 활용할지 결정함

# 그것을 위해 아래와 같이 사용자가 직접 노드를 담당할 클래스를 만드는 것도 가능
# class BasicToolNode:
#    """A node that runs the tools requested in the last AIMessage"""

#    def __init__(self, tools: list) -> None:
#        self.tools_by_name = {tool.name: tool for tool in tools}

#    def __call__(self, inputs: dict):
#        if messages := inputs.get("messages", []):
#            message = messages[-1]
#        else:
#            raise ValueError("No message found in input")
#        outputs = []
#        for tool_call in message.tool_calls:
#            tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                tool_call["args"]
#            )
#            outputs.append(
#                ToolMessage(
#                    content=json.dumps(tool_result),
#                    name=tool_call["name"],
#                    tool_call_id=tool_call["id"]
#                )
#            )
#        return {"messages": outputs}

# 하지만 langgraph의 내장된 도구 방식을 사용한다면..? > from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# 도구 역할을 하는 부분을 그래프에 더하기
# tool_node = BasicToolNode(tools=[tool])
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


#def route_tools(
#    state: State,
#) -> Literal["tools", "__end__"]:
#    """
#    Use in the conditional_edge to route to the ToolNode if the la
#    has tool calls. Otherwise, route to the end.
#    """
#    if isinstance(state, list):
#        ai_message = state[-1]
#    elif messages := state.get("messages",[]):
#        ai_message = messages[-1]
#    else:
#        raise ValueError(f"No messages found in input state to tool")
#    if hasattr(ai_message, "tool_calls") and len(ai_message.tool):
#        return "tools"
#    else:
#        return "__end__"


#graph_builder.add_conditional_edges(
#    "bot",
#    route_tools,
#    {"tools": "tools", "__end__": "__end__"},
#)

# 위의 주석 처리된 부분들은 내장된 함수를 사용하면서 쓰지 않아도 되는 것 - 매우 편리해짐

graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

graph_builder.add_node("bot", bot)


graph_builder.set_entry_point("bot")


memory = SqliteSaver.from_conn_string(":memory:") # 내부에 데이터 경로를 집어넣으면 해당 데이터 경로에 저장, memory로 설정하면 일시적으로 memory에 저장하여 사용함


# 실행 전에 compile
graph = graph_builder.compile(
  checkpointer=memory,
  interrupt_before=["tools"], # 도구 호출 전에 잠시 멈추고 사용자 입력을 받을 것
) 

# thread_id를 지정해야 사용자별로 구분하여 사용 가능
config = {
  "configurable": {"tread_id": 1} # thread id 지정!!
}

user_input = ""

evnets = graph.stream(
  {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
  event["messages"][-1].pretty_print()

# snapshot = memory.get_snapshot() 
snapshot = graph.get_state(config)

next_step = snapshot.next # 다음 단계를 실행하도록 하는 것

print(next_step) # 여기에 사용자의 입력을 받도록 지정하도록 루프를 짤 것

existing_message = snapshot.values["messages"][-1]
all_tools = existing_message.tool_calls

