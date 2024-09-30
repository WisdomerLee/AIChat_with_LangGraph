import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

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


def bot(state: State):
    print(state["messages"])
    return {"messages": [model.invoke(state["messages"])]}
#

#
graph_builder = StateGraph(State)
graph_builder.add_node("bot", bot) # node를 langgraph에 넣을 땐 이름과, 그 이름에 연결되는 함수, agent 등을 넣을 것

graph_builder.set_entry_point("bot") # bot 노드를 처음 으로 설정
graph_builder.set_finish_point("bot") # bot 노드를 끝으로 설정

# 실행 전에 compile
graph = graph_builder.compile()

res = graph.invoke({"messages": ["안녕, 반가워. 질문을 하나 해도 될까?"]})
print(res["messages"])

# 아래와 같이 처리하면
# 사용자의 입력을 받아서
# 받는대로 stream으로 즉시 출력하며 응답이 오게 됨
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# 검색 도구로 Tavily를 활용
# 검색 엔진에 AI가 붙은 것
