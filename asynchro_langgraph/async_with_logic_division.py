#async.py
from dotenv import load_dotenv
import operator
from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

load_dotenv()

class State(TypedDict):
  aggregate: Annotated[list, operator.add]
  which: str



class ReturnNodeValue:
  def __init__(self, node_secret: str):
    self._value = node_secret

  def __call__(self, state: State) -> Any:
    print(f"Adding {self._value} to {state['aggregate']}")
    return {"aggregate": [self._value]}

#async로 호출할 graph를 만드는데 아래의 함수에는 async 함수로 선언된 것이 없음을 주목할 것
#a에서 갈라지는 분기가 3개, 그리고 각각의 조건에 따라 b, c/ c,d로 나뉘고, 그 뒤에 e를 실행하게 될 것

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm a"))
builder.add_edge(START, "a")
builder.add_node("b", ReturnNodeValue("I'm b"))
builder.add_node("c", ReturnNodeValue("I'm c"))
builder.add_node("d", ReturnNodeValue("I'm d"))
builder.add_node("e", ReturnNodeValue("I'm e"))

def route_bc_or_cd(state: State) -> Sequence[str]:
  if state["which"] == "cd":
    return ["c", "d"]
  return ["b", "c"]

intermediates=["b", "c", "d"]

builder.add_conditional_edge("a",
                             route_bc_or_cd,
                             intermediates # 만약 이것을 넣지 않으면 LangGraph에서는 a 에서 다른 모든 노드로 이동할 수 있다고 판단하여 바로 end나, e로 가는 경로를 conditional_edge에 추가하게 됨 > 즉, conditional_edge로 갈 수 있는 node의 가능성을 여기에 제한해두는 것
                             )
for node in intermediates:
  builder.add_edge(node, "e")
builder.add_edge("e", END)


builder.add_edge("a", route_bc_or_cd)


graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="async_graph.png")


if __name__ == "__main__":
  print("Hello Async Graph")
  graph.invoke({"aggregate": [], "which": "cd"}, {"configurable": {"thread_id": "foo"}})
