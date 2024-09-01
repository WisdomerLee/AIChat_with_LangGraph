#async.py
from dotenv import load_dotenv
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

load_dotenv()

class State(TypedDict):
  aggregate: Annotated[list, operator.add]


class ReturnNodeValue:
  def __init__(self, node_secret: str):
    self._value = node_secret

  def __call__(self, state: State) -> Any:
    print(f"Adding {self._value} to {state['aggregate']}")
    return {"aggregate": [self._value]}

#async로 호출할 graph를 만드는데 아래의 함수에는 async 함수로 선언된 것이 없음을 주목할 것
#a에서 갈라지는 분기가 2개!! > Langgraph에서 알아서 동시에 처리하게 함!
#우리는 graph의 분기만 n개로 나누어주면 됨

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm a"))
builder.add_edge(START, "a")
builder.add_node("b", ReturnNodeValue("I'm b"))
builder.add_node("b2", ReturnNodeValue("I'm b2"))
builder.add_node("c", ReturnNodeValue("I'm c"))
builder.add_node("d", ReturnNodeValue("I'm d"))
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="async_graph.png")


if __name__ == "__main__":
  print("Hello Async Graph")
  graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
