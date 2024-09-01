from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver #기억을 저장하기 위해 필요

class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
  print("---Step 1---")

def human_feedback(state: State) -> None:
  print("---Human Feedback---")

def step_3(state: State) -> None:
  print("---Step 3---")


builder = StateGraph(State)

builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"]) #여기서 interrupt_before라는 부분이 들어가는데, 다음으로 진행하기 전에 일시적으로 멈추고 사용자의 입력을 받을 수 있음 - human_feedback이라는 노드를 실행하기 직전에 멈추는 구간이 있음을 알려줌

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
