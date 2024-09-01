from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
# state에 대한 scheme을 정의해야 함 -> dictionary 타입으로 정의해야 함
from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

AGENT_REASON = "agent_reason"
ACT = "act"

#agent가 모든 동작을 완료한 상태인지 확인하고 그 상태가 아니라면 행동을 반복하도록 아래의 함수를 설정
def should_continue(state: AgentState) -> str:
  if isinstance(state["agent_outcome"], AgentFinish):
    return END
  else:
    return ACT

flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, execute_tools)

flow.add_conditional_edges(AGENT_REASON, should_continue,)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
#완성된 그래프의 형태를 그림으로! 꼭 이 형태로 확인해서 그래프가 사용자가 생각한대로 짜여있는지 확인할 것

if __name__ == "__main__":
  print("")
  res = app.invoke(
      input={
          "input": "질문거리"
      }
  )
  print(res["agent_outcome"].return_values["output"])
