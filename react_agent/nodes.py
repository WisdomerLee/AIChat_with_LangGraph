from dotenv import load_dotenv
from langgraph.prebuilt.tool_executor import ToolExecutor

from react import react_agent_runnable, tools
# react_agent_runnable : 추론 도구로 llm을 활용할 것
# tools는 agent가 사용하게 될 도구들
from state import AgentState
# 위에서 지정한 AgentState를 사용할 것
# 각 노드는 agent의 상태를 입력으로 받음


load_dotenv()

# 우리는 노드가 입력을 받고, react prompt에 결과값을 주기를 원함, 해당 결과값은 agent가 끝나거나 agent의 행동 오브젝트
# 그리고 상태를 갱신하기를 원함

def run_agent_reasoning_engine(state: AgentState):
  agent_outcome = react_agent_runnable.invoke(state)
  return {"agent_outcome": agent_outcome}

tool_executor = ToolExecutor(tools)

def execute_tools(state: AgentState):
  agent_action = state["agent_outcome"]
  output = tool_executor.invoke(agent_action)

  return {"intermediate steps": [(agent_action, str(output))]}
