import operator
from typing import Annotated, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
  input: str
  agent_outcome: Union[AgentAction, AgentFinish, None]
  intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] #agent가 취할 수 있는 행동과 그 결과?를 묶은 것, 또한 annotated type은 add라는 함수로 표시
  #위에 표시된 것은 langgraph에 이 리스트를 중간 단계로 집어넣을 것을 알려줌, 이것은 react prompt에서 agent scratchpad에 들어감
  #agent의 상태가 업데이트 되고, node에서 값을 돌려받으면 특정 속성을 덮어쓰게 됨



