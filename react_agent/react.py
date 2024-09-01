from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

react_prompt: PromptTemplate = hub.pull("hwchase17/react")
# 위의 프롬프트는 추론을 위해 많이 쓰이는 프롬프트 템플릿 중 하나
# 해당 프롬프트에서 입력받는 변수는 tools, tool_names, input, agent_scratchpad등이 있음
# agent_scratchpad는 agent에서 주고받은 짤막한 채팅 히스토리 내역

# LLM을 추론 도구로 쓰기 위해서 reactagent를 도입하는데, 도입할 때 LangGraph의 형태로 도입할 것

@tool
def triple(num:float) -> float:
  """
  :param num: a number to triple
  :return: the number tripled -> multiplied by 3
  """
  return 3 * float(num)

tools = [TavilySearchResults(max_results=1), triple]

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

react_agent_runnable = create_react_agent(llm, tools, react_prompt)
