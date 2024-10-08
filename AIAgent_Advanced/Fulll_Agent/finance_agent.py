# 프로젝트 폴더 내부에 간단한 csv 파일로 재무재표 데이터가 있다고 가정하고 프로젝트를 진행
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
from tavily import TavilyClient
from pydantic import BaseModel # 강의에서는 langchain_core.pydantic_v1을 사용하나, langchain이 업데이트 되며 pydantic자체를 활용할 수 있게 바뀌고 있음

# agent의 상태를 기억하기 위해 아래와 같이 먼저 선언
memory = SqliteSaver.from_conn_string(":memory:")

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

tavily = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-4o-mini"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

tavily = TavilyClient(api_key=tavily)

# LangGraph에서 공통으로 사용하게 될 class, 해당 클래스는 langgraph의 로직마다 모두 다 다르게 적용될 수 있음
class AgentState(TypedDict):
  task: str
  competitors: List[str]
  csv_file: str
  financial_data: str
  analysis: str
  competitor_data: str
  comparison: str
  feedback: str
  report: str
  content: List[str]
  revision_number: int
  max_revision: int

# 사용자의 요청을 모두 저장할 타입
class Queries(BaseModel):
  queries: List[str]

# 노드 등의 agent들에서 사용하게 될 prompt들 < langsmith에 들어가서 그곳에서 본인에게 맞는 프롬프트를 찾아서 적용하거나 해당 프롬프트를 수정하여 아래에 각각 적용할 것
GATHER_FINANCIALS_PROMPT = """"""
ANALYZE_DATA_PROMPT = """"""
RESEARCH_COMPETITORS_PROMPT = """"""
COMPETE_PERFORMANCE_PROMPT = """"""

FEEDBACK_PROMPT = """"""
WRITE_REPORT_PROMPT = """"""
RESEARCH_CRITIQUE_PROMPT = """"""

# 노드에서 돌려주는 것들은 모두 langgraph로 빌드될 state의 이름, 데이터 타입이 모두 일치하는 형태로 있어야 함

def gather_financials_node(state: AgentState):
  csv_file = state.get("csv_file")
  df = pd.read_csv(StringIO(csv_file))

  financial_data_str = df.to_string(index=False)

  combined_content = (
    f"{state.get("task")}\n\n Here is the financial data:\n\n{financial_data_str}"
  )
  messages = [
    SystemMessage(content=GATHER_FINANCIALS_PROMPT),
    HumanMessage(content=combined_content),
  ]
  response = model.invoke(messages)
  return {"financial_data": response.content}

def analyze_data_node(state: AgentState):
  messages = [
    SystemMessage(content=ANALYZE_DATA_PROMPT),
    HumanMessage(content=state.get("financial_data")),
  ]
  response = model.invoke(messages)
  return {"analysis": response.content}

def research_competitors_node(state: AgentState):
  content = state.get("content", [])
  for competitor in state.get("competitors"):
    queries = model.with_structured_output(Queries).invoke(
      [
        SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
        HumanMessage(content=competitor),
      ]
    )
    for q in queries.queries:
      response = tavily.search(query=q, max_results=2)
      for r in response.get("results"):
        content.append(r.get("content"))
  return {"content": content}

def compare_performance_node(state: AgentState):
  content = "\n\n".join(state.get("content", []))
  user_message = HumanMessage(
    content=f"{state.get('task')}\n\nHere is the financial analysis {state.get('analysis')}"
  )
  messages = [
    SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
    user_message,
  ]
  response = model.invoke(messages)
  return {
    "comparison": response.content,
    "revision_number": state.get("revision_number", 1) + 1
  }

def research_critique_node(state: AgentState):
  queries = model.with_structured_output(Queries).invoke(
    [
      SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
      HumanMessage(content=state.get("feedback")),
    ]
  )
  content = state.get("content", [])
  for q in queries.queries:
    response = tavily.search(query=q, max_results=2)
    for r in response.get("results"):
      content.append(r.get("content"))
  return {"content": content}

def collect_feedback_node(state: AgentState):
  messages = [
    SystemMessage(content=FEEDBACK_PROMPT),
    HumanMessage(content=state.get("comparison")),
  ]
  response = model.invoke(messages)
  return {"feedback": response.content}

def write_report_node(state: AgentState):
  messages = [
    SystemMessage(content=WRITE_REPORT_PROMPT),
    HumanMessage(content=state.get("comparison")),
  ]
  response = model.invoke(messages)
  return {"report": response.content}

def should_continue(state):
  if state.get("revision_number") > state.get("max_revisions"):
    return END
  return "collect_feedback"


builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)
builder.add_node("write_report", write_report_node)

builder.set_entry_point("gather_financials")

builder.add_conditional_edges(
  "compare_performance",
  should_continue,
  {END: END, "collect_feedback": "collect_feedback"},
)

builder.add_edge("gather_financials", "analyzedata")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile(checkpointer=memory)

==============Console Testing============================
def read_csv_file(file_path):
  with open(file_path, "r") as file:
    print("Reading CSV file...")
    return file.read()


if __name__ == "__main__":
  task = "Analyze the financial performance of our () company"
  competitors = ["Microsoft", "Nvidia", "Google"]
  csv_file_path = (
    "./data/financials.csv"
  )

if not os.path.exists(csv_file_path):
  print(f"csv file not found at {csv_file_path}")
else:
  print("Starting the conversation")
  csv_data = read_csv_file(csv_file_path)
  initial_state = {
    "task": task,
    "competitors": competitors,
    "csv_file": csv_data,
    "max_revisions": 2,
    "revision_number": 1,
  }
  thread = {"configurable": {"thread_id": "1"}} # 사용자 구분 필요

for s in graph.stream(initial_state, thread):
  print(s)
============ Console Testing===============================

===========Streamlit UI==========================
import streamlit as st

def main():
  st.title("Financial Performance Reporting Agent")
  
  task = st.text_input(
    "Enter the task",
    "Analyze the financial performance of our company (MYAICo.AI) compared to competitors",
  )
  competitors = st.text_area("Enter competitor names (one per line):").split("\n")
  max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
  upload_file = st.file_uploader(
    "Upload a CSV file with the company's financial data", type=["csv"], 
  )

if st.button("Start Analysis") and uploaded_file is not None:
  csv_data = uploaded_file.getvalue().decode("utf-8")
  initial_state={
    "task": task,
    "competitors": competitors,
    "csv_file": csv_data,
    "max_revisions": 2,
    "revision_number": 1,
  }
  thread = {"configurable": {"thread_id": "1"}}

final_state = None
for s in graph.stream(initial_state, thread):
  st.write(s)
  final_state = s
if final_state and "report" in final_state:
  st.subheader("Final Report")
  st.write(final_state.get("report"))

if __name__ == "__main__":
  main()

  

=================================================

# agent는 최적화 할 수 있는 기술이 다양한 편
# 모델 자체를 최적화하는 방법
# 보다 효과적인 모델을.. 더 작은 단위들로 쪼개어 진행하기 - LLM 모델 변경
# Model Quantization
# Fine- Tuning
# Parallel Processing
# Distributed computing
# Batch processing
# Caching, Preprocessing
# Data caching
# Preprocessing
