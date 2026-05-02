from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()
llm = ChatOllama(model='gemma3:4b')

class AgentState(TypedDict):
    message: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['message'])
    print(response.content)

    return state

graph = StateGraph(AgentState)
graph.add_node("bot", process)

graph.set_entry_point("bot")
graph.set_finish_point("bot")

agent = graph.compile()

user_input = input("Enter message: ")
while user_input != 'exit':
    response = agent.invoke({'message': [HumanMessage(content=user_input)]})
    user_input = input("Enter message: ")
