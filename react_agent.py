from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def tool_add(a: int, b:int) -> int:
    """Tool that takes two numbers as input and adds them"""
    return a+b

@tool
def tool_sub(a:int, b:int) -> int:
    """Tool that takes two numbers as input and subtracts them"""
    return a-b

tools = []
tools.append(tool_add)
tools.append(tool_sub)

llm = ChatOllama(model="llama3.2:3b").bind_tools(tools)

def process(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a mathematician. Give answers only when you are confident, do not assume anything")
    
    response = llm.invoke(
        [system_prompt] + list(state["messages"])  
    )
    return {'messages': response}

def should_continue(state: AgentState) -> str:
    """Function to decide if tool calling should continue"""
    last_messgae = state["messages"][-1]

    if not last_messgae.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", process)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node","agent")

graph.set_entry_point("agent")
graph.set_finish_point("agent")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages']
        if isinstance(message, tuple):
            print(message)
        else:
            for m in message:
                m.pretty_print()

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "what is 34 added to 21? What is 5-10?"}
    ]
})

print_stream(agent.stream(result, stream_mode="values"))