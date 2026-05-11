from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

document_content = ""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def tool_update(content: str):
    """Tool to update the document content"""
    global document_content
    document_content = content

    return {'messages': f'\nDocument has been updated! Current content: {document_content}\n'}


def tool_save(filename: str):
    """Tool to save the document in a text file"""

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    global document_content

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
            print(f'Document saved successfully! Filename: {filename}')
            return f'Document saved successfully! Filename: {filename}'
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [tool_update, tool_save]

llm = ChatOllama(model="qwen3").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = input("\nReady when you are: ")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input('\nWhat would you like to do with this document?  ')
        print(f'\nUser: {user_input}')
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "exit"
    
    return "continue"
        
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", agent)

tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_edge("agent","tools")

graph.add_conditional_edges(
    'tools',
    should_continue,
    {
        'continue': 'agent',
        "exit": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()