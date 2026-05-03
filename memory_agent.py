from typing import List, Dict, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="gemma3:4b")

def process(state: AgentState) -> AgentState:
    """Processor"""
    response = llm.invoke(state['message'])
    state['message'].append(AIMessage(response.content))
    print(f'\n AI Message: {response.content}\n')
    print(f'\n\nCURRENT STATE: {state["message"]}\n\n')

graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.set_entry_point('process')
graph.set_finish_point('process')

agent = graph.compile()

conv_history = []

user_input = input('Human Message: ')
while user_input != 'exit':
    conv_history.append(HumanMessage(content=user_input))
    result = agent.invoke({'message': conv_history})
    #print(result['message'])
    conv_history = result['message']

    user_input = input('Human Message: ')


with open('logging.txt', 'w') as file:
    file.write('Conversation Log: \n')
    for messages in conv_history:
        if(isinstance(messages, HumanMessage)):
            file.write(f'Human Message: {messages.content}')
        else:
            file.write(f'AI Message: {messages.content}')
    file.write('\nEnd of conversation\n')

print('Convo logged')