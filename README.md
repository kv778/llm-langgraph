# LangGraph and Local LLMs

This repository contains the below:

### Graphs

Five graphs, each isolating a single LangGraph pattern. These don't necessarily call an LLM — they exist to make the control-flow concepts concrete.
 
| # | Concept | What it demonstrates |
|---|---------|----------------------|
| 1 | **Hello World graph** | The simplest possible single-node graph — defining state, one node, and compiling/invoking the graph. |
| 2 | **Multiple inputs** | A node that processes a list of values held in state and produces an aggregate result. |
| 3 | **Sequential graph** | Chaining multiple nodes so they run in order, each one reading and writing the shared state. |
| 4 | **Conditional graph** | Using conditional edges to route execution down different branches based on the current state. |
| 5 | **Looping graph** | Building cycles in the graph so a node (or set of nodes) repeats until a stopping condition is met. |
 
Each of these has a corresponding exercise in the course; my solutions live alongside the walkthrough implementations.
 
### AI agents
 
This is where a local LLM (qwen3) running via Ollama gets plugged into the graph structure. Four agents, increasing in capability:
 
- **Simple Bot** — a single-turn conversational agent. Takes a user message, sends it to the LLM, returns the response. The "hello world" of LLM-in-a-graph.
- **Memory Bot** — extends the simple bot with conversation history, accumulating messages in state (via `add_messages`) so the model has the full context of the dialogue across turns.
- **ReAct Agent** — a reasoning-and-acting agent that can call **tools**. The LLM decides when to invoke a tool, the graph routes to a tool node to execute it, and the result is fed back to the model — looping until the model has what it needs to answer.
- **Drafter** — a document-writing assistant that helps the user create and iteratively update a document using `update` and `save` tools, demonstrating a focused, tool-driven agent with a clear task and termination condition.
