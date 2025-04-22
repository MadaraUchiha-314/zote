
import logging
import sys

from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
import random

# Configure the logger to output to stdout with INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s\n",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


@tool
def sample_tool_1() -> Command[Literal["sample_node_1", "sample_node_2"]]:
    """Sample tool 1"""
    logger.info("Inside sample_tool_1")
    random_number = random.randint(1, 100)
    logger.info(f"Generated random number: {random_number}")

    if random_number % 2 == 0:
        return Command(goto="sample_node_1")
    else:
        return Command(goto="sample_node_2")

class AgentState(MessagesState):
    pass

class MillibelleAgent:
    
    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("start", self.start)
        graph.add_node("tool", ToolNode([sample_tool_1]))
        graph.add_node("sample_node_1", self.sample_node_1)   
        graph.add_node("sample_node_2", self.sample_node_2)   
        graph.add_node("end", self.end)

        graph.set_entry_point("start")
        graph.add_edge("start", "tool")
        graph.add_edge("sample_node_1", "end")
        graph.add_edge("sample_node_2", "end")
        graph.set_finish_point("end")

        self.graph = graph.compile(checkpointer=MemorySaver())


    def start(self, state: AgentState) -> AgentState:
        logger.info("Inside start node")
        state["messages"].append(SystemMessage(content="Start!"))
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sample_tool_1",
                        "args": {},
                        "id": "123123123",
                        "type": "tool_call",
                    }
                ],
            )
        )
        return state
    
    def sample_node_1(self, state: AgentState) -> AgentState:
        logger.info("Inside sample node 1")
        return state
    
    def sample_node_2(self, state: AgentState) -> AgentState:
        logger.info("Inside sample node 2")
        return state

    def end(self, state: AgentState) ->  AgentState:
        logger.info("Inside end node")
        return state
    
    def invoke(self, input, config):
        return self.graph.invoke(input=input, config=config)
    

if __name__ == "__main__":
    agent = MillibelleAgent()
    config = {"configurable": {"thread_id": "0"}}

    response = agent.invoke(input={"messages": []}, config=config)
    logger.info(f"response from agent is: {response}")