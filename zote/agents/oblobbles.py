import logging
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

# Configure the logger to output to stdout with INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s\n",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


class AgentState(MessagesState):
    pass


class OblobblesAgent:

    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("start", self.start)
        graph.add_node("human", self.human)
        graph.add_node("end", self.end)

        graph.set_entry_point("start")
        graph.add_edge("start", "human")
        graph.add_edge("human", "end")
        graph.set_finish_point("end")

        self.graph = graph.compile(checkpointer=MemorySaver())

    def start(self, state: AgentState):
        state["messages"].append(SystemMessage(content="Start!"))
        return state

    def end(self, state: AgentState):
        state["messages"].append(SystemMessage(content="End!"))
        return state

    def human(self, state: AgentState):
        logger.info(f"Inside human node. State is: {state}")
        response = "static response. This should not be returned by the graph"
        # Get the response from human
        logger.info("Before 1st interrupt")
        response = interrupt("Hello Oblobbles!")
        logger.info("After 1st interrupt")
        # After the 1st interrupt is resumed, the following piece of code will be executed
        # But since the node interrupts again, the state is not returned/saved by the graph
        state["messages"].append(HumanMessage(content=response))
        # Get the response from human. Again!
        logger.info("Before 2nd interrupt")
        response = interrupt("Hello again Human!")
        logger.info("After 2nd interrupt")
        state["messages"].append(HumanMessage(content=response))
        return state

    def invoke(self, *args, **kwargs):
        return self.graph.invoke(*args, **kwargs)


if __name__ == "__main__":
    agent = OblobblesAgent()
    config = {"configurable": {"thread_id": "0"}}
    response = agent.invoke(input={"messages": []}, config=config)
    logger.info(f"response from agent is: {response}")
    logger.info(f"graph state: {agent.graph.get_state(config=config)}")
    response = agent.invoke(Command(resume="Hello Oblobbles!"), config=config)
    logger.info(f"response from agent is: {response}")
    logger.info(f"graph state: {agent.graph.get_state(config=config)}")
    response = agent.invoke(Command(resume="Hello again Oblobbles!"), config=config)
    logger.info(f"response from agent is: {response}")
    logger.info(f"graph state: {agent.graph.get_state(config=config)}")
