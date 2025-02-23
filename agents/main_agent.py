import json
import re
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agents.stark_agent import stark_agent
from config.config import config
from state.shared_state import shared_state
from utils.llm import llm_manager


class MainAgent:
    def __init__(self):
        try:
            # Define routing tools using StructuredTool
            self.routing_tools = [
                StructuredTool.from_function(
                    func=self.route_to_stark_agent,
                    name="stark_evaluation_agent",
                    description="""Use for evaluating LLM retrieval on semi-structured knowledge bases.
                    Best for: benchmarking retrieval systems, evaluating knowledge graphs, computing metrics""",
                ),
            ]

            # Create the agent using create_react_agent
            self.agent = create_react_agent(
                model=llm_manager.llm,
                tools=self.routing_tools,
                messages_modifier=config.MAIN_AGENT_PROMPT,
            )

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def route_to_stark_agent(self, query: str) -> Dict[str, Any]:
        """Route queries to StarkQA evaluation agent.

        Args:
            query: The user's query about evaluating retrieval performance.

        Returns:
            Dict containing the response from the StarkQA agent.
        """
        try:
            shared_state.set(config.StateKeys.CURRENT_AGENT, config.AgentNames.STARK)
            result = stark_agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            return {
                "status": "success",
                "response": (
                    result["messages"][-1].content
                    if result.get("messages")
                    else "No response from StarkQA agent"
                ),
            }
        except Exception as e:
            return {"status": "error", "response": f"Error in StarkQA agent: {str(e)}"}

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input state through the agent"""
        try:
            return self.agent.invoke(state)
        except Exception as e:
            return {"error": str(e), "response": f"Error in main agent: {str(e)}"}


# Create a global instance
main_agent = MainAgent()
