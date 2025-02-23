import json
import re
import traceback
from typing import Any, Dict, List, Type, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, create_react_agent

from config.config import config
from state.shared_state import shared_state
from tools.stark import stark_tools  # This will be our tools list
from tools.stark.evaluate_retrieval import evaluate_stark_retrieval
from utils.llm import llm_manager


class StarkAgent:
    def __init__(self):
        try:
            print("Initializing StarkQA Agent...")

            # Create the agent using create_react_agent
            self.agent = create_react_agent(
                model=llm_manager.llm,
                tools=stark_tools,  # List of available tools
                messages_modifier=config.STARK_AGENT_PROMPT,
            )

            print("StarkQA Agent initialized successfully")

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input state through the agent"""
        try:
            shared_state.set(config.StateKeys.CURRENT_AGENT, config.AgentNames.STARK)
            return self.agent.invoke(state)
        except Exception as e:
            return {"error": str(e), "response": f"Error in StarkQA agent: {str(e)}"}


# Create a global instance
stark_agent = StarkAgent()
