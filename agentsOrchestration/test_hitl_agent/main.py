import configparser
import os

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import StartEvent
from llama_index.llms.gemini import Gemini

from agentsOrchestration.test_hitl_agent.MyGeminiModel import MyGeminiModel
from agentsOrchestration.test_hitl_agent.AgentConfig import AgentConfig
from agentsOrchestration.test_hitl_agent.HITLAgent import ToolRequestEvent, ProgressEvent, ToolApprovedEvent, \
    OrchestratorAgent
import asyncio

from agentsOrchestration.utils import FunctionToolWithContext

config = configparser.ConfigParser()
config.read("../../config.ini")
os.environ["GOOGLE_API_KEY"] = config.get('API', 'gemini_key')


def add_two_numbers(a: int, b: int) -> int:
    """Used to add two numbers together."""
    print("banana")
    return a + b


add_two_numbers_tool = FunctionToolWithContext.from_defaults(fn=add_two_numbers)

agent_configs = AgentConfig(
    name="Addition Agent",
    description="Used to add two numbers together.",
    system_prompt="You are an agent that adds two numbers together. Do not help the user with anything else.",
    tools=[add_two_numbers_tool],
    tools_requiring_human_confirmation=[],
)

llm = MyGeminiModel(model_name="models/gemini-1.5-flash-latest")


workflow = OrchestratorAgent()

async def main():
    handler = workflow.run(
        agent_configs=[agent_configs],
        user_msg="What is 10 + 10?",
        chat_history=[],
        initial_state={"user_name": "Logan"},
        llm=llm,
    )
    # Stream and process events
    async for event in handler.stream_events():
        print(f"Event type: {type(event)}")
        if isinstance(event, ProgressEvent):
            print("omar1")
            # Handle progress events
            print(event.msg)
        elif isinstance(event, ToolRequestEvent):
            print("omar2")
            # Handle tool request events
            print(f"Tool {event.tool_name} requires human approval. Approving!")
            handler.ctx.send_event(ToolApprovedEvent(
                approved=True,
                tool_name=event.tool_name,
                tool_id=event.tool_id,
                tool_kwargs=event.tool_kwargs,
            ))

    # Separator for clarity
    print("-----------")

    # Await the final result of the handler
    final_result = await handler
    print(final_result["response"])


if __name__ == "__main__":
    asyncio.run(main())
