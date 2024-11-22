from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.workflow import Event, Workflow, Context, StartEvent, StopEvent, step
from llama_index.core.workflow.events import InputRequiredEvent, HumanResponseEvent
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.tools import ToolSelection
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

from MyGeminiModel import MyGeminiModel
from agentsOrchestration.test_hitl_agent import AgentConfig
from agentsOrchestration.test_hitl_agent.CustomFunctionCallingLLM import  CustomFunctionCallingLLM


class LLMCallEvent(Event):
    pass


class ToolCallEvent(Event):
    pass


class ToolCallResultEvent(Event):
    chat_message: ChatMessage


class ProgressEvent(Event):
    msg: str


# Our two new events!
class ToolRequestEvent(InputRequiredEvent):
    tool_name: str
    tool_id: str
    tool_kwargs: dict


class ToolApprovedEvent(HumanResponseEvent):
    tool_name: str
    tool_id: str
    tool_kwargs: dict
    approved: bool
    response: str | None = None


class HITLAgent(Workflow):

    @step
    async def setup(
            self, ctx: Context, ev: StartEvent
    ) -> LLMCallEvent:
        """Sets up the workflow, validates inputs, and stores them in the context."""
        agent_config = ev.get("agent_config")
        user_msg = ev.get("user_msg")
        llm: LLM = ev.get("llm", default=MyGeminiModel(api_key=""))
        chat_history = ev.get("chat_history", default=[])
        initial_state = ev.get("initial_state", default={})

        if (
                user_msg is None
                or llm is None
                or chat_history is None
        ):
            raise ValueError(
                "User message, llm, and chat_history are required!"
            )

        # if not llm.metadata.is_function_calling_model:
        #     raise ValueError("LLM must be a function calling model!")

        await ctx.set("agent_config", agent_config)
        await ctx.set("llm", llm)

        chat_history.append(ChatMessage(role="user", content=user_msg))
        await ctx.set("chat_history", chat_history)

        await ctx.set("user_state", initial_state)

        return LLMCallEvent()

    @step
    async def speak_with_agent(
            self, ctx: Context, ev: LLMCallEvent
    ) -> ToolCallEvent | StopEvent:
        """Speaks with the active sub-agent and handles tool calls (if any)."""
        # Setup the agent
        agent_config: AgentConfig = await ctx.get("agent_config")
        chat_history = await ctx.get("chat_history")
        llm = await ctx.get("llm")

        user_state = await ctx.get("user_state")
        user_state_str = "\n".join([f"{k}: {v}" for k, v in user_state.items()])
        system_prompt = (
                agent_config.system_prompt.strip()
                + f"\n\nHere is the current user state:\n{user_state_str}"
        )

        llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history
        tools = agent_config.tools

        response = await llm.achat_with_tools(tools, chat_history=llm_input)

        tool_calls: list[ToolSelection] = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if len(tool_calls) == 0:
            chat_history.append(response.message)
            await ctx.set("chat_history", chat_history)
            return StopEvent(
                result={
                    "response": response.message.content,
                    "chat_history": chat_history,
                }
            )

        await ctx.set("num_tool_calls", len(tool_calls))

        # New logic for HITL
        for tool_call in tool_calls:
            if tool_call.tool_name in agent_config.tools_requiring_human_confirmation:
                ctx.write_event_to_stream(
                    ToolRequestEvent(
                        prefix=f"Tool {tool_call.tool_name} requires human approval.",
                        tool_name=tool_call.tool_name,
                        tool_kwargs=tool_call.tool_kwargs,
                        tool_id=tool_call.tool_id,
                    )
                )
            else:
                ctx.send_event(
                    ToolCallEvent(tool_call=tool_call, tools=agent_config.tools)
                )

        chat_history.append(response.message)
        await ctx.set("chat_history", chat_history)

    @step
    async def handle_tool_approval(
            self, ctx: Context, ev: ToolApprovedEvent
    ) -> ToolCallEvent | ToolCallResultEvent:
        """Handles the approval or rejection of a tool call."""
        if ev.approved:
            agent_config = await ctx.get("agent_config")
            return ToolCallEvent(
                tools=agent_config.tools,
                tool_call=ToolSelection(
                    tool_id=ev.tool_id,
                    tool_name=ev.tool_name,
                    tool_kwargs=ev.tool_kwargs,
                ),
            )
        else:
            return ToolCallResultEvent(
                chat_message=ChatMessage(
                    role="tool",
                    content=ev.response or "Tool call was not approved.",
                )
            )

    @step(num_workers=4)
    async def handle_tool_call(
            self, ctx: Context, ev: ToolCallEvent
    ) -> ToolCallResultEvent:
        """Handles the execution of a tool call."""
        tool_call = ev.tool_call
        tools_by_name = {tool.metadata.get_name(): tool for tool in ev.tools}

        tool_msg = None

        tool = tools_by_name.get(tool_call.tool_name)
        additional_kwargs = {
            "tool_call_id": tool_call.tool_id,
            "name": tool.metadata.get_name(),
        }
        if not tool:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Tool {tool_call.tool_name} does not exist",
                additional_kwargs=additional_kwargs,
            )

        try:
            tool_output = await tool.acall(**tool_call.tool_kwargs)

            tool_msg = ChatMessage(
                role="tool",
                content=tool_output.content,
                additional_kwargs=additional_kwargs,
            )
        except Exception as e:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Encountered error in tool call: {e}",
                additional_kwargs=additional_kwargs,
            )

        ctx.write_event_to_stream(
            ProgressEvent(
                msg=f"Tool {tool_call.tool_name} called with {tool_call.tool_kwargs} returned {tool_msg.content}"
            )
        )

        return ToolCallResultEvent(chat_message=tool_msg)

    @step
    async def aggregate_tool_results(
            self, ctx: Context, ev: ToolCallResultEvent
    ) -> LLMCallEvent:
        """Collects the results of all tool calls and updates the chat history."""
        num_tool_calls = await ctx.get("num_tool_calls")
        results = ctx.collect_events(ev, [ToolCallResultEvent] * num_tool_calls)
        if not results:
            return

        chat_history = await ctx.get("chat_history")
        for result in results:
            chat_history.append(result.chat_message)
        await ctx.set("chat_history", chat_history)

        return LLMCallEvent()