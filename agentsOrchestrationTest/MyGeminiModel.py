from typing import Sequence, Optional, Union, List, Any, Dict, get_args

from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.tools import BaseTool
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai.utils import resolve_tool_choice, OpenAIToolCall


class MyGeminiModel(Gemini,FunctionCallingLLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Union[str, dict] = "auto",
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        if self.metadata.is_function_calling_model:
            for tool_spec in tool_specs:
                if tool_spec["type"] == "function":
                    tool_spec["function"]["strict"] = strict
                    tool_spec["function"]["parameters"][
                        "additionalProperties"
                    ] = False  # in current openai 1.40.0 it is always false.

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice) if tool_specs else None,
            **kwargs,
        }



    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, get_args(OpenAIToolCall)):
                raise ValueError("Invalid tool_call object")
            if tool_call.type != "function":
                raise ValueError("Invalid tool type. Unsupported by OpenAI")

            # this should handle both complete and partial jsons
            try:
                argument_dict = parse_partial_json(tool_call.function.arguments)
            except ValueError:
                argument_dict = {}

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections