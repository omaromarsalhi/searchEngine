import json
from typing import Sequence, Optional, Union, List, Any, Dict, get_args

from google.generativeai.types import FunctionLibraryType
from llama_index.core.base.llms.generic_utils import completion_response_to_chat_response
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse, CompletionResponse
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.tools import BaseTool
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.utils import _error_if_finished_early
from llama_index.llms.openai.utils import resolve_tool_choice, OpenAIToolCall
import google.generativeai as genai


def add_two_numbers(a: int, b: int) -> int:
    """Used to add two numbers together."""
    print("banana")
    return a + b


def get_order_status(order_id: str) -> str:
    """Fetches the status of a given order ID."""
    # Mock data for example purposes
    order_statuses = {
        "12345": "Shipped",
        "67890": "Processing",
        "11223": "Delivered"
    }
    return order_statuses.get(order_id, "Order ID not found.")


class MyGeminiModel(Gemini, FunctionCallingLLM):

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

        # if self.metadata.is_function_calling_model:
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

    def _prepare_chat_with_My_tools(
            self,
            tools: List["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        new_messages=''
        for chat in chat_history:
            new_messages += str(self.to_gemini_message_dict(chat))

        return {
            "contents": new_messages,
            "tools": tool_specs,
        }

    async def my_achat_with_tools(
            self,
            tools: List["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
            verbose: bool = False,
            allow_parallel_tool_calls: bool = False,
            **kwargs: Any,
    ) -> ChatResponse:
        """Async chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

        response = await self.achat(**chat_kwargs)

        return self._validate_chat_with_tools_response(
            response,
            tools,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

    async def my_achat_with_tools_for_agent_test(
            self,
            tools: List["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
    ) -> ChatResponse:
        """Async chat with function calling."""
        chat_kwargs = self._prepare_chat_with_My_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
        )
        print("my content: ",chat_kwargs.get("contents"))
        response = await self.my_complete(chat_kwargs.get("contents"), chat_kwargs.get("tools"), )
        print("response from the ai: ", response.raw)

        return ChatResponse(message=ChatMessage())

    @llm_completion_callback()
    async def my_complete(
            self, content, tools
    ) -> CompletionResponse:
        result = self._model.generate_content(contents=content, tools=tools)
        return self.my_completion_from_gemini_response(result)

    @staticmethod
    def my_completion_from_gemini_response(response: genai.types.GenerateContentResponse) -> CompletionResponse:

        top_candidate = response.candidates[0]
        _error_if_finished_early(top_candidate)

        raw = {
            **(type(top_candidate).to_dict(top_candidate)),
            **(type(response.prompt_feedback).to_dict(response.prompt_feedback)),
        }
        if response.usage_metadata:
            raw["usage_metadata"] = type(response.usage_metadata).to_dict(
                response.usage_metadata
            )
        return CompletionResponse(text="function calling return", raw=raw)

    @staticmethod
    def to_gemini_message_dict(message: ChatMessage, drop_none: bool = False, ):
        """Convert generic message to Gemini message dict."""
        message_dict = {
            "role": message.role.value,
            "content": message.content,
        }

        null_keys = [key for key, value in message_dict.items() if value is None]
        # if drop_none is True, remove keys with None values
        if drop_none:
            for key in null_keys:
                message_dict.pop(key)

        return message_dict  # type: ignore

    def get_tool_calls_from_response(
            self,
            response: "ChatResponse",
            error_on_no_tool_call: bool = True,
            **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""

        tool_calls = self.extract_tool_calls(response)

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            # this should handle both complete and partial jsons
            try:
                argument_dict = parse_partial_json(tool_call.get('function').get('arguments'))

            except ValueError:
                argument_dict = {}

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.get('id'),
                    tool_name=tool_call.get('function').get("name"),
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    def extract_tool_calls(self, chat_response):
        chat_response_str = chat_response.model_dump_json()

        chat_response_str = chat_response_str.replace('assistant: ```json', '').replace('```', '').strip()

        try:
            response_dict = json.loads(chat_response_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

        content_str = response_dict.get("message", {}).get("content", "")

        if isinstance(content_str, str) and content_str.startswith('json'):
            try:
                content_json = json.loads(content_str[4:])
                tool_calls = content_json.get("message", {}).get("additional_kwargs", {}).get("tool_calls", [])
                return tool_calls
            except json.JSONDecodeError as e:
                print(f"Error decoding content JSON: {e}")
                return []
        else:
            print("No valid content to parse.")
            return []
