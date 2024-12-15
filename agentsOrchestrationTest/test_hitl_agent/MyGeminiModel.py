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
from llama_index.core.utilities.gemini_utils import ROLES_FROM_GEMINI
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.utils import _error_if_finished_early
from llama_index.llms.openai.utils import resolve_tool_choice, OpenAIToolCall
import google.generativeai as genai

from agentsOrchestrationTest.test_hitl_agent.GeminiTools import GeminiTools


def request_transfer() -> None:
    """Used to indicate that your job is done and you would like to transfer control to another agent."""
    pass


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
        # tool_specs = [tool.metadata. for tool in tools]

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

    @staticmethod
    def _prepare_chat_with_agent_tools(
            tools: List["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        tool_specs = [GeminiTools.to_gemini_tool(tool.metadata) for tool in tools]
        print("tools specs : ", tool_specs)

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        new_messages = ''
        for chat in chat_history:
            new_messages += str(GeminiTools.to_gemini_message_dict(chat))

        print("chat history : ", new_messages)

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
        chat_kwargs = self._prepare_chat_with_agent_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
        )

        response = await self.my_complete(chat_kwargs.get("contents"), chat_kwargs.get("tools"))
        print(response.raw)
        role = ROLES_FROM_GEMINI[response.raw["content"]["role"]]

        return ChatResponse(message=ChatMessage(role=role, content=str(response.raw['content'])), raw=response.raw)

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

    def get_tool_calls_from_response(
            self,
            response: "ChatResponse",
            error_on_no_tool_call: bool = True,
            **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""

        tool_calls = response.raw["content"]["parts"]

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            # args={}
            # for key in tool_call["function_call"]["args"]:
            #     args[key] = float(tool_call["function_call"]["args"][key])
            # print("new args: ",args)
            tool_selections.append(
                ToolSelection(
                    tool_id="123",
                    tool_name=tool_call["function_call"]["name"],
                    tool_kwargs=tool_call["function_call"]["args"],
                )
            )

        return tool_selections

    def get_agent_calls_from_response(
            self,
            response: "ChatResponse",
            error_on_no_tool_call: bool = True,
            **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""

        tool_calls = GeminiTools.extract_tool_calls(response)

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
