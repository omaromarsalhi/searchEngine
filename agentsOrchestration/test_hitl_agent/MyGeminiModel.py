from typing import Sequence, Optional, Union, List, Any, Dict, get_args
from google.generativeai.types import content_types, generation_types, GenerateContentResponse
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse, CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.tools import BaseTool
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.utils import completion_from_gemini_response
from llama_index.llms.openai.base import force_single_tool_call
from llama_index.llms.openai.utils import resolve_tool_choice, OpenAIToolCall
from agentsOrchestration.test_hitl_agent.utils import transform_gemini_response_to_chat_response


class MyGeminiModel(Gemini):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def provide_tool(self, prompt: Optional[Union[str, ChatMessage]] = None,
                     tools: Optional[ChatMessage] | None = None,
                     **kwargs: Any) -> ChatResponse:

        response = self._model.generate_content(prompt, tools=tools, **kwargs)
        return transform_gemini_response_to_chat_response(response)

    def achat_with_tools(
            self,
            contents: Optional[Union[str, ChatMessage]],
            tools: content_types.FunctionLibraryType | None = None,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs: Any,
    ) -> GenerateContentResponse:
        """Chat with function calling."""

        messages = chat_history or []
        if contents:
            chat_message = ChatMessage(role="user", content=contents)
            messages.append(chat_message)

        response =  self.provide_tool(
            prompt=contents,
            tools=tools,
            **kwargs,
        )

        print(response)
        return response

    def _validate_chat_with_tools_response(
            self,
            response: ChatResponse,
            tools: List["BaseTool"],
            allow_parallel_tool_calls: bool = False,
            **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
            self,
            response: GenerateContentResponse,
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

# from typing import Sequence, Optional, Union, List, Any, Dict, get_args
#
# from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
# from llama_index.core.llms.function_calling import FunctionCallingLLM
# from llama_index.core.llms.llm import ToolSelection
# from llama_index.core.llms.utils import parse_partial_json
# from llama_index.core.tools import BaseTool
# from llama_index.llms.gemini import Gemini
# from llama_index.llms.openai.utils import resolve_tool_choice, OpenAIToolCall
#
#
# class MyGeminiModel(Gemini,FunctionCallingLLM):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def _prepare_chat_with_tools(
#         self,
#         tools: List["BaseTool"],
#         user_msg: Optional[Union[str, ChatMessage]] = None,
#         chat_history: Optional[List[ChatMessage]] = None,
#         verbose: bool = False,
#         allow_parallel_tool_calls: bool = False,
#         tool_choice: Union[str, dict] = "auto",
#         strict: Optional[bool] = None,
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
#         """Predict and call the tool."""
#         print(tools)
#         tool_specs = [tool.metadata.to_openai_tool() for tool in tools]
#         print(tool_specs)
#
#         # if self.metadata.is_function_calling_model:
#         for tool_spec in tool_specs:
#             if tool_spec["type"] == "function":
#                 tool_spec["function"]["strict"] = strict
#                 tool_spec["function"]["parameters"][
#                     "additionalProperties"
#                 ] = False  # in current openai 1.40.0 it is always false.
#
#         if isinstance(user_msg, str):
#             user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
#
#         messages = chat_history or []
#         if user_msg:
#             messages.append(user_msg)
#
#         print(resolve_tool_choice(tool_choice))
#         print(tool_choice)
#         return {
#             "messages": messages,
#             "tools": tool_specs or None,
#             "tool_choice": resolve_tool_choice(tool_choice) if tool_specs else None,
#             **kwargs,
#         }
#
#
#
#     def get_tool_calls_from_response(
#         self,
#         response: "ChatResponse",
#         error_on_no_tool_call: bool = True,
#         **kwargs: Any,
#     ) -> List[ToolSelection]:
#         """Predict and call the tool."""
#         tool_calls = response.message.additional_kwargs.get("tool_calls", [])
#
#         if len(tool_calls) < 1:
#             if error_on_no_tool_call:
#                 raise ValueError(
#                     f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
#                 )
#             else:
#                 return []
#
#         tool_selections = []
#         for tool_call in tool_calls:
#             if not isinstance(tool_call, get_args(OpenAIToolCall)):
#                 raise ValueError("Invalid tool_call object")
#             if tool_call.type != "function":
#                 raise ValueError("Invalid tool type. Unsupported by OpenAI")
#
#             # this should handle both complete and partial jsons
#             try:
#                 argument_dict = parse_partial_json(tool_call.function.arguments)
#             except ValueError:
#                 argument_dict = {}
#
#             tool_selections.append(
#                 ToolSelection(
#                     tool_id=tool_call.id,
#                     tool_name=tool_call.function.name,
#                     tool_kwargs=argument_dict,
#                 )
#             )
#
#         return tool_selections
