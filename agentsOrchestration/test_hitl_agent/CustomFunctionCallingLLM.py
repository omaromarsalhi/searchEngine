from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse, LLMMetadata, \
    ChatResponseGen, CompletionResponseGen, ChatResponseAsyncGen, CompletionResponseAsyncGen
from typing import List, Union, Optional, Any, Sequence


class CustomFunctionCallingLLM(FunctionCallingLLM):
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        pass

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        pass

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        pass

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        pass

    @property
    def metadata(self) -> LLMMetadata:
        pass

    def _prepare_chat_with_tools(
        self,
        tools: List[Any],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> dict:
        # Prepare the chat with tools logic
        return {
            "tools": tools,
            "user_msg": user_msg,
            "chat_history": chat_history,
            "verbose": verbose,
            "allow_parallel_tool_calls": allow_parallel_tool_calls,
            **kwargs
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List[Any],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        # Validate the chat response logic
        return response

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Implement the chat logic
        return ChatResponse(messages=[ChatMessage(role="system", content="Response from LLM")])

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Implement the async chat logic
        return ChatResponse(messages=[ChatMessage(role="system", content="Async response from LLM")])

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Implement the completion logic
        return CompletionResponse(text="Completion response from LLM")

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Implement the async completion logic
        return CompletionResponse(text="Async completion response from LLM")
