import json
from typing import Dict, Any

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import ToolMetadata


class GeminiTools:
    @staticmethod
    def to_gemini_tool(tool: ToolMetadata, skip_length_check: bool = False) -> Dict[str, Any]:
        """To OpenAI tool."""
        if not skip_length_check and len(tool.description) > 1024:
            raise ValueError(
                "Tool description exceeds maximum length of 1024 characters. "
                "Please shorten your description or move it to the prompt."
            )
        function_declaration = {
            "function_declarations": [{
                "name": tool.name,
                "description": tool.description,
            }]
        }

        parameters = GeminiTools.get_parameters_dict(tool)
        if parameters:
            function_declaration["function_declarations"][0]["parameters"] = parameters

        return function_declaration

    @staticmethod
    def get_parameters_dict(tool: ToolMetadata) -> dict:
        if tool.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = tool.fn_schema.model_json_schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required", "definitions"]
            }
            if parameters["properties"] == {}:
                return {}
            else:
                for item in parameters["properties"].values():
                    del item["title"]

        return parameters

    @staticmethod
    def to_gemini_message_dict(message: ChatMessage, drop_none: bool = True, ):
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

    @staticmethod
    def extract_tool_calls(chat_response):
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
