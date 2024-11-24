from typing import Dict

from google.generativeai.types import generation_types
from llama_index.core.base.llms.types import ChatResponse, ChatMessage


def transform_gemini_response_to_chat_response(response: generation_types.GenerateContentResponse) -> ChatResponse:
    """
    Transforms a GenerateContentResponse object into a ChatResponse object.

    :param response: The GenerateContentResponse object to be transformed.
    :return: A ChatResponse object.
    """
    # Extract top candidate
    if not response or not response.candidates:
        raise ValueError("No candidates available in the response.")

    top_candidate = response.candidates[0]
    content_parts = top_candidate.content.parts

    # Combine parts into a single string for content
    content = ""
    for part in content_parts:
        if isinstance(part, dict) and "function_call" in part:
            function_call = part["function_call"]
            args = function_call.get("args", {})
            content += str(args)
        else:
            content += str(part)

    # Create ChatMessage instance
    message = ChatMessage(
        role=top_candidate.content.role,
        content=content,
    )

    # Transform data into ChatResponse
    chat_response = ChatResponse(
        message=message,
        raw=response,
        delta=None,  # Add delta if provided in the response
        logprobs=None,  # Add logprobs if needed
    )

    return chat_response