from pydantic import BaseModel, Field,ConfigDict
from llama_index.core.tools import BaseTool



class AgentConfig(BaseModel):
    """Used to configure an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[BaseTool] | None = None
    tools_requiring_human_confirmation: list[str] = Field(default_factory=list)