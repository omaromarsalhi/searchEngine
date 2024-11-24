import asyncio
import configparser
import os
from llama_index.core.memory import ChatMemoryBuffer
from agentsOrchestration.MyGeminiModel import MyGeminiModel
from agentsOrchestration.main import get_account_balance_tools
from workflow import (
    AgentConfig,
    ConciergeAgent,
    ProgressEvent,
    ToolRequestEvent,
    ToolApprovedEvent,
)
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from utils import FunctionToolWithContext
import mysql.connector



config = configparser.ConfigParser()
config.read("../config.ini")
os.environ["GOOGLE_API_KEY"] = config.get('API', 'gemini_key')


def get_initial_state() -> dict:
    return {
        "username": None,
        "session_token": None,
        "account_id": None,
        "account_balance": None,
    }


def get_authentication_tools() -> list[BaseTool]:
    # async def is_authenticated(ctx: Context) -> bool:
    #     """Checks if the user has a session token."""
    #     ctx.write_event_to_stream(ProgressEvent(msg="Checking if authenticated"))
    #     user_state = await ctx.get("user_state")
    #     return user_state["session_token"] is not None
    #
    async def store_username(ctx: Context, first_name: str, last_name: str) -> None:
        """Adds the username to the user state."""
        ctx.write_event_to_stream(ProgressEvent(msg="Recording username"))
        user_state = await ctx.get("user_state")
        user_state["username"] = first_name +'_'+ last_name
        await ctx.set("user_state", user_state)

    async def login(ctx: Context) -> str:
        """Given a full name, logs in and stores a session token in the user state."""

        file_name = "example.txt"

        # Open the file in write mode (creates the file if it doesn't exist)
        with open(file_name, "w") as file:
            # Write content to the file
            file.write("Hello, this is a new file created with Python!\n")
            file.write("You can write multiple lines like this.\n")

        user_state = await ctx.get("user_state")
        username = user_state["username"]

        # Connect to MySQL database
        # ctx.write_event_to_stream(ProgressEvent(msg=f"connecting to database"))
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='rag_db'
        )

        if connection.is_connected():
            # ctx.write_event_to_stream(ProgressEvent(msg=f"Connected to MySQL database"))
            cursor = connection.cursor(dictionary=True)

            # Query to search for user
            query = """
                    SELECT * FROM customer
                    WHERE first_name = %s AND last_name = %s
                    """
            cursor.execute(query, (username.split('_')[0], username.split('_')[1]))
            # Fetch results
            user = cursor.fetchone()
            ctx.write_event_to_stream(ProgressEvent(msg=f"Fetch results"))
            if user:
                session_token = "1234567890"
                user_state["session_token"] = session_token
                user_state["account_id"] = "123"
                user_state["account_balance"] = 1000
                await ctx.set("user_state", user_state)
                cursor.close()
                connection.close()
                return f"Logged in user {username} with session token {session_token}. They have an account with id {user_state['account_id']} and a balance of ${user_state['account_balance']}."
            else:
                cursor.close()
                connection.close()
                return f" user {username} not found."

    return [
        FunctionToolWithContext.from_defaults(async_fn=store_username),
        FunctionToolWithContext.from_defaults(async_fn=login),
        # FunctionToolWithContext.from_defaults(async_fn=is_authenticated),
    ]


def get_agent_configs() -> list[AgentConfig]:
    return [
        AgentConfig(
            name="Authentication Agent",
            description="Handles user authentication",
            system_prompt="""
You are a helpful assistant that is authenticating a user.
Your authentication is not like the standard one so there is no need for password just first and last name of the user are enough
Your task is to get a valid session token stored in the user state.
To do this, the user must supply you with a firstname and a lastname. You can ask them to supply these.
If the user supplies a firstname and a lastname, call the tool "login" to log them in.
Once the user is logged in and authenticated, you can transfer them to another agent.
            """,
            tools=get_authentication_tools(),
            tools_requiring_human_confirmation=["login"]
        )
        ,
        AgentConfig(
            name="Account Balance Agent",
            description="Checks account balances",
            system_prompt="""
    You are a helpful assistant that is looking up account balances.
    The user may not know the account ID of the account they're interested in,
    so you can help them look it up by the name of the account.
    The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
    If they aren't authenticated, tell them to authenticate first and call the "RequestTransfer" tool.
    If they're trying to transfer money, they have to check their account balance first, which you can help with.
                """,
            tools=get_account_balance_tools(),
        ),
    ]



async def main():
    """Main function to run the workflow."""

    from colorama import Fore, Style

    # llm = OpenAI(model="gpt-4o", temperature=0.4)
    llm = MyGeminiModel()
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    initial_state = get_initial_state()
    agent_configs = get_agent_configs()
    workflow = ConciergeAgent(timeout=None)

    # draw a diagram of the workflow
    # draw_all_possible_flows(workflow, filename="workflow.html")

    handler = workflow.run(
        user_msg="Hello!",
        agent_configs=agent_configs,
        llm=llm,
        chat_history=[],
        initial_state=initial_state,
    )

    while True:
        async for event in handler.stream_events():
            if isinstance(event, ToolRequestEvent):
                print(
                    Fore.GREEN
                    + "SYSTEM >> I need approval for the following tool call:"
                    + Style.RESET_ALL
                )
                print(event.tool_name)
                print(event.tool_kwargs)
                print()

                approved = input("Do you approve? (y/n): ")
                if "y" in approved.lower():
                    handler.ctx.send_event(
                        ToolApprovedEvent(
                            tool_id=event.tool_id,
                            tool_name=event.tool_name,
                            tool_kwargs=event.tool_kwargs,
                            approved=True,
                        )
                    )
                else:
                    reason = input("Why not? (reason): ")
                    handler.ctx.send_event(
                        ToolApprovedEvent(
                            tool_name=event.tool_name,
                            tool_id=event.tool_id,
                            tool_kwargs=event.tool_kwargs,
                            approved=False,
                            response=reason,
                        )
                    )
            elif isinstance(event, ProgressEvent):
                print(Fore.GREEN + f"SYSTEM >> {event.msg}" + Style.RESET_ALL)

        result = await handler
        print(Fore.BLUE + f"AGENT >> {result['response']}" + Style.RESET_ALL)

        # update the memory with only the new chat history
        for i, msg in enumerate(result["chat_history"]):
            if i >= len(memory.get()):
                memory.put(msg)

        user_msg = input("USER >> ")
        if user_msg.strip().lower() in ["exit", "quit", "bye"]:
            break

        # pass in the existing context and continue the conversation
        handler = workflow.run(
            ctx=handler.ctx,
            user_msg=user_msg,
            agent_configs=agent_configs,
            llm=llm,
            chat_history=memory.get(),
            initial_state=initial_state,
        )


if __name__ == "__main__":
    asyncio.run(main())
