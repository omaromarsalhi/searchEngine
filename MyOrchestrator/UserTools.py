from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from workflow import (
    ProgressEvent,
)
from utils import FunctionToolWithContext
import mysql.connector


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
        FunctionToolWithContext.from_defaults(async_fn=login, description=(
            "Logs in a user by validating their first and last name against a MySQL database. "
            "If the user exists, updates the user state with a session token, account ID, and balance."
        )),
        # FunctionToolWithContext.from_defaults(async_fn=is_authenticated),
    ]
