from llama_index.core import PromptTemplate


instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

# context = """Purpose: The primary role of this agent is to assist users by answering their questions and perform all the actions they asks for. """
context = """Purpose: The primary role of this agent is to assist users by providing accurate and informative answers to their questions. 
    It can perform a wide range of tasks, including retrieving information, generating content, 
    and executing user requests. This agent is designed to enhance the user experience by delivering helpful responses in a conversational manner,
    ensuring that users feel supported and informed throughout their interaction."""

