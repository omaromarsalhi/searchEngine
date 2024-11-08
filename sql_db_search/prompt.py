from llama_index.core import PromptTemplate

# System message for the SQL agent in LlamaIndex (Modified for direct query execution)
system_prefix = """\
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct MySQL query to run.
Only execute the query and return the result, no explanations or descriptions.
You must return only the result of the query, no other information.
You are not allowed to explain or describe the SQL query itself.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table; only ask for the relevant columns given the question.

You must only use the information returned by the query to construct your final answer.
Only use the given tools to interact with the database.

You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

If the question does not seem related to the database, return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:
"""

# Few-shot examples can be added as part of the prompt for better context
examples = """
User input: Show me the top 5 customers by total orders.
SQL query: SELECT CustomerId, SUM(TotalAmount) AS TotalOrders FROM Orders GROUP BY CustomerId ORDER BY TotalOrders DESC LIMIT 5;

User input: How many orders were placed in the last month?
SQL query: SELECT COUNT(*) FROM Orders WHERE OrderDate >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);

User input: List the names of customers who placed more than 3 orders.
SQL query: SELECT CustomerName FROM Customers WHERE CustomerId IN (SELECT CustomerId FROM Orders GROUP BY CustomerId HAVING COUNT(OrderId) > 3);
"""

# Combining the prefix, examples, and user input into one template
# llamaindex_prompt = PromptTemplate(
#     """\
# {system_prefix}
#
# {examples}
#
# User input: {query}
#
# SQL query:
# """
# )
#
# # Setting up the template by injecting the system prefix and examples
# new_prompt = llamaindex_prompt.format(
#     system_prefix=system_prefix,
#     examples=examples,
#     dialect="MySQL",
#     top_k=5
# )

# # # Example usage with `input`, `dialect`, and `top_k` replacements
# input_prompt = new_prompt.format(
#     query="Get the total amount of each order by customer",
# )
#
# print(input_prompt)
