from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType

combined_prompt = (
    "You are an agent designed to interact with a SQL database. "
    "Given an input question, create a syntactically correct MySQL query to run. "
    "Only execute the query and return the result, no explanations or descriptions. "
    "You must return only the result of the query, no other information. "
    "You are not allowed to explain or describe the SQL query itself.\n\n"

    "Always limit your query to at most 5 results. "
    "If the use specify a number of thing to select then limit your query to that number and ignore the 5 results rule"
    "You can order the results by a relevant column to return the most interesting examples in the database. "
    "Never query for all the columns from a specific table; only ask for the relevant columns given the question.\n\n"

    "You must only use the information returned by the query to construct your final answer. "
    "Only use the given tools to interact with the database. "
    "You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n\n"

    "Here are some examples of user inputs and their corresponding SQL queries:\n\n"
    "{examples}\n\n"

    "{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "

)

examples = """
User input: Show me the top 5 customers by total orders.
SQL query: SELECT CustomerId, SUM(TotalAmount) AS TotalOrders FROM Orders GROUP BY CustomerId ORDER BY TotalOrders DESC LIMIT 5;

User input: How many orders were placed in the last month?
SQL query: SELECT COUNT(*) FROM Orders WHERE OrderDate >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);

User input: List the names of customers who placed more than 3 orders.
SQL query: SELECT CustomerName FROM Customers WHERE CustomerId IN (SELECT CustomerId FROM Orders GROUP BY CustomerId HAVING COUNT(OrderId) > 3);
"""

# Use this combined prompt in the query engine
custom_prompt = PromptTemplate(
    combined_prompt,
    prompt_type=PromptType.TEXT_TO_SQL,
)

custom_prompt.format(examples=examples)


response_prompt = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
    "If the question does not related to the database, return 'This question is not related to the database'.\n\n"
    "If the SQL query involves restricted operations (like DELETE, INSERT, UPDATE, or DROP), "
    "respond with 'Sorry, this operation is not allowed'. Otherwise, provide the result of the query."
)


response_prompt = PromptTemplate(
    response_prompt,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS_V2,
)



