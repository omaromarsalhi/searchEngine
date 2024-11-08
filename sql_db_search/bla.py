import re
from datetime import datetime


def correct_insert_statements(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    corrected_lines = []
    for sql in lines:
        # Step 1: Add backticks around the table name "Order" if it's used
        corrected_sql = re.sub(r'\bOrder\b', 'Order', sql)

        # Step 2: Find the date in formats like 'Jul 10 2012 12:00:00:000AM'
        date_match = re.search(r"'([A-Za-z]{3} \d{1,2} \d{4} \d{1,2}:\d{2}:\d{2}:\d{3}[AP]M)'", corrected_sql)

        if date_match:
            original_date_str = date_match.group(1)

            # Step 3: Convert the date to MySQL-compatible format 'YYYY-MM-DD HH:MM:SS'
            try:
                # Parse the date with milliseconds and AM/PM
                parsed_date = datetime.strptime(original_date_str, "%b %d %Y %I:%M:%S:%f%p")
                mysql_date_str = parsed_date.strftime("%Y-%m-%d %H:%M:%S")

                # Replace the original date in the SQL with the MySQL-compatible date
                corrected_sql = corrected_sql.replace(original_date_str, mysql_date_str)
            except ValueError:
                print(f"Date format error in line: {sql}")

        # Add the corrected line to the list
        corrected_lines.append(corrected_sql)

    # Write all corrected SQL statements to the output file
    with open(output_file, 'w') as file:
        file.writelines(corrected_lines)




input_sql_file = 'C:\\Users\\salhi\\Downloads\\download-sample-database\\output3.sql'
output_sql_file = 'C:\\Users\\salhi\\Downloads\\download-sample-database\\output4.sql'
correct_insert_statements(input_sql_file, output_sql_file)
