from neurosql import NeuroSQLModel

model = NeuroSQLModel.from_pretrained("0xV4h3/neurosql")
sql = model.generate(
    query="Show all sales in Armenia for 2020",
    context="CREATE TABLE sales(id INT, country VARCHAR(20), year INT, sales INT);"
)
print(f"Predicted: {sql}")
print(f"Actual:    SELECT * FROM sales WHERE country='Armenia' AND year=2020;")

print()

sql = model.generate(
    query="Show the top 5 employees with the highest salaries",
    context="CREATE TABLE employees(id INT, name VARCHAR(50), salary INT, department VARCHAR(50));"
)
print(f"Predicted: {sql}")
print(f"Actual:    SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;")
print()

sql = model.generate(
    query="How many books were published by 'Penguin' after 2015?",
    context="CREATE TABLE books(id INT, title TEXT, publisher TEXT, release_year INT);"
)
print(f"Predicted: {sql}")
print(f"Actual:    SELECT COUNT(*) FROM books WHERE publisher='Penguin' AND release_year > 2015;")
print()
sql = model.generate(
    query="Average rating for each movie genre",
    context="CREATE TABLE movies(id INT, title TEXT, genre TEXT, rating FLOAT);"
)
print(f"Predicted: {sql}")
print(f"Actual:    SELECT genre, AVG(rating) FROM movies GROUP BY genre;")
print()
