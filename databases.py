import sqlite3 

book_title = "A Light in the Attic"
book_price = 51.77
book_rating = "Three"
# 1. Connect to the database. 
# This creates a 'book_database.db' file if it's not there.
conn = sqlite3.connect('book_database.db')
cursor = conn.cursor()

# ? is used to sanitise input & prevent SQL injection
insert_sql = "INSERT INTO books (title, price, rating) VALUES (?, ?, ?)"

#Pass the data as a tuple to the execute method
cursor.execute(insert_sql, (book_title, book_price, book_rating))

conn.commit()
print(f"'{book_title}' was added to the database.")
select_sql = "SELECT * FROM books"
cursor.execute(select_sql)
#fetchall() returns a list of tuples, where each tuple is a row
all_books = cursor.fetchall()

print(f"Books in database, size: {len(all_books)} ")
for book in all_books:
    print(book)
    
conn.close()