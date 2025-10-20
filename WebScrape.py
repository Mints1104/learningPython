import requests
import pandas as pd
import sqlite3
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
books_data = []
try:
    
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    test = soup.find_all("article",class_="product_pod")
    for item in test:
        h3 = item.find("h3")
        title = h3.a["title"]
        price = item.find("p",class_="price_color")
        rating_tag = item.find("p",class_="star-rating")
        rating = rating_tag['class'][1]
        stock_availability = item.find("p",class_="instock availability")
       
        book_info = {
           "Title": title,
           "Price": price.text,
           "Rating": rating,
           "Availability": stock_availability.text.strip()
           }
        books_data.append(book_info)
           
        print(title)
        print(price.text)
        print(f"Rating: {rating} Stars")
        print(stock_availability.text.strip())
    df = pd.DataFrame(books_data)
    print(df)
    df.to_csv('books.csv',index=False)
    
except requests.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    
try:
    conn = sqlite3.connect('book_database.db')
    cursor = conn.cursor()
    insert_sql = "INSERT INTO books (title, price, rating) VALUES (?, ?, ?)"
   
    for book in books_data:
        title = book['Title']
        price = float(book['Price'].replace("£", "").replace(",", ""))
        rating = book['Rating']
        cursor.execute(insert_sql,(title,price,rating))

    conn.commit()
    print(f"Successfully inserted {len(books_data)} books into the database")
except Exception as e:
    print(f"An error occurred during database insertion: {e}")
finally:
    if conn:
        conn.close()
        print("Database connection closed")
conn_new = sqlite3.connect('book_database.db')
select_sql = "SELECT * FROM books"
cursor.execute(select_sql)
all_books = cursor.fetchall()
print(f"Books in database, size: {len(all_books)} ")
for book in all_books:
    print(book)
