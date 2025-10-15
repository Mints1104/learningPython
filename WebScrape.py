import requests
import pandas as pd
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