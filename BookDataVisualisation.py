import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_and_plot():
    conn = None
    try:
        conn = sqlite3.connect('book_database.db')
        df = pd.read_sql_query("SELECT rating, price FROM books",conn)
        average_price_by_rating = df.groupby('rating')['price'].mean().reset_index()
        rating_order = ["One", "Two", "Three", "Four", "Five"]
        average_price_by_rating['Rating'] = pd.Categorical(average_price_by_rating['rating'], categories=rating_order, ordered=True)
        average_price_by_rating = average_price_by_rating.sort_values('rating')
        
        print("Average Price Per Star Rating")
        print(average_price_by_rating)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=average_price_by_rating,x='rating',y='price', palette='viridis')
       
        plt.xlabel("Star Rating")
        plt.ylabel("Number of Books")
        plt.title("Distribution of Book Ratings")
        
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")    
    finally:
        if conn:
            conn.close()
         
analyse_and_plot()