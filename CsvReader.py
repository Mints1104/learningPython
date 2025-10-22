

products = []
with open('sales.csv','r') as file:
    
    for line in file:
        if "Product" in line:
           print("skipping")
           continue
        
        cleaned_line = line.split(",")
        product = cleaned_line[0]
        category = cleaned_line[1]
        price = cleaned_line[2]
        quantity_sold = cleaned_line[3].strip()
        
        product_info = {
            "Product": product,
            "Category": category,
            "Price": price,
            "Quantity Sold": quantity_sold
        }
        products.append(product_info)
        
       
       
       
       
category_totals = {}
for product in products:
    category = product['Category']
    quantity = int(product['Quantity Sold'])

    category_totals[category] = category_totals.get(category,0) + quantity
    
print("\n--- Total Quantity Sold Per Category ---")
for category, total in category_totals.items():
    print(f"{category}: {total}")       
     