import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
list_a = [1,2,3,4]
doubled_list = [x*2 for x in list_a]
print(f"Python List: {doubled_list}")

#numpy array
array_a = np.array([1,2,3,4])
doubled_array = array_a * 2
print(f"Numpy Array: {doubled_array}")

#creating dataframe from dict

data = {
    'Name': ['Harun','Alice','Bob'],
    'Age': [25,30,22],
    'City': ['Bristol','London','Bristol']
}
df = pd.DataFrame(data)
print("--- Full DataFrame ---")
print(df)

# This is like WHERE City = 'Bristol' in SQL
bristol_residents = df[df['City'] == 'Bristol']
print("\n--- Filtered DataFrame (Bristol Residents)")
print(bristol_residents)

# 1. Sample data features (X) and labels (Y)
# Features: [height (cm), weight (kg)]
X_train = [[180,85],[160,65],[175,80],[165,70]]
# Labels: 0 for male, 1 for female
y_train = [0,1,0,1]

#2. Initialise the model
model = DecisionTreeClassifier()
#3. Fit (train) the model on the data
model.fit(X_train, y_train)

#4. Predict on new unseen data
X_new = [[178,82]]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("Prediction: Male")
else:
    print("Prediction: Female")


data = [101,102,104]
series = pd.Series(data,index=["a","b","c"])
print(series)