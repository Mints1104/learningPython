import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

try:
    df = pd.read_csv('train.csv')
    
    #1. Look at first 5 rows
    print("--- First 5 Rows---")
    print(df.head())
    
    #2. Get a summary of the DataFrame, including data types and non null counts
    print("\n--- Data Info ---")
    print(df.info())
    
    #3. Check for the total numb of missing values in each column
    print("\n--- Missing Values Count ---")
    print(df.isnull().sum())
    
    # --- Start of Cleaning ---
    
    #1. Handle missing age: fill with the median
    median_age = df['Age'].median()
    df['Age'].fillna(median_age,inplace=True)
    
    #2. Handle missing Embarked: Fill with the mode (most common value)
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked,inplace=True)
    
    #3. Drop columns that are not useful for model
    df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    print("\n--- DataFrame after cleaning missing values and dropping columns ---")
    print(df.info())
    print("--- Test ---")
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    print("\n--- DataFrame after converting categorical columns ---")
    print(df.head())
    
    # We want to predict survived so we drop it from X and y will be what we are predicting
    X = df.drop('Survived',axis=1)
    y = df['Survived']
    # Split the data into train/test, randoms_state=42 ensures split is the same everytime
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test,predictions)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    

except FileNotFoundError:
    print("file not found")