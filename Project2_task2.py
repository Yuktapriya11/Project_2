#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the CSV file (Ensure the file is in the same directory as the notebook)
df = pd.read_csv(r"C:\Users\hp\Downloads\archive\superstore.csv")

# Display first few rows to verify
print(df.head())



# Step 1: Data Cleaning
print("Missing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert 'Order.Date' column to datetime format
df['Order.Date'] = pd.to_datetime(df['Order.Date'])

# Step 2: Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Analysis - Sales Trend Over Time
plt.figure(figsize=(10, 5))
df.groupby('Order.Date')['Sales'].sum().plot()
plt.title("Sales Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

# Scatter Plot - Profit vs. Discount
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Discount'], y=df['Profit'])
plt.title("Profit vs. Discount")
plt.show()

# Sales by Region
plt.figure(figsize=(8, 5))
df.groupby('Region')['Sales'].sum().plot(kind='bar', color='skyblue')
plt.title("Sales by Region")
plt.show()

# Sales by Category
plt.figure(figsize=(8, 5))
df.groupby('Category')['Sales'].sum().plot(kind='bar', color='orange')
plt.title("Sales by Category")
plt.show()

# Step 3: Predictive Modeling - Sales Prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting Features and Target Variable
X = df[['Profit', 'Discount']]
y = df['Sales']

# Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")


# In[ ]:
