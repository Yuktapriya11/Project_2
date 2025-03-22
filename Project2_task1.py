#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

# Handle outliers using IQR
Q1 = df[['Sales', 'Profit']].quantile(0.25)
Q3 = df[['Sales', 'Profit']].quantile(0.75)
IQR = Q3 - Q1

# Removing outliers
df = df[~((df[['Sales', 'Profit']] < (Q1 - 1.5 * IQR)) | (df[['Sales', 'Profit']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 2: Statistical Analysis
print("\nStatistical Summary:")
print(df[['Sales', 'Profit', 'Discount']].describe())

# Step 3: Data Visualization
plt.figure(figsize=(8, 5))
sns.histplot(df['Sales'], bins=20, kde=True)
plt.title("Sales Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Profit'])
plt.title("Profit Boxplot")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[['Sales', 'Profit', 'Discount']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:





# In[ ]:
