# Imported needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv('Car_Purchasing_Data.csv')

# viewing the contents of the dataframe
print(df)


# you may use the head() to view only the top five rows of the dataframe
print("First 5 rows of the dataset:\n")
print(df.head())

##### Explore the data set #####
print(df.isnull().sum()) # summary of missing values ni

print("\nthis is the summary of information in the data set: ")
print(df.info()) # shows data set summary

## Viewing Basic Statistical Values(mean, meadian and quartiles) of the given dataset
print(df.describe())


# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix using only numeric columns
correlation_matrix = numeric_df.corr()

# Plotting with heatmap
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.title("\nCorrelation Heatmap of Numeric Features in Car_Purchasing_Data", fontsize=16)
plt.show()

# perform Exploratory Data Analysis using seaborn pairplot
sns.pairplot(numeric_df)
plt.show()


# Relational plot (Annual Salary vs Car Purchase Amount, grouped by Gender)
sns.relplot(x="Annual Salary", y="Car Purchase Amount", hue='Gender', data=df)
plt.show()

# Histogram of Car Purchase Amount
sns.histplot(df['Car Purchase Amount'], kde=True)
plt.show()