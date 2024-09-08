#Task 2

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/hp/Desktop/Intership\Prodigy Task 2/train.csv')

#displaying first few rows and last rows of the dataset
print(df.head())
print(df.tail())

#Cleaning Data

#missing values
missing_values=df.isnull().sum()
print(missing_values)

#create a DataFrame for missing values
missing_df = pd.DataFrame({'Column Name': missing_values.index, 'Missing values': missing_values.values})

#fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Dropping 'Cabin' due to too many missing values


#Verify that there are no more missing values
missing_values = df.isnull().sum()
print(missing_values)

"""
#Cleaned DataSet
df.to_csv('cleaned_data.csv', index=False)
#making sure that the data is cleaned 
df=pd.read_csv('C:/Users/hp/Desktop/Intership\Prodigy Task 2/cleaned_data.csv')
missing=df.isnull().sum()
print(missing)
"""

# Exploratory Data Analysis (EDA)
#summary statistics
print(df.describe())

#relationships between variables
#visualization

#heatmap:
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#histogram of passengers ages
sns.histplot(df['Age'],kde=True)
plt.title("Age Distribution of Titanic passengers")
plt.show()

#histogram of passengers fares
sns.histplot(df['Fare'],kde=True)
plt.title("Fares Distribution of Titanic passengers")
plt.show()

#Survived count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

#survival count by class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Count by Passenger Class')
plt.show()

# The hue parameter splits each bar (representing Pclass) into two parts based on the Survived variable (whether the passenger survived or not).
#One part of the bar will represent passengers who survived, and the other part will represent those who did not.

#survival count by gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival Count by Gender')
plt.show()

#Identifying and handling outliers

from scipy import stats

#Detect outliers in the fare columns using z-score
df['fare_zscore']=stats.zscore(df['Fare'])

#Filtering out rows where z-score is greater than 3 (considered as outliers)
df_no_outliers=df[df['fare_zscore'].abs() <= 3]

#removing outliers help clean data by removing extreme values and visualizes the updated distribution for better analysis

#Visualize
sns.histplot(df_no_outliers['Fare'],kde=True)
plt.title("Fare Distribution after outlier removal")
plt.show()


#Comparison between fare before removing outliers and after removing outliers
plt.subplot(1, 2, 1)  
sns.histplot(df['Fare'], kde=True)
plt.title("Fares Distribution of Titanic passengers")

plt.subplot(1, 2, 2)  
sns.histplot(df_no_outliers['Fare'], kde=True)
plt.title("Fare Distribution after Outlier Removal")

plt.tight_layout()  # Adjust spacing
plt.show()


