import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset
df = pd.read_csv(r"C:\LPU\LPU S4\INT375\districtwise-ipc-crimes-2017-onwards.csv")

# Checking the correct loading of the dataset
print(df)

#--------------------------------------------------------------------------------------------------------------------------------

#Exploring dataset

pd.set_option('display.max_info_columns', 140)
print(df.info())
print("\nDescription of the dataset:\n", df.describe().round(2))

#--------------------------------------------------------------------------------------------------------------------------------

#Checking for missing values in the dataset
missing_values = df.isnull().sum().sort_values(ascending=False)
print("\nMissing Values per Column:")
print(missing_values[missing_values > 0])
print("\nThe Total number of missing values in the dataset: ",df.isnull().sum().sum())

# Handling missing values
df = df.drop(['rioting_vigilants'],axis=1)
df['atmpt_acid_attack']= df['atmpt_acid_attack'].fillna(df['atmpt_acid_attack'].mean())

print("\nAfter handling missing values, Total missing values in the dataset: ",df.isnull().sum().sum())

# remove duplicates rows if present
df = df.drop_duplicates()
# remove rows with any negative values in numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns
df = df[(df[num_cols] >= 0).all(axis=1)]

#-----------------------------------------------------------------------------------------------------------------------------------

# Performing basic operations
print("Shape of the dataset: ", df.shape)
print("Actual column names :\n")
print(df.columns.tolist())
print("\nHead of the dataset:\n", df.head(11))
print("\nTail of the dataset:\n", df.tail(11))
print("Datatypes of dataset:\n", df.dtypes)
print("Number of unique values in each column:\n", df.nunique())

df.to_csv("Crime_cleaned.csv", index=False)

#-----------------------------------------------------------------------------------------------------------------------------------

# Objective 1: Year-wise Trend Analysis of Major Crimes

yearly_trends = df.groupby('year')[['murder', 'hit_and_run', 'arson']].sum().reset_index()
plt.figure()
sns.lineplot(data=yearly_trends, x="year", y="murder", marker='o', label='Murder')
sns.lineplot(data=yearly_trends, x="year", y="hit_and_run", marker='o', label='Hit and Run')
sns.lineplot(data=yearly_trends, x="year", y="arson", marker='o', label='Arson')
plt.title("Year-wise Trend of Major Crimes")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.legend()
plt.tight_layout()
plt.grid(False)
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Objective 2: Top 10 Districts with Highest Total IPC Crimes

df['total_crimes'] = df.iloc[:, 7:].sum(axis=1)
district_crimes_total = df.groupby('district_name')['total_crimes'].sum().sort_values(ascending=False).head(10)
state_crimes_total = df.groupby('state_name')['total_crimes'].sum().sort_values(ascending=False).head(10)

fig, axs = plt.subplots(1, 2, figsize=(16,6))
district_crimes_total.plot(kind='bar', color='skyblue', ax=axs[0])
axs[0].set_title("Top 10 Districts with Highest Total IPC Crimes")
axs[0].set_xlabel("")
axs[0].set_ylabel("Total Crimes")
axs[0].tick_params(axis='x', rotation=45)

state_crimes_total.plot(kind='bar', color='coral', ax=axs[1])
axs[1].set_title("Top 10 States with Highest Total IPC Crimes")
axs[1].set_xlabel("")
axs[1].set_ylabel("Total Crimes")
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Objective 3: State-wise Crime Rate Heatmap

state_crimes = df.groupby('state_name')['total_crimes'].sum().reset_index()
state_crimes = state_crimes.sort_values(by='total_crimes', ascending=False)
plt.figure(figsize=(10, 12))
sns.heatmap(state_crimes.set_index('state_name'), annot=True, fmt=".0f", cmap="Reds")
plt.title("State-wise Total IPC Crimes")
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Objective 4: Crime Type Distribution in a Selected State

state = 'West Bengal'
state_data = df[df['state_name'] == state]

crime_distribution = state_data.select_dtypes(include='number').sum().sort_values(ascending=False).head(10)

crime_df = crime_distribution.reset_index()
crime_df.columns = ['Crime_Type', 'Cases']
plt.figure(figsize=(10, 6))
sns.barplot(x='Crime_Type', y='Cases', hue='Crime_Type', data=crime_df, palette='magma', legend=False)
plt.title(f"Top 10 Crime Types in {state}")
plt.ylabel("Number of Cases")
plt.xlabel("Crime Type")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()


#--------------------------------------------------------------------------------------------------------------------------------

# Objective 5: Correlation Between Different Crime Categories

selected_columns = ['murder', 'hit_and_run', 'arson', 'cheating_impersonation', 'criminal_trespass']
correlation_matrix = df[selected_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Between Selected Crime Categories")
plt.xticks(rotation=0)
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Objective 6: Crime Distribution

plt.figure(figsize=(10,6))
df['murder'].hist(bins=30, color='lightcoral', edgecolor='black')
plt.title('Distribution of Murder Cases')
plt.xlabel('Number of Cases')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

# Objective 7: Crime Category Distribution

crime_types = ['murder', 'arson', 'hit_and_run', 'criminal_intimidation','insult_modesty_women', 'crlty_husbnd_relatives', 'other_ipc_crimes']
category_totals = df[crime_types].sum().sort_values(ascending=False)

plt.figure(figsize=(8,8))
plt.pie(category_totals, labels=category_totals.index, startangle=140, wedgeprops=dict(width=0.4))
plt.title('Distribution of Selected Crime Categories')
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

#Objective 8: Crime Trend Analysis Over Years
df['total_crimes'] = df.iloc[:, 7:].sum(axis=1)

yearly_crimes = df.groupby('year').sum(numeric_only=True)
yearly_crimes_total = yearly_crimes.sum(axis=1)
plt.figure(figsize=(10,6))
plt.plot(yearly_crimes_total.index, yearly_crimes_total.values, marker='o', color='orange')
plt.title('Total IPC Crimes per Year')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()
