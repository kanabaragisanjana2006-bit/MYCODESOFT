import pandas as pd

# 1. Import the dataset
df = pd.read_csv("dataset.csv")

print("----- ORIGINAL DATASET -----")
print(df)

# 2. Inspect the structure
print("\n----- DATASET INFORMATION -----")
print(df.info())

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())

# 3. Clean text entries
df["Name"] = df["Name"].astype("string").str.strip().str.title()
df["City"] = df["City"].astype("string").str.strip().str.title()

# Make city names consistent
df["City"] = df["City"].replace({
    "Belgaum": "Belagavi",
    "Bengaluru": "Bangalore",
    "Mysuru": "Mysore"
})

# 4. Correct data types
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce", dayfirst=True)

# 5. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["City"] = df["City"].fillna("Unknown")
df["Join_Date"] = df["Join_Date"].fillna(pd.Timestamp("2024-01-01"))

# 6. Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# 7. Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\n----- CLEANED DATASET -----")
print(df)

print("\n----- CLEANED DATASET INFORMATION -----")
df.info()

print("\nCleaned dataset saved successfully as cleaned_data.csv")
