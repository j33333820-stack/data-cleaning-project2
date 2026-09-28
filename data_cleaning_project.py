import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. CREATE A MESSY BUSINESS DATASET
# ---------------------------------------------------------

np.random.seed(42)

n = 12000

data = {
    "Order_ID": range(10001, 10001 + n),
    "Order_Date": pd.date_range("2024-01-01", periods=n, freq="2h"),
    "Product": np.random.choice(
        ["Laptop", "Phone", "Tablet", "Headphones", "Keyboard", "Mouse"],
        n
    ),
    "Category": np.random.choice(
        ["Electronics", "Accessories"],
        n
    ),
    "Quantity": np.random.randint(1, 10, n),
    "Unit_Price": np.random.uniform(500, 50000, n).round(2),
    "Discount": np.random.uniform(0, 0.30, n).round(2),
    "Cost_Price": np.random.uniform(300, 40000, n).round(2),
    "Customer_Age": np.random.randint(18, 65, n),
    "Payment_Mode": np.random.choice(
        ["UPI", "Credit Card", "Debit Card", "Cash"],
        n
    )
}

df = pd.DataFrame(data)

# Calculate sales amount
df["Sales_Amount"] = (
    df["Quantity"] * df["Unit_Price"] * (1 - df["Discount"])
).round(2)

# Calculate profit
df["Profit"] = (
    df["Sales_Amount"] - (df["Quantity"] * df["Cost_Price"])
).round(2)


# ---------------------------------------------------------
# 2. ADD MISSING VALUES
# ---------------------------------------------------------

df.loc[np.random.choice(df.index, 150, replace=False), "Product"] = np.nan

df.loc[np.random.choice(df.index, 120, replace=False), "Customer_Age"] = np.nan

df.loc[np.random.choice(df.index, 100, replace=False), "Payment_Mode"] = np.nan


# ---------------------------------------------------------
# 3. ADD DUPLICATE RECORDS
# ---------------------------------------------------------

duplicates = df.sample(100, random_state=42)

df = pd.concat([df, duplicates], ignore_index=True)


# ---------------------------------------------------------
# 4. CREATE INCONSISTENT DATA TYPES
# ---------------------------------------------------------

df["Quantity"] = df["Quantity"].astype(str)

df["Customer_Age"] = df["Customer_Age"].astype(str)

df["Order_Date"] = df["Order_Date"].astype(str)


# ---------------------------------------------------------
# 5. SAVE RAW DATASET
# ---------------------------------------------------------

df.to_csv("raw_dataset.csv", index=False)

print("RAW DATASET CREATED")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ---------------------------------------------------------
# 6. BEFORE CLEANING CHECK
# ---------------------------------------------------------

print("\n========== BEFORE CLEANING ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)


# ---------------------------------------------------------
# 7. FIX DATA TYPES
# ---------------------------------------------------------

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Customer_Age"] = pd.to_numeric(
    df["Customer_Age"],
    errors="coerce"
)


# ---------------------------------------------------------
# 8. HANDLE MISSING VALUES
# ---------------------------------------------------------

# Product → most common value
df["Product"] = df["Product"].fillna(
    df["Product"].mode()[0]
)

# Customer Age → median
df["Customer_Age"] = df["Customer_Age"].fillna(
    df["Customer_Age"].median()
)

# Payment Mode → most common value
df["Payment_Mode"] = df["Payment_Mode"].fillna(
    df["Payment_Mode"].mode()[0]
)


# ---------------------------------------------------------
# 9. REMOVE DUPLICATES
# ---------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicates removed:",
      before_duplicates - after_duplicates)


# ---------------------------------------------------------
# 10. HANDLE OUTLIERS
# ---------------------------------------------------------

# Function for IQR outlier treatment
def handle_outliers(dataframe, column):

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    dataframe[column] = dataframe[column].clip(
        lower_limit,
        upper_limit
    )

    return dataframe


df = handle_outliers(df, "Quantity")
df = handle_outliers(df, "Customer_Age")
df = handle_outliers(df, "Unit_Price")
df = handle_outliers(df, "Sales_Amount")


# ---------------------------------------------------------
# 11. FEATURE ENGINEERING
# ---------------------------------------------------------

# Extract Year
df["Year"] = df["Order_Date"].dt.year

# Extract Month
df["Month"] = df["Order_Date"].dt.month

# Extract Month Name
df["Month_Name"] = df["Order_Date"].dt.month_name()

# Calculate Profit Margin
df["Profit_Margin"] = (
    (df["Profit"] / df["Sales_Amount"]) * 100
).round(2)


# ---------------------------------------------------------
# 12. FINAL DATA TYPE STANDARDIZATION
# ---------------------------------------------------------

df["Quantity"] = df["Quantity"].astype(int)

df["Customer_Age"] = df["Customer_Age"].round().astype(int)

df["Year"] = df["Year"].astype(int)

df["Month"] = df["Month"].astype(int)


# ---------------------------------------------------------
# 13. AFTER CLEANING CHECK
# ---------------------------------------------------------

print("\n========== AFTER CLEANING ==========")

print("\nFirst 5 cleaned rows:")
print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate records after cleaning:")
print(df.duplicated().sum())

print("\nData types after cleaning:")
print(df.dtypes)

print("\nFinal dataset shape:")
print(df.shape)


# ---------------------------------------------------------
# 14. EXPORT CLEAN DATASET
# ---------------------------------------------------------

df.to_csv(
    "clean_dataset.csv",
    index=False
)

print("\n===================================")
print("CLEAN DATASET EXPORTED SUCCESSFULLY")
print("File Name: clean_dataset.csv")
print("===================================")


# ---------------------------------------------------------
# 15. BASIC SUMMARY
# ---------------------------------------------------------

print("\n========== DATA SUMMARY ==========")

print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))

print("\nTotal Sales:",
      round(df["Sales_Amount"].sum(), 2))

print("Total Profit:",
      round(df["Profit"].sum(), 2))

print("Average Profit Margin:",
      round(df["Profit_Margin"].mean(), 2), "%")
