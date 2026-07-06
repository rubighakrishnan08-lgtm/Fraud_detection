import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

plt.style.use("ggplot")


transaction = pd.read_csv("train_transaction.csv")
identity = pd.read_csv("train_identity.csv")
df = transaction.merge(identity,on="TransactionID",how="left")
print("Dataset Loaded Successfully")

# to print first five head of the  dataset 
print(df.head())

# to check dimensions of dataset 
print(df.shape)

# # to check the datatype
print(df.dtypes)

# to check null values count and percentage 
nulls = pd.DataFrame({
    "Count": df.isnull().sum(),
    "Percent": (df.isnull().sum()/len(df))*100
})
nulls = nulls.sort_values(
    by="Percent",
    ascending=False
)
print(nulls)


# columns more than 20%missing values
more_than_20 = nulls[nulls["Percent"] > 20]
print(more_than_20)

# columns that have 20% or less missing values 
# Find columns with <=20% missing values
below20 = nulls[nulls["Percent"]<=20].index
for col in below20:
        if pd.api.types.is_numeric_dtype(df[col]):
             df[col] = df[col].fillna(
             df[col].median()
        )


# null percentage before removing duplicate
null_before = (df.isnull().sum()/len(df))*100

# count and remove duplicates
duplicate_count = df.duplicated().sum()
print("Number of duplicated rows:", duplicate_count)
rows_before = df.shape[0]
df = df.drop_duplicates()
rows_after = df.shape[0]
print("Rows before removing:", rows_before)
print("Rows after removing :", rows_after)
print(f"Rows removed:{rows_before - rows_after}" )


# chech whether null percentage changed and compare with original null percentage
null_after = (df.isnull().sum()/len(df))*100
compare = pd.DataFrame({
    "Before":nulls["Percent"],
    "After":null_after
})
print(compare)

#  display all object columns
object_cols = df.select_dtypes(include='object').columns
print("Number of object columns:",len(object_cols))
print(object_cols)


# Memory and conversion
memory_before = df.memory_usage(deep=True).sum()
print("Memory before:", memory_before)
print(df.dtypes)
for col in object_cols:
    df[col] = df[col].astype("category")
memory_after = df.memory_usage(deep=True).sum()
print("Memory after :", memory_after)
print("Memory saved:", memory_before - memory_after)

# statistical summary 
print(df.describe())
skewness = df.skew(numeric_only = True )
print(skewness)
top10 = skewness.abs().sort_values(ascending=False).head(10)
print("Top 10 Most Skewed Columns")
print(top10)
highest_skew = skewness.abs().idxmax()
print("Column with highest skew:",highest_skew)
print("Skewness value:",skewness[highest_skew])

# outliers detection 
top2 = skewness.abs().sort_values(ascending=False).head(2).index
numeric_cols = list(top2)
print(top2)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nColumn: {col}")
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Number of Outliers: {len(outliers)}")


# Line plot 
plt.figure(figsize=(12,5))
sample = df.sort_values("TransactionDT").head(1000)
plt.plot(sample["TransactionDT"],sample["TransactionAmt"],linewidth=1)
plt.title("Transactions Amount over Time")
plt.xlabel("Transaction Time ")
plt.ylabel("Transaction Amount")
plt.grid(True)
plt.show()

# Bar Graph
mean_amt = df.groupby("ProductCD")['TransactionAmt'].mean()
plt.figure(figsize=(7,5))
mean_amt.plot(kind='bar')
plt.title("Average Transaction Amount by Product")
plt.xlabel("Product Category")
plt.ylabel("Average Transaction Amount")
plt.xticks(rotation=0)
plt.show()

# Histogram graph
plt.figure(figsize=(8,5))
sns.histplot(df['TransactionAmt'], bins=30)
plt.yscale("log")
plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Scatter plot 
sample = df.sample(5000, random_state=42)
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=sample,
    x='TransactionAmt',
    y='D1'
)
plt.title("Transaction Amount vs D1")
plt.xlabel("Transaction Amount")
plt.ylabel("D1")
plt.show()

# Box plot 
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x='ProductCD',
    y='TransactionAmt',
    showfliers=False
)
plt.title("Transaction Amount by Product Category")
plt.xlabel("Product")
plt.ylabel("Transaction Amount")
plt.show()

# Correlation and plotting heatmap
# Correlation matrix
selected_cols = [
    "TransactionAmt",
    "TransactionDT",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "D1",
    "D2",
    "D10",
    "id_01",
    "id_02",
    "card1",
    "card2",
    "card3",
    "card5",
    "isFraud"
]

corr = df[selected_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()

corr = df[selected_cols].select_dtypes(include="number").corr()
np.fill_diagonal(corr.values,0)
# Ignore self-correlation
highest = corr.unstack().idxmax()
value = corr.unstack().max()
print("Highest correlation pair:", highest)
print("Correlation:", value)

# (a)imputation stragtegy comparison ,spearman rank correlation  and grouped aggregation
print("\nMean vs Median")
for col in top2:
    print(f"\nColumn: {col}")
    print(f"Mean   : {df[col].mean()}")
    print(f"Median : {df[col].median()}")

for col in top2:
    df[col] = df[col].fillna(df[col].median())

print("\nRemaining Null Values")

for col in top2:
    print(col, ":", df[col].isnull().sum())

# (b)spearman rank correlation
pearson = df[selected_cols].corr(method="pearson")
print("\nPearson Correlation Matrix")
print(pearson)

spearman = df[selected_cols].corr(method="spearman")
print("\nSpearman Correlation Matrix")
print(spearman)

difference = (spearman - pearson).abs()
print("\nAbsolute Difference")
print(difference)
mask = np.triu(np.ones(difference.shape), k=1).astype(bool)
top3 = (
    difference.where(mask).stack().sort_values(ascending=False).head(3))
print("\nTop 3 Largest Differences")
print(top3)

# (c)
group_stats = (df.groupby("ProductCD", observed=True)["TransactionAmt"].agg(["mean", "std", "count"]))
print(group_stats)

print("\nHighest Mean")
print(group_stats["mean"].idxmax())

print("\nHighest Standard Deviation")
print(group_stats["std"].idxmax())

highest_mean = group_stats["mean"].max()
lowest_mean = group_stats["mean"].min()
ratio = highest_mean / lowest_mean
print("\nHighest/Lowest Mean Ratio:", ratio)

# Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)
print("Cleaned dataset saved successfully as 'cleaned_data.csv'")
