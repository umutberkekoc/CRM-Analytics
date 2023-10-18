import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 700)
# pd.set_option("display.max_rows", 700)
# I don't want to prefer to use this code to don't see each row for always without using head() function.

# 1) read dataset and create a copy
df_ = pd.read_csv("scanner_data.csv")
df = df_.copy()
print(df)

# 2) Create a function that shows some information about the dataset
def show_info(dataframe):
    print( "***** HEAD *****")
    print(df.head())
    print("***** TAIL *****")
    print(df.tail())
    print("***** SHAPE *****")
    print(df.shape)
    print("***** SIZE *****")
    print(df.size)
    print("***** INFO *****")
    print(df.info())
    print("***** DESCRIBE *****")
    print(df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
    print("***** NA Values *****")
    print(df.isnull().sum())
    print("***** COLUMNS *****")
    print(df.columns)
show_info(df)

# Business Understanding:
# Unnamed --> indexes
# Date --> Sale / transaction date
# Customer_ID --> Unique id's for each customers
# Transaction_ID --> Unique transaction id's for each customer's transaction.
# One customer can have more than 1 diff. transaction_id
# SKU_Category --> Product category. Different products can be inside of the same category.
# SKU --> Product code, each product has a unique SKU code
# Quantity --> Number of unit purchased by a customer for that transaction
# Sales_Amount --> (Unit price times quantity. For unit price, please divide Sales Amount by Quantity.)

# 3) Data Preparation: (dropna, add a new variable as unit price, drop Unnamed column permanently, check describe)
df.dropna(inplace=True)
df.drop("Unnamed: 0", inplace=True, axis=1)
df["Unit_Price"] = df["Sales_Amount"] / df["Quantity"]
print(df.describe().T)
df = df[df["Quantity"] > 0]


# Check if there is any repetition in the variables to get information
print(df["Customer_ID"].nunique())     # --> 22625 unique customer
print(df["Transaction_ID"].nunique())  # --> 64682 unqiue transaction (it means some customers have more than 1 transaction/frequency)
print(df["SKU"].nunique())             # --> 5242 different product.
print(df.sort_values("Customer_ID", ascending=True).head(20))

#Find how many people bought each product:
print(df["SKU"].value_counts()) # 1.way
print(df.groupby("SKU").agg({"Quantity": "count"}).sort_values("Quantity", ascending=False))  # 2. way
print(df.pivot_table("Customer_ID", "SKU", aggfunc="count").sort_values("Customer_ID", ascending=False))  # 3.way

#Find out how many units of each product were sold in total:
print(df.groupby("SKU").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False))  # 1.way
print(df.pivot_table("Quantity", "SKU", aggfunc="sum").sort_values("Quantity", ascending=False))  # 2.way

# Find out how much money was paid in total per invoice:
print(df.groupby("Transaction_ID").agg({"Sales_Amount": "sum"}).sort_values("Sales_Amount", ascending=False))  # 1.way
print(df.pivot_table("Sales_Amount", "Transaction_ID", aggfunc="sum").sort_values("Sales_Amount", ascending=False))  # 2.way

# Find out how much each customer paid in total:
print(df.groupby("Customer_ID").agg({"Sales_Amount": "sum"}).sort_values("Sales_Amount", ascending=False))  # 1.way
print(df.pivot_table("Sales_Amount", "Customer_ID", aggfunc="sum").sort_values("Sales_Amount", ascending=False))  # 2.way

# Change the type of Date variable:
df["Date"] = df["Date"].astype("datetime64[ns]")

# Calculation of RFM Metrics:
# Recency --> Today (Analyze day) Date - Last Transaction Date
# Frequency --> Total Number of transaction
# Monetary --> Total Price
# It is better when the recency is lower and opposites for the other variables.

print(df["Date"].max())
today = dt.datetime(year=df["Date"].dt.year.max()+1,
                    month=1,
                    day=2)
print(today)

rfm = df.groupby("Customer_ID").agg({"Date": lambda x: (today - x.max()).days,  # Recency
                                     "Transaction_ID": "nunique",  # Frequency
                                     "Sales_Amount": "sum"})   # Monetary

rfm.columns = ["Recency", "Frequency", "Monetary"]
print(rfm)
print(rfm.describe().T)


# Calculation of RFM Scores:
rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])    # Inverse Ratio
rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])  # Correct Ratio
rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])  # Correct Ratio

rfm["RFM_Score"] = rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str) # we do not add monetary score
print(rfm)

# Creating and Analysing RFM Segments:

seg_map = {
    #(1.row recency, 2.row frequency)
    r'[1-2][1-2]': "hibernating",  #Starts with 1 or 2 (recency) and ends with 1 or 2 (frequency)
    r'[1-2][3-4]': "at_risk",
    r'[1-2]5': "cant_loose",  #S tarts with 1 or 2 and ends with 5
    r'3[1-2]': "about_to_sleep",  # starts with 3 and ends with 1 or 2 (frequency)
    r'33': "need_attention",
    r'[3-4][4-5]': "loyal_customers",
    r'41': "promising",
    r'51': "new_customers",  # starts with 5 and ends with 1
    r'[4-5][2-3]': "potential_loyalists",
    r'5[4-5]': "champions",
}

rfm["Segment"] = rfm["RFM_Score"].replace(seg_map, regex=True)
rfm.index = rfm.index.astype(int)
print(rfm.head())

print(rfm.groupby("Segment").agg({"Recency": "mean",
                                  "Frequency": "mean",
                                  "Monetary": "mean"}))  # 1.way

print(rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"]))  # 2.way


# Show the info whose RFM_Score is 55 and 11:
print(rfm[(rfm["RFM_Score"] == "55") | (rfm["RFM_Score"] == "11")])


# Print the index information of new_customers to the screen and write it to a csv file:
new_customer_ids = pd.DataFrame()
new_customer_ids["new_customers_id"] = rfm[rfm["Segment"] == "new_customers"].index.astype(int)
print(new_customer_ids.head(20))
print(new_customer_ids.to_csv("new_customer_id.csv"))


# Show the RFM_Scores for each Segment:
rfm["RFM_Score"] = rfm["RFM_Score"].astype(int)
plt.figure(figsize=(12, 12))
sns.barplot(data=rfm, x="Segment", y="RFM_Score", palette="viridis")
plt.title("RFM Score By Segment")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

# What is the Distribution of Recency, Frequency and Monetary?
sns.histplot(data=rfm, x="Recency", palette="red")
plt.title("Recency")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())
sns.histplot(data=rfm, x="Frequency", palette="pink", bins=20)
plt.title("Frequency")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())
sns.histplot(data=rfm, x="Monetary", palette="green", bins=100)
plt.title("Monetary")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())