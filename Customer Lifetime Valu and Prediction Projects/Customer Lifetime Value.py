# Customer Lifetime Value:
# It is the monetary value that a customer will bring to a company during the
# relationship-communication he establishes with this company.
# Average earnings per purchase * number of purchases

# CLTV niye önemli --> Kaynakları verimli kullanmak için ( Kaynaklar verimli Kullanılarak (para ve zaman) mşteri bazlı
# çalışmalar gerçekleştirielbilir. Şirket için orta-uzun vadede karlılığı arttırır.

# CLTV = (Customer Value / Churn Rate) * Profit Margin
# Customer Value = Average Order Value * Purchase Frequency
# Average Order Value = Total Price / Total Transaction
# Purchase Frequency = Total Transaction / Total Number of Customers
# Churn Rate (Dropout Rate) = 1 - Repeat Rate
# Repeat Rate = Number of customer for more than 1 transaction / total number of customers
# Profit Margin = Total Price * '0.10' (Profit rate takes by 0.10 for worse scenario)

import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# Read Data --> "0nline_retail_II.xlsx" from kaggle, and sheet_name = "Year 2009-2010"
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
print(df.head())


# Business Understanding
# Invoice shows the invoice. Those starting with C show the invoices. Invoices are multiple
# because one person can buy more than one product, that's why there are more than one invoice number and it shows different products
# stockCode shows the code of the product. There is a unique code for each product
# description indicates the product description
# quantity shows how many units of that product were purchased on that invoice.
# InvoiceDate indicates the invoice date
# Price shows the product price (unit price by sterling)
# Customer ID unique customer number. Multiplexes like invoices
# Country Indicates the country where the customer lives.
# Create a function to show info of dataframe:

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

# Data Preprocessing:
df = df[~df["Invoice"].str.contains("C", na=False)]  # Invoices starting with C mean refund.
df.dropna(inplace=True)  # Dropped na values permanently
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["total_price"] = df["Price"] * df["Quantity"]  #total price variable created by price*quantity

print(df.describe().T)
print(df.isnull().sum())

clv_df = df.groupby("Customer ID").agg({"Invoice": "nunique",  # frequency
                                        "total_price": "sum",  # monetary
                                        "Quantity": "sum"})    # extra info


clv_df2 = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                        "total_price": lambda y: y.sum(),
                                        "Quantity": lambda z: z.sum()})  # 2.way


clv_df.columns = ["Total Transaction", "Total Price", "Total Unit"]
print(clv_df.head())

repeat_rate = clv_df[clv_df["Total Transaction"] > 1].shape[0] / clv_df.shape[0]
churn_rate = 1 - repeat_rate

clv_df["Profit Margin"] = 0.10 * clv_df["Total Price"]

clv_df["Purchase Frequency"] = clv_df["Total Transaction"] / clv_df.shape[0]

clv_df["Average Order Value"] = clv_df["Total Price"] / clv_df["Total Transaction"]

clv_df["Customer Value "] = clv_df["Average Order Value"] * clv_df["Purchase Frequency"]

clv_df["CLV"] = (clv_df["Customer Value "] / churn_rate * clv_df["Profit Margin"])

print(clv_df.head(10))

clv_df["Segment"] = pd.qcut(clv_df["CLV"], 4, labels=["D", "C", "B", "A"])
print(clv_df.sort_values("CLV", ascending=False).head())
print(clv_df.groupby("Segment").agg({"mean", "sum", "count"}))
print(clv_df.to_csv("clv_df.csv"))

