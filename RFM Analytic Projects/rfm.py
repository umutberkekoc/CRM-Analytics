import pandas as pd
import numpy as np
import datetime as dt

pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.head()

def show_info(dataframe):
    print("##### HEAD #####")
    print(dataframe.head())
    print("##### TAIL #####")
    print(dataframe.tail())
    print("##### SHAPE #####")
    print(dataframe.shape)
    print("##### INFO #####")
    print(dataframe.info())
    print("##### NAN #####")
    print(dataframe.isnull().sum())
    print("##### DESCRIPTIVE STATS. #####")
    print(dataframe.describe().T)

show_info(df)

df[df["Description"].isnull()].head(10)

df[df["StockCode"] == 22139].head(30)

df["Description"] = (df.groupby("StockCode")["Description"].
                     transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)))


df[df["Customer ID"].isnull()].head(10)

df[df["Invoice"] == 536544]

df["Customer ID"] = (df.groupby("Invoice")["Customer ID"].
                     transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)))

df.dropna(inplace=True)

df = df[df["Price"] > 0]
df = df[~df["Invoice"].str.contains("C", na=False)]

show_info(df)

def show_nunique(dataframe):
    print("##### Invoice #####")
    print(dataframe["Invoice"].nunique())
    print("##### StockCode #####")
    print(dataframe["StockCode"].nunique())
    print("##### Description #####")
    print(dataframe["Description"].nunique())
    print("##### Customer ID #####")
    print(dataframe["Customer ID"].nunique())
    print("##### Country #####")
    print(dataframe["Country"].nunique())

show_nunique(df)

print(df["StockCode"].value_counts())
print(df["Description"].value_counts())
print(df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head())

df.loc[:, "TotalPrice"] = df["Price"] * df["Quantity"]

df.head()


print("Last Transaction Date:", df["InvoiceDate"].max())

today_date = dt.datetime(year=df["InvoiceDate"].dt.year.max(),
                         month=12, day=11)


rfm = df.groupby("Customer ID").agg({'InvoiceDate': lambda x: (today_date - x.max()).days,
                                      "Invoice": lambda x: x.nunique(),
                                      "TotalPrice": lambda x: x.sum()})

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["recency_score"] = pd.qcut(x=rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(x=rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(x=rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

show_info(rfm)

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)


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

rfm["Segment"] = rfm["rf_score"].replace(seg_map, regex=True)

rfm.groupby("Segment").agg({"Recency": ["mean", "count"],
                            "Frequency": ["mean", "count"],
                            "Monetary": ["mean", "count"]})

rfm[rfm["Segment"] == "champions"].index
rfm[rfm["Segment"] == "new_customers"].index
rfm[rfm["Segment"] == "loyal_customers"].index

