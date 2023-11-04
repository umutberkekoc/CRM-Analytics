# Customer Segmentation-RFM:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 700)
pd.set_option("display.max_rows", 700)
pd.set_option("display.expand_frame_repr", False)

df_ = pd.read_csv("customer_segmentation_500k.csv")
df = df_.copy()

def show_info(dataframe):
    print("*** HEAD ***")
    print(dataframe.head())
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** NA ***")
    print(dataframe.isnull().sum())
    print("*** DESCRIPTIVE STATS. ***")
    print(dataframe.describe().T)
    print("*** VALUE COUNTS ***")
    print(dataframe.value_counts())

show_info(df)

# Data Preprocessing:

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.10)
    q3 = dataframe[variable].quantile(0.90)
    range = q3 - q1
    upper_limit = q3 + 1.5 * range
    lower_limit = q1 - 1.5 * range
    return lower_limit, upper_limit

def change_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit


change_outliers(df, "qtt_order")
change_outliers(df, variable="total_spent")

df = df[df["qtt_order"] > 0]
print(df.describe().T)

# Creating RFM Metriks:
print(df.head())
print(df.info())
df["last_order"] = df["last_order"].astype("datetime64[ns]")

print(df["last_order"].max())
print(df["last_order"].dt.month.max())
today = dt.datetime(year=df["last_order"].dt.year.max(),
                    month=1, day=24)

rfm = df.groupby("customer_id").agg({"last_order": lambda x: (today - x.max()).days,  # Recency
                                     "qtt_order": "sum",            # Frequency
                                     "total_spent": "sum"})         # Monetary

rfm.columns = ["recency", "frequency", "monetary"]

# Creating RF Scores:
rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

print(rfm)

# Creating Segments:
seg_map = {
    #(1.row recency, 2.row frequency)
    r'[1-2][1-2]': "hibernating",
    r'[1-2][3-4]': "at_risk",
    r'[1-2]5': "cant_loose",
    r'3[1-2]': "about_to_sleep",
    r'33': "need_attention",
    r'[3-4][4-5]': "loyal_customers",
    r'41': "promising",
    r'51': "new_customers",
    r'[4-5][2-3]': "potential_loyalists",
    r'5[4-5]': "champions",
}

rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)

show_info(rfm)
rfm.info()
rfm["rf_score"] = rfm["rf_score"].astype(int)
print(rfm.groupby("segment").agg({"recency": "mean",
                                   "frequency": "mean",
                                   "monetary": "mean",
                                  "rf_score": ["mean", "max", "min", "count"]}).sort_values(by=("rf_score", "mean"), ascending=False).head(10))


sns.countplot(data=rfm, x="segment")
plt.title("Number of People By Segments")
plt.xlabel("segments")
plt.ylabel("Number of People")
plt.grid()
plt.xticks(rotation=45)
plt.show()


sns.barplot(data=rfm, x="segment", y="rf_score", estimator="mean", palette="viridis")
plt.title("Average rf score By Segments")
plt.xlabel("segments")
plt.ylabel("Average RF Score")
plt.grid()
plt.xticks(rotation=45)
plt.show()


# Customer Lifetime Value-(CLTV/CLV):

print(show_info(df))
print(df.info())

cltv = df.groupby("customer_id").agg({"qtt_order": "sum",  # frequency
                                      "total_spent": "sum"})  # monetary

cltv.columns = ["frequency", "monetary"]

cltv["profit_margin"] = float(input("Enter Profit Rate")) * cltv["monetary"]

repeat_rate = cltv[(cltv["frequency"] > 1)].shape[0] / cltv.shape[0]
churn_rate = 1- repeat_rate

cltv["average_order_value"] = cltv["monetary"] / cltv["frequency"]
cltv["purchase_frequency"] = cltv["frequency"] / cltv.shape[0]
cltv["customer_value"] = cltv["average_order_value"] * cltv["purchase_frequency"]

cltv["clv"] = (cltv["customer_value"] / churn_rate) * cltv["profit_margin"]

print(cltv)
cltv["segment"] = pd.qcut(cltv["clv"], q=4, labels=["D", "C", "B", "A"])

print(cltv.groupby("segment").agg({"clv": "mean"}).sort_values("clv", ascending=False).head(10))
print(cltv.sort_values("clv", ascending=False).head(12))

sns.countplot(data=cltv, x="clv")
plt.title("Number of Customers By Segments")
plt.xlabel("Segments")
plt.ylabel("Number of Customers")
plt.grid()
plt.xticks(rotation=45)
plt.show()


# Customer Lifetime Value Prediction:
from datetime import timedelta
import random
df.head()
df.info()
df["last_order"] = df["last_order"].astype("datetime64[ns]")

df['first_order'] = (df['last_order'] -
                     pd.to_timedelta([timedelta(days=random.randint(1, 30)) for _ in range(len(df))]))
df.info()
df.head()

df.describe().T
df["last_order"].max() 
df["last_order"].dt.month.max()

today = dt.datetime(year=df["last_order"].dt.year.max(),
                    month=1, day=24)

cltv_df = df.groupby("customer_id").agg({"first_order": lambda x: (today - x.min()).days,  # T (Customer Age)
                                         "qtt_order": "sum",                               # Frequency
                                         "total_spent": "sum"})                            # Monetary

cltv_df["Recency"] = (df["last_order"] - df["first_order"]).dt.days
cltv_df.columns = ["T", "Frequency", "Monetary", "Recency"]
print(cltv_df.head())
print(cltv_df.info())

cltv_df = cltv_df[cltv_df["Frequency"] > 1]
cltv_df["T"] = cltv_df["T"] / 7
cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]
print(cltv_df.head())

# Creating BG/NBD Model:

bgf = BetaGeoFitter(penalizer_coef=0.001)  # pen.coef is up to you
bgf.fit(cltv_df["Frequency"],
        cltv_df["Recency"],
        cltv_df["T"])

cltv_df["excepted_purchase_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                                  cltv_df["Frequency"],
                                                                                  cltv_df["Frequency"],
                                                                                  cltv_df["T"])

cltv_df["excepted_purchase_1_month"] = bgf.predict(4*1,
                                                    cltv_df["Frequency"],
                                                    cltv_df["Recency"],
                                                    cltv_df["T"])

cltv_df["excepted_purchase_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                  cltv_df["Frequency"],
                                                                                  cltv_df["Recency"],
                                                                                  cltv_df["T"])

print(cltv_df.sort_values("excepted_purchase_3_month", ascending=False))


# Creating GAMMA-GAMMA Model:

ggf = GammaGammaFitter(penalizer_coef=0.01)  # pen.coef is up to you
ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])

cltv_df["conditional_expected_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                                                                 cltv_df["Monetary"])

print(cltv_df.sort_values("conditional_expected_profit", ascending=False))

customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                      cltv_df["Frequency"],
                                                      cltv_df["Recency"],
                                                      cltv_df["T"],
                                                      cltv_df["Monetary"],
                                                      time=6,  # 6 months
                                                      discount_rate=0.01,  # disc.rate is up tou you
                                                      freq="W"  # freq is weekly

cltv_df["clv"] = customer_lifetime_value



cltv_final = cltv_df.merge(customer_lifetime_value, on="customer_id", how="left")

print(cltv_final.sort_values("clv", ascending=False).head(10))


# Creating Segments:

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

print(cltv_final.sort_values("clv", ascending=False).head(10))
print(cltv_final.groupby("segment").agg({"mean", "sum", "count"}))