import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
warnings.filterwarnings("ignore")
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 700)
pd.set_option("display.max_rows", 700)

# Business Understanding:
'''
Attributes
ID: Customer's unique identifier
Year_Birth: Customer's birth year
Education: Customer's education level
Marital_Status: Customer's marital status
Income: Customer's yearly household income
Kidhome: Number of children in customer's household
Teenhome: Number of teenagers in customer's household
Dt_Customer: Date of customer's enrollment with the company
Recency: Number of days since customer's last purchase
Complain: 1 if the customer complained in the last 2 years, 0 otherwise

Products

MntWines: Amount spent on wine in last 2 years
MntFruits: Amount spent on fruits in last 2 years
MntMeatProducts: Amount spent on meat in last 2 years
MntFishProducts: Amount spent on fish in last 2 years
MntSweetProducts: Amount spent on sweets in last 2 years
MntGoldProds: Amount spent on gold in last 2 years

Promotion

NumDealsPurchases: Number of purchases made with a discount
AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

Place

NumWebPurchases: Number of purchases made through the company’s website
NumCatalogPurchases: Number of purchases made using a catalogue
NumStorePurchases: Number of purchases made directly in stores
NumWebVisitsMonth: Number of visits to company’s website in the last month
'''


df_ = pd.read_csv("marketing_campaign.csv", sep="\t")
df = df_.copy()
df.head()

# Data Understanding:
def show_info(dataframe):
    print("*** HEAD ***")
    print(df.head())
    print("*** SHAPE ***")
    print(df.shape)
    print("*** SIZE ***")
    print(df.size)
    print("*** COLUMNS ***")
    print(df.columns)
    print("*** INFO ***")
    print(df.info())
    print("*** DESCRIPTIVE ***")
    print(df.describe().T)
    print("*** NA ***")
    print(df.isnull().sum())
    print(df["ID"].nunique())

show_info(df)

print(df["NumWebPurchases"].nunique())
print(df["NumCatalogPurchases"].nunique())
print(df["NumDealsPurchases"].nunique())
print(df["NumStorePurchases"].nunique())

print(df[df["ID"] == 5524])
print(df[df["ID"] == 4141])


# Data Preparation:
df.dropna(inplace=True)

unnecessary_columns = df.columns[df.columns.str.contains("Accepted")]
df.drop(unnecessary_columns, axis=1, inplace=True)

print(df.columns)
print(df.isnull().sum().any())

# RFM Metriklerinin Oluşturulması:

purchase_amount_columns = df.columns[df.columns.str.contains("Purchase")]

df["total_purchase_amount"] = df[purchase_amount_columns].sum(axis=1)

spent_columns = df.columns[df.columns.str.contains("Mnt")]

df["total_spent"] = df[spent_columns].sum(axis=1)
print(df.head())


rfm = df[["ID", "total_purchase_amount", "Recency", "total_spent"]]
rfm.columns = ["customer_id", "Frequency", "Recency", "Monetary"]

print(rfm.head())
print(rfm.describe([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

rfm = rfm[rfm["Frequency"] > 0]


# RFM Skorlarının Hesaplanması:

rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])

rfm["recency_score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])

rfm["monetary_score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

print(rfm.head())
print(rfm.sort_values("rf_score", ascending=False).head(15))


# Segmentlerin Oluşturulması:

seg_map = {
    #(1.row recency, 2.row frequency)
    r'[1-2][1-2]': "hibernating", #1 veya 2 ile başlayan (recency) ve 1 veya 2 ile biten (frequency)
    r'[1-2][3-4]': "at_risk",
    r'[1-2]5': "cant_loose", #1 veya 2 ile başlayan ve 5 ile biten
    r'3[1-2]': "about_to_sleep", #3 ile başlayan ve 1 veya 2 ile biten
    r'33': "need_attention",
    r'[3-4][4-5]': "loyal_customers",
    r'41': "promising",
    r'51': "new_customers", #5 ile başlayan ve 1 ile biten
    r'[4-5][2-3]': "potential_loyalists",
    r'5[4-5]': "champions",
}

rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)
print(rfm.head())

rfm["rf_score"] = rfm["rf_score"].astype(int)

print(rfm.groupby("segment").agg({"Frequency": "mean",
                                  "Recency": "mean",
                                  "Monetary": ["mean", "count"],
                                  "rf_score": "mean"}).sort_values(by=("rf_score", "mean"), ascending=False))


new_customer_id = rfm[rfm["segment"] == "new_customers"]["customer_id"]
champions_id = rfm[rfm["segment"] == "champions"]["customer_id"]

# Customer Lifetime Value (CLV / CLTV):

# average order value / churn rate * profit margin

# profit matgin = 0.10 * total price

# churn rate = 1 - repeat rate
#repeat rate = number of cust. purch. more than 1 / total number of cust.

# average order value = totan number of transaction * total order average
# total number of trans. = total transaction / total cust.
# total order average = total purchase value / total transaction


show_info(df)

cltv_df = df.groupby("ID").agg({"total_purchase_amount": "sum",
                                         "total_spent": "sum"})

cltv_df = cltv_df[cltv_df["total_purchase_amount"] > 0]



churn_rate = 1 - (df[df["total_purchase_amount"] > 1].shape[0] / df.shape[0])

profit_rate = float(input("Enter The Profit Rate"))
cltv_df["profit_margin"] = profit_rate * df["total_spent"]

cltv_df["Purchase Frequency"] = df["total_purchase_amount"] / df.shape[0]

cltv_df = cltv_df[cltv_df["Purchase Frequency"] > 0]

cltv_df["Average Order Value"] = df["total_spent"] / df["total_purchase_amount"]

customer_value = cltv_df["Purchase Frequency"] * cltv_df["Average Order Value"]

cltv_df["cltv"] = (customer_value / churn_rate) * cltv_df["profit_margin"]

print(cltv_df.head())
df.head()

# Segment Oluşturma:
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])

print(cltv_df.sort_values("cltv", ascending=False).head(20))
print(cltv_df.groupby("segment").agg({"cltv": ["mean", "sum", "count"]}))
print(cltv_df.describe().T)



# CLTV Prediction:
df_ = pd.read_csv("marketing_campaign.csv", sep="\t")
df = df_.copy()
df.head()
def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.95)
    range = q3 - q1
    upper_limit = q3 + 1.5 * range
    lower_limit = q1 - 1.5 * range
    return lower_limit, upper_limit

def change_with_th(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > upper_limit, variable] = upper_limit
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit

print(df.describe().T)

outlier_variables = df.columns[df.columns.str.contains("Mnt")]

for i in outlier_variables:
    change_with_th(df, i)

print(df.describe().T)

today = dt.datetime(year=2014, month=10, day=29)
df["last_order_date"] = pd.to_datetime(-df["Recency"], unit="D", origin=today)
df["Dt_Customer"] = df["Dt_Customer"].astype("datetime64[ns]")
print(df.head())
print(df.info())


cltv = pd.DataFrame()
purchase_columns = df.columns[df.columns.str.contains("Purchase")]
cltv["frequency"] = df[purchase_columns].sum(axis=1)
spent_columns = df.columns[df.columns.str.contains("Mnt")]
df["total_spent"] = df[spent_columns].sum(axis=1)


cltv["Recency_weekly"] = (df["last_order_date"] - df["Dt_Customer"]).dt.days / 7
cltv["T_weekly"] = (today - df["Dt_Customer"]).dt.days / 7


cltv = cltv[cltv["frequency"] > 1]


cltv["monetary_avg"] = df["total_spent"] / cltv["frequency"]

cltv.head()
cltv = cltv[(cltv["Recency_weekly"] > 0) & (cltv["T_weekly"] >= cltv["Recency_weekly"])]
# BG/NBD Modelling:
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv['frequency'],
        cltv['Recency_weekly'],
        cltv['T_weekly'])

#predictions for next 3 months and then next 6 months
cltv["exp_sales_3_month"] = bgf.predict(4 * 3,
                                        cltv['frequency'],
                                        cltv['Recency_weekly'],
                                        cltv['T_weekly'])

cltv["exp_sales_3_month"].sort_values(ascending=False).head(10)

cltv["exp_sales_6_month"] = bgf.predict(4 * 6,
                                        cltv['frequency'],
                                        cltv['Recency_weekly'],
                                        cltv['T_weekly'])

cltv["exp_sales_6_month"].sort_values(ascending=False).head(10)

# Tahmin Sonuçlarının Değerlendirilmesi:
plot_period_transactions(bgf)
print(plt.show())

# GAMMA GAMMA Modelling:

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv["frequency"], cltv["monetary_avg"])
cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                    cltv['monetary_avg'])


# clv oluşturma:

customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                      cltv["frequency"],
                                                      cltv["Recency_weekly"],
                                                      cltv["T_weekly"],
                                                      cltv["monetary_avg"],
                                                      time=3,
                                                      freq="W",
                                                      discount_rate=0.01)

cltv["clv"] = customer_lifetime_value
cltv["customer_id"] = df["ID"]
column_order = ["customer_id"] + [i for i in cltv.columns if i != "customer_id"]
cltv = cltv[column_order]
cltv.head()

print(cltv.sort_values("clv", ascending=False).head(10))

# segmentin oluşturulması
cltv["segment"] = pd.qcut(cltv["clv"], q=4, labels=["D", "C", "B", "A"])
cltv.head()

print(cltv.groupby("segment").agg({"frequency": "mean",
                                   "Recency_weekly": "mean",
                                   "T_weekly": "mean",
                                   "monetary_avg": "mean",
                                   "clv": ["mean", "count"]}))
# shorter way!
print(cltv.pivot_table(["frequency", "Recency_weekly", "T_weekly", "monetary_avg", "clv"], "segment", aggfunc="mean"))


#####################################################
# Tüm CLV Prediction Sürecinin Fonksiyonlaştırılması:
def outlier(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    range = q3- q1
    upper_limit = q3 + 1.5 * range
    lower_limit = q1 - 1.5 * range
    return lower_limit, upper_limit

def change_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier(dataframe, variable)
    dataframe.loc[dataframe[variable] > upper_limit, variable] = upper_limit
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit


def clv_prediction(dataframe, time=3, freq=str(input("Enter freq as Y,W,D"))):
    bgf_pen_coef = float(input("enter the bgf penalizer coefficient value."))
    ggf_pen_coef = float(input("enter the ggf penalizer coefficient value."))
    discount_rate = float(input("enter the discount rate value."))

    outlier_variables = df.columns[df.columns.str.contains("Mnt")]

    for i in outlier_variables:
        change_with_th(df, i)

    print(df.describe().T)

    today = dt.datetime(year=2014, month=10, day=29)
    df["last_order_date"] = pd.to_datetime(-df["Recency"], unit="D", origin=today)
    df["Dt_Customer"] = df["Dt_Customer"].astype("datetime64[ns]")
    print(df.head())
    print(df.info())

    cltv = pd.DataFrame()
    purchase_columns = df.columns[df.columns.str.contains("Purchase")]
    cltv["frequency"] = df[purchase_columns].sum(axis=1)
    spent_columns = df.columns[df.columns.str.contains("Mnt")]
    df["total_spent"] = df[spent_columns].sum(axis=1)

    cltv["Recency_weekly"] = (df["last_order_date"] - df["Dt_Customer"]).dt.days / 7
    cltv["T_weekly"] = (today - df["Dt_Customer"]).dt.days / 7

    cltv = cltv[cltv["frequency"] > 1]

    cltv["monetary_avg"] = df["total_spent"] / cltv["frequency"]

    cltv.head()
    cltv = cltv[(cltv["Recency_weekly"] > 0) & (cltv["T_weekly"] >= cltv["Recency_weekly"])]
    # BG/NBD Modelling:
    bgf = BetaGeoFitter(penalizer_coef=bgf_pen_coef)
    bgf.fit(cltv['frequency'],
            cltv['Recency_weekly'],
            cltv['T_weekly'])

    # predictions for next 3 months and then next 6 months
    cltv["exp_sales_3_month"] = bgf.predict(4 * 3,
                                            cltv['frequency'],
                                            cltv['Recency_weekly'],
                                            cltv['T_weekly'])

    cltv["exp_sales_3_month"].sort_values(ascending=False).head(10)

    cltv["exp_sales_6_month"] = bgf.predict(4 * 6,
                                            cltv['frequency'],
                                            cltv['Recency_weekly'],
                                            cltv['T_weekly'])

    cltv["exp_sales_6_month"].sort_values(ascending=False).head(10)

    # Tahmin Sonuçlarının Değerlendirilmesi:
    plot_period_transactions(bgf)
    print(plt.show())

    # GAMMA GAMMA Modelling:

    ggf = GammaGammaFitter(penalizer_coef=ggf_pen_coef)
    ggf.fit(cltv["frequency"], cltv["monetary_avg"])
    cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                        cltv['monetary_avg'])

    # clv oluşturma:

    customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                          cltv["frequency"],
                                                          cltv["Recency_weekly"],
                                                          cltv["T_weekly"],
                                                          cltv["monetary_avg"],
                                                          time=time,
                                                          freq=freq,
                                                          discount_rate=discount_rate)

    cltv["clv"] = customer_lifetime_value
    cltv["customer_id"] = df["ID"]
    column_order = ["customer_id"] + [i for i in cltv.columns if i != "customer_id"]
    cltv = cltv[column_order]
    cltv.head()

    print(cltv.sort_values("clv", ascending=False).head(10))

    # segmentin oluşturulması
    cltv["segment"] = pd.qcut(cltv["clv"], q=4, labels=["D", "C", "B", "A"])
    cltv.head()

    print(cltv.groupby("segment").agg({"frequency": "mean",
                                       "Recency_weekly": "mean",
                                       "T_weekly": "mean",
                                       "monetary_avg": "mean",
                                       "clv": ["mean", "count"]}))
    # shorter way!
    print(
        cltv.pivot_table(["frequency", "Recency_weekly", "T_weekly", "monetary_avg", "clv"], "segment", aggfunc="mean"))

clv_prediction(df)