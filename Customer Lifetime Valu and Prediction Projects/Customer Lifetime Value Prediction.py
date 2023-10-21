# Customer Lifetime Value Prediction:

# Customer Value = Purchase Frequency * Average Order Value
# CLTV = Expected Number of Transaction * Expected Average Profit
# CLTV = BG/NBD Model * Gamma Gamma Submodel


# Expected Number of Transaction With BG/NBD (Beta Geometric / Negative Binomial Distribution):
# BG/NBD --> Buy Till You Die
# BG/NBD Model --> 2 Model for  Expected number of transaction için
# Transaction Process (Buy) + Dropout Process (Till You Die)
# Transaction Process (Buy):

#As long as a customer is alive, the number of transactions to be performed by a customer in a given time period is poisson distributed.
# !Transaction Rates vary for each customer and are distributed "gamma" for the entire audience (r,a)

# Dropout Process (Till you die):
# After a customer makes a purchase, there will be a drop with a certain probability (p). (stops shopping)
# Dropout rates vary for each customer and are distributed "beta" for the entire audience (a,b)

# x --> Number of purchase/transaction, frequency
# tx --> Last Transaction Date - First Transaction Date (Weekly)
# T --> Today Date - first transaction/purchase Date. (Customer's Age) (Weekly)
# r,a --> Gamma Dist. parameters
# a,b --> Beta Dist. parameters

# Expected Average Order Profit With Gamma Gamma Submodel:
# Used to estimate how much profit a customer can generate on average per transaction
# The monetary value of a customer's transactions is the average of transaction values. scattered randomly around
# Average transaction value is gamma distributed across all customers
# x--> Frequency Value
# mx --> Monetary (observed transaction values)


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

import pandas as pd
import datetime as dt
import lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import matplotlib.pyplot as plt
pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
print(df.head())
print(df.describe().T)
print(df.info())

def outlier_thresholds(dataframe, variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    range = quantile3 - quantile1
    upper_limit = quantile3 + 1.5 * range
    lower_limit = quantile1 - 1.5 * range
    return lower_limit, upper_limit

def replace_with_thresholds(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit



df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")
print(df.describe().T)

df["total_price"] = df["Price"] * df["Quantity"]


# x --> Number of purchase/transaction, frequency
# tx --> Last Transaction Date - First Transaction Date (Weekly)
# T --> Today Date - first transaction/purchase Date. (Customer's Age) (Weekly)
print(df["InvoiceDate"].max())
today = dt.datetime(year=df["InvoiceDate"].dt.year.max(),
                    month=df["InvoiceDate"].dt.month.max(),
                    day=11)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (today - x.min()).days,     # T, Customer Age
                                                         lambda x: (x.max() - x.min()).days],  # xt, Recency
                                         "Invoice": "nunique",   # Frequency
                                         "total_price": "sum"})  # Monetary

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["T", "Recency", "Frequency", "Monetary"]

cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]
cltv_df["T"] = cltv_df["T"] / 7
cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df = cltv_df[cltv_df["Frequency"] > 1]
print(cltv_df)

# Creating #BG-NBD Model:
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"])
print(bgf)

# Who are the 10 customers from whom we expect the most purchases in 1 week? and add it as a new variable in cltv
cltv_df["expected_purc_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                                          cltv_df["Frequency"],
                                                                                          cltv_df["Recency"],
                                                                                          cltv_df["T"])

print(bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                cltv_df["Frequency"],
                                                                cltv_df["Recency"],
                                                                cltv_df["T"]).sort_values(ascending=False).head(10))


# Who are the 10 customers from whom we expect the most purchases in 1 month? and add it as a new variable in cltv

print(bgf.conditional_expected_number_of_purchases_up_to_time(4*1,
                                                                cltv_df["Frequency"],
                                                                cltv_df["Recency"],
                                                                cltv_df["T"]).sort_values(ascending=False).head(10))



cltv_df["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*1,
                                                                                          cltv_df["Frequency"],
                                                                                          cltv_df["Recency"],
                                                                                          cltv_df["T"])
#Who are the 5 customers from whom we expect the most purchases in 3 months? and add it as a new variable in cltv

print(bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                              cltv_df["Frequency"],
                                                              cltv_df["Recency"],
                                                              cltv_df["T"]).sort_values(ascending=False).head(5))

cltv_df["expected_purc_3_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                            cltv_df["Frequency"],
                                                                                            cltv_df["Recency"],
                                                                                            cltv_df["T"])

# You can use Predict instead of Conditional_expected_number_of_purchases_tp_to_time function!!!

# Evaluation of Forecast Results:
plot_period_transactions(bgf)
plt.grid()
print(plt.show())

# GAMMA GAMMA Model:
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])

ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                        cltv_df["Monetary"])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"], cltv_df["Monetary"])

print(cltv_df.sort_values("expected_average_profit", ascending=False).head(10))

# Creating cltv:
customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                      cltv_df["Frequency"],
                                                      cltv_df["Recency"],
                                                      cltv_df["T"],
                                                      cltv_df["Monetary"],
                                                      time=3,   # months
                                                      freq="W", # Week
                                                      discount_rate=0.01)

print(customer_lifetime_value.head())
customer_lifetime_value = customer_lifetime_value.reset_index()
cltv_final = cltv_df.merge(customer_lifetime_value, on="Customer ID", how="left")
print(cltv_final.sort_values("clv", ascending=False).head(10))


# Creating Segments:

cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
print(cltv_final.sort_values("clv", ascending=False).head(10))
print(cltv_final.groupby("Segment").agg({"mean", "sum", "count"}))




# Functionalization of the Entire Process:

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    range = q3 - q1
    upper_limit = q3 + 1.5 * range
    lower_limit = q1 - 1.5 * range
    return lower_limit, upper_limit

def replace_with_thresholds(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
def create_cltvfinal(dataframe, time=3):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Price"] > 0]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Price")
    replace_with_thresholds(dataframe, "Quantity")
    dataframe["total_price"] = dataframe["Price"] * dataframe["Quantity"]

    today = dt.datetime(year=dataframe["InvoiceDate"].dt.year.max(),
                        month=dataframe["InvoiceDate"].dt.month.max(),
                        day=11)

    cltv = dataframe.groupby("Customer ID").agg({"InvoiceDate":[lambda x: (today - x.min()).days,     # Customer Age T
                                                                lambda x: (x.max() - x.min()).days],  # Recency tx
                                                 "Invoice": "nunique",                                # Frequency
                                                 "total_price": "sum"})                               # Monetary

    cltv.columns = cltv.columns.droplevel(0)
    cltv.columns = ["T", "Recency", "Frequency", "Monetary"]

    cltv["Monetary"] = cltv["Monetary"] / cltv["Frequency"]
    cltv = cltv[cltv["Frequency"] > 1]
    cltv["Recency"] = cltv["Recency"] / 7
    cltv["T"] = cltv["T"] / 7

    print(cltv)

    # BG-NBD Modelinin Kurulması:
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv["Frequency"], cltv["Recency"], cltv["T"])

    cltv["expected_purch_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                                            cltv["Frequency"],
                                                                                            cltv["Recency"],
                                                                                            cltv["T"])

    cltv["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                                                            cltv["Frequency"],
                                                                                            cltv["Recency"],
                                                                                            cltv["T"])

    cltv["expected_purc_3_month"] = bgf.predict(4*3,
                                                cltv["Frequency"],
                                                cltv["Recency"],
                                                cltv["T"])

    # Gamma Gamma Modelinin Kurulması:
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv["Frequency"], cltv["Monetary"])

    cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv["Frequency"], cltv["Monetary"])

    # BG-NBD ve GG Modeli İle CLTV'nin Hesaplanması:

    customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                          cltv["Frequency"],
                                                          cltv["Recency"],
                                                          cltv["T"],
                                                          cltv["Monetary"],
                                                          time=time,
                                                          discount_rate=0.01,
                                                          freq="W")
    customer_lifetime_value = customer_lifetime_value.reset_index()
    cltv_final = cltv.merge(customer_lifetime_value, on="Customer ID", how="left")

    cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

print(create_cltvfinal(df))
print(create_cltvfinal(df).to_csv("cltv_prediction.csv"))