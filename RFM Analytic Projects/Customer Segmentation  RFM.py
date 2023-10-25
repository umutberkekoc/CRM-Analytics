# Customer Segmentation-RFM:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 700)
pd.set_option("display.max_rows", 700)
df_ = pd.read_csv("customer_segmentation_100k.csv")
df = df_.copy()

# General Informations:
df.head()
df.shape
df.size
df.info()
df.columns
df.describe().T
df.isnull().sum()

# Data Preperation:

df = df[df["qtt_order"] > 0]
df = df[df["total_spent"] > 0]
df["last_order"] = df["last_order"].astype("datetime64[ns]")
df.info()
df.describe().T

df["last_order"].max()

today = dt.datetime(year=df["last_order"].dt.year.max(),
                    month=int(input("Enter a month (last month:01)")),
                    day=int(input("enter a day (last day: 19)")))

df["customer_id"].nunique()
df["qtt_order"].nunique()
# rfm matriklerinin oluşturulması:
rfm = df.groupby("customer_id").agg({"last_order": lambda x: (today - x.max()).days, # recency
                                     "qtt_order": "sum",  # frequency
                                     "total_spent": "sum"})   # monetary

rfm.head()
rfm.describe().T

rfm.columns = ["Recency", "Frequency", "Monetary"]

# rfm skorlarının hesaplanması
rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

rfm.head()

# segmentlerin oluşturulması:
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
rfm.sort_values("rf_score", ascending=False).head(20)

rfm["rf_score"] = rfm["rf_score"].astype(int)
rfm.info()

rfm.groupby("segment").agg({"Recency": "mean",
                            "Frequency": "mean",
                            "Monetary": ["mean", "count"],
                            "rf_score": ["max", "min", "std", "mean"]})

rfm.index = rfm.index.astype(int)

new_customers_id = rfm[rfm["segment"] == "new_customers"].index
champions_id = rfm[rfm["segment"] == "new_customers"].index

new_df = pd.DataFrame()
new_df["new_customers_id"] = new_customers_id
new_df["champions_id"] = champions_id

print(new_df[["new_customers_id", "champions_id"]].to_csv("new_customers_&champions_id.csv"))


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
