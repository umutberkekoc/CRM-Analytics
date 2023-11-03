###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################
import pandas as pd

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# 1. flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
# 2. Veri setinde
# a. İlk 10 gözlem,
print(df.head(10))
# b. Değişken isimleri,
print(df.columns)
# c. Betimsel istatistik,
print(df.describe([0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
# d. Boş değer,
print(df.isnull().sum().any())
print(df.isnull().sum())
# e. Değişken tipleri, incelemesi yapınız.
print(df.info())

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
for i in df.columns:
    if "date" in i:
        df[i] = df[i].astype("datetime64[ns]")  # 1.way

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].astype("datetime64[ns]")  # 2.way
print(df.info())

# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
print(df.groupby("order_channel").agg({"master_id": "count",
                                       "total_transaction": ["mean", "sum"],
                                       "total_customer_value": ["mean", "sum"]}))

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
print(df.groupby("master_id").agg({"total_customer_value": "sum"}).sort_values("total_customer_value",
                                                                               ascending=False).head(10))  # 1.way
print(
    df.pivot_table("total_customer_value", "master_id", aggfunc=["sum"]).sort_values(by=("sum", "total_customer_value"),
                                                                                     ascending=False).head(10))  # 2.way

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
print(df.groupby("master_id").agg({"total_transaction": "sum"}).sort_values("total_transaction", ascending=False).head(
    10))  # 1.way
print(df.pivot_table("total_transaction", "master_id", aggfunc=["sum"]).sort_values(by=("sum", "total_transaction"),
                                                                                    ascending=False).head(10))  # 2.way

# GÖREV 2: RFM Metriklerinin Hesaplanması
print(df["last_order_date"].max())
today = dt.datetime(year=df["last_order_date"].dt.year.max(), month=6, day=1)
rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today - x.max()).days,  # Recency
                                   "total_transaction": "sum",  # Frequency
                                   "total_customer_value": "sum"})  # Monetary

rfm.columns = ["recency", "frequency", "monetary"]
print(rfm.head())
print(rfm.describe().T)

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
seg_map = {
    # (1.row recency, 2.row frequency)
    r'[1-2][1-2]': "hibernating",  # 1 veya 2 ile başlayan (recency) ve 1 veya 2 ile biten (frequency)
    r'[1-2][3-4]': "at_risk",
    r'[1-2]5': "cant_loose",  # 1 veya 2 ile başlayan ve 5 ile biten
    r'3[1-2]': "about_to_sleep",  # 3 ile başlayan ve 1 veya 2 ile biten
    r'33': "need_attention",
    r'[3-4][4-5]': "loyal_customers",
    r'41': "promising",
    r'51': "new_customers",  # 5 ile başlayan ve 1 ile biten
    r'[4-5][2-3]': "potential_loyalists",
    r'5[4-5]': "champions",
}

rfm["segments"] = rfm["rf_score"].replace(seg_map, regex=True)
print(rfm.head())

# GÖREV 5: Aksiyon zamanı!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
print(rfm.groupby("segments").agg({"recency": "mean",
                                   "frequency": "mean",
                                   "monetary": "mean"}))  # 1.way

print(rfm.pivot_table(index="segments", values=["recency", "frequency", "monetary"], aggfunc="mean"))  # 2.way
# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
# ortalama 250 TL üzeri ve "sadece kadın" kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
rfm.reset_index(inplace=True)
final_df = rfm.merge(df[["master_id", "interested_in_categories_12"]], on='master_id', how="left")
print(final_df[((final_df["segments"] == "champions") | (final_df["segments"] == "loyal_customers")) &
         (final_df["interested_in_categories_12"].str.contains("KADIN")) & (~final_df["interested_in_categories_12"].str.contains(",")) &
               (final_df["monetary"] > 250)])

customer_id = final_df[((final_df["segments"] == "champions") | (final_df["segments"] == "loyal_customers")) &
         (final_df["interested_in_categories_12"].str.contains("KADIN")) & (~final_df["interested_in_categories_12"].str.contains(",")) &
               (final_df["monetary"] > 250)].index

index = pd.DataFrame()
index["customer_id"] = customer_id
index.to_csv("yeni_marka_hedef_müşteri_id.cvs")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv olarak kaydediniz.

print(final_df[((final_df["segments"].isin(["cant_loose", "about_to_sleep", "new_customers"])) &
                    (final_df["interested_in_categories_12"].str.contains("ERKEK|COCUK")) &
                    (~final_df["interested_in_categories_12"].str.contains(",")))])

customer_id2 = final_df[((final_df["segments"].isin(["cant_loose", "about_to_sleep", "new_customers"])) &
                    (final_df["interested_in_categories_12"].str.contains("ERKEK|COCUK")) &
                    (~final_df["interested_in_categories_12"].str.contains(",")))].index

index["customer_id2"] = customer_id3
index.to_csv("indirim_hedef_müşteri_ids.csv")

# Customer Lifetime Value:
# CLTV = (Customer Value / Churn Rate) * Profit Margin
# Customer Value = Average Order Value * Purchase Frequency
# Average Order Value = Total Price / Total Transaction
# Purchase Frequency = Total Transaction / Total Number of Customers
# Churn Rate (Dropout Rate) = 1 - Repeat Rate
# Repeat Rate = Number of customer for more than 1 transaction / total number of customers
# Profit Margin = Total Price * '0.10' (Profit rate takes by 0.10 for worse scenario)

print(df.head())
print(df.info())
print(df.size)
print(df.shape)
print(df.columns)
print(df.describe().T)
print(df.isnull().sum())

cltv = df.groupby("master_id").agg({"total_transaction": "sum",  # Frequency
                                    "total_customer_value": "sum",  # Monetary
                                    })
print(cltv.head())

repeat_rate = cltv[cltv["total_transaction"] > 1].shape[0] - cltv.shape[0]
churn_rate = 1 - repeat_rate

cltv["profit_margin"] = 0.10 * cltv["total_customer_value"]

cltv["purchase_frequency"] = cltv["total_transaction"] / cltv.shape[0]

cltv["average_order_value"] = cltv["total_customer_value"] / cltv["total_transaction"]

cltv["customer_value"] = cltv["average_order_value"] * cltv["purchase_frequency"]

cltv["clv"] = cltv["customer_value"] / churn_rate * cltv["profit_margin"]

cltv["segments"] = pd.qcut(cltv["clv"], 4, labels=["D", "C", "B", "A"])
print(cltv.sort_values("clv", ascending=False).head(20))
# Segmentlere göre analiz yapınız
print(cltv.groupby("segments").agg({"mean", "sum", "count"}))
# Segmenti "D" olanları gösteriniz
print(cltv[cltv["segments"] == "D"])
# her bir segmentteki kişi sayısını bulunuz
print(cltv["segments"].value_counts())
# En yüksek ve en düşük değere sahip müşteri bilgilerini getiriniz
print(cltv[cltv["clv"] == cltv["clv"].max()])
print(cltv[cltv["clv"] == cltv["clv"].min()])
# son dataframei bir csv dosyasına atınız
# print(cltv.to_csv("cltv.csv"))

# Her bir Segmentteki Kişi sayısını görselleştiriniz
sns.countplot(data=cltv, x="segments", palette="viridis")
plt.title("Number of Customer by Segments")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())
# Her bir segmentteki ortalama frekans ve monetary değerlerini görselleştiriniz
sns.barplot(data=cltv, x="segments", y="total_transaction", estimator="mean",
            palette=["black", "gray", "purple", "blue"])
plt.title("Segmentlerin Ortalama Frekansı")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

sns.barplot(data=cltv, x="segments", y="total_customer_value", estimator="mean",
            palette=["pink", "yellow", "orange", "red"])
plt.title("Segmentlerin Ortalama Monetary Değerleri")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

# Customer Lifetime Value Prediction:

# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
print(df.head())
print(df.describe().T)
# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    range = quantile3 - quantile1
    upper_limit = round(quantile3 + 1.5 * range)
    lower_limit = round(quantile1 - 1.5 *range)
    return lower_limit, upper_limit

def replace_with_thresholds(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > upper_limit, variable] = upper_limit
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.
degisken = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for i in degisken:
    replace_with_thresholds(df, i)
print(df.describe().T)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
for i in df.columns:
    if "date" in i:
        df[i] = df[i].astype("datetime64[ns]")

print(df.info())

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
print(df["last_order_date"].max())
analiz_tarihi = dt.datetime(year=df["last_order_date"].dt.year.max(), month=6, day=1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
# Recency (tx) --> son satın alma tarihi - ilk satın alma tarihi (haftalık)
# T --> CUSTOMER AGE --> Analiz tarihi - ilk satın alma tarihi (haftalık)
# frequency --> toplam satın alam sayısı
# Monetary --> Ortalama harcanan para
cltv_df = df.groupby("master_id").agg({"first_order_date": lambda x: (analiz_tarihi - x.min()).days,  # T
                                        "total_transaction": "sum",  # frequency
                                        "total_customer_value": "sum"})  # monetary

cltv_df["recency"] = (df.groupby("master_id")["last_order_date"].max() - df.groupby("master_id")["first_order_date"].min()).dt.days
cltv_df.columns = ["T_weekly", "frequency", "monetary_cltv_df_avg", "recency_cltv_df_weekly"]
print(cltv_df.head())

cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7
cltv_df["recency_cltv_df_weekly"] = cltv_df["recency_cltv_df_weekly"] / 7
cltv_df["monetary_cltv_df_avg"] = cltv_df["monetary_cltv_df_avg"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]
print(cltv_df.head())
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_df_weekly"], cltv_df["T_weekly"])

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                       cltv_df["frequency"],
                                                                                       cltv_df["recency_cltv_df_weekly"],
                                                                                       cltv_df["T_weekly"])

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                                                       cltv_df["frequency"],
                                                                                       cltv_df["recency_cltv_df_weekly"],
                                                                                       cltv_df["T_weekly"])

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_df_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_df_avg"])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
customer_lifetime_value = ggf.customer_lifetime_value(bgf, cltv_df["frequency"],
                                                      cltv_df["recency_cltv_df_weekly"],
                                                      cltv_df["T_weekly"],
                                                      cltv_df["monetary_cltv_df_avg"],
                                                      time=6,
                                                      freq="W",
                                                      discount_rate=0.01)

cltv_df["cltv"] = customer_lifetime_value
print(cltv_df.head())
# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
print(cltv_df.sort_values("cltv", ascending=False).head(20))

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head(10)
print(cltv_df.sort_values("cltv_segment", ascending=False)[:10])


# BONUS: Tüm süreci fonksiyonlaştırınız.

def cltv(dataframe, time=3):
    degisken = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
                "customer_value_total_ever_online"]
    for i in degisken:
        replace_with_thresholds(df, i)
    print(df.describe().T)

    # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
    # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

    dataframe["total_expense"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    dataframe["total_transaction"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]

    # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
    '''df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
    df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
    df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
    df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")'''

    for i in dataframe.columns:
        if "date" in i:
            dataframe[i] = dataframe[i].astype("datetime64[ns]")



    #print(dataframe["last_order_date"].max())
    today = dt.datetime(year=dataframe["last_order_date"].dt.year.max(), month=6, day=1)

    # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
    # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
    # T, Customer Age --> today - last transaction date
    # tx, Recency --> last transaction date - first transaction date

    cltv_df = dataframe.groupby("master_id").agg({"first_order_date": lambda x: (today - x.min()).days,  # T
                                           "total_transaction": "sum",  # frequency
                                           "total_expense": "sum"})  # monetary

    cltv_df["recency"] = (dataframe.groupby("master_id")["last_order_date"].max() - dataframe.groupby("master_id")[
        "first_order_date"].min()).dt.days

    cltv_df.columns = ["T_weekly", "frequency", "monetary_cltv_avg", "recency_cltv_weekly"]


    cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"] / cltv_df["frequency"]
    cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7
    cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7
    cltv_df = cltv_df[cltv_df["frequency"] > 1]  # şart mı


    # GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
    # 1. BG/NBD modelini fit ediniz.
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])

    # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

    cltv_df["exp_sales_3_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                                            cltv_df["frequency"],
                                                                                            cltv_df[
                                                                                                "recency_cltv_weekly"],
                                                                                            cltv_df["T_weekly"])

    # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

    cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df[
                                                                                               "recency_cltv_weekly"],
                                                                                           cltv_df["T_weekly"])

    # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                           cltv_df["monetary_cltv_avg"])

    # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
    customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                          cltv_df["frequency"],
                                                          cltv_df["recency_cltv_weekly"],
                                                          cltv_df["T_weekly"],
                                                          cltv_df["monetary_cltv_avg"],
                                                          time=time,
                                                          freq="W")
    cltv_df["cltv"] = customer_lifetime_value
    cltv_df = cltv_df.reset_index()



    # GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
    # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
    print(cltv_df.head())

    # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
    print(cltv_df.groupby("cltv_segment").agg({"frequency": "mean",
                                               "recency_cltv_weekly": "mean",
                                               "monetary_cltv_avg": "mean",
                                               "T_weekly": "mean"}))
cltv(df,6)