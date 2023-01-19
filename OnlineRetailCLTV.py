# pip install lifetimes
# pip install sqlalchemy
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import helpers.helpers as helpers

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = df[df["Country"] == "United Kingdom"]

helpers.check_df(df)

df.describe().T

df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]

df.head()

replace_with_thresholds(df, "Quantity", q1=0.01, q3=0.99)
replace_with_thresholds(df, "Price", q1=0.01, q3=0.99)

df["TotalPrice"] = df["Price"] * df["Quantity"]

df = df[df["TotalPrice"] > 0]

today_date = dt.datetime(2011, 12, 11)

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma ve ilk satın alma arasındaki fark. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# monetary değerinin satın alma başına ortalama kazanç olarak ifade edilmesi
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# monetary sıfırdan büyük olanların seçilmesi
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# BGNBD için recency ve T'nin haftalık cinsten ifade edilmesi
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency'nin 1'den büyük olması gerekmektedir.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

##############################################################
#  BG-NBD Modelinin Kurulması
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])



##############################################################
#  GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

##############################################################
# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

#  1 - 6 aylık CLTV Prediction
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
df[df["Customer ID"] == 12747.0000]["InvoiceDate"].max()

cltv_df["last_buy_day"] = (cltv_df["T"] - cltv_df["recency"]) * 7
cltv_df = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary', 'expected_purc_6_month', 'expected_average_profit', 'clv_6_month']
cltv_df["monetary*freq"] = cltv_df["expected_average_profit"] * cltv_df["expected_purc_6_month"]
cltv_df["monetary*freq+monetary"] = (cltv_df["expected_average_profit"] * cltv_df["expected_purc_6_month"]) + cltv_df["expected_average_profit"]

cltv_df.sort_values(by="clv_6_month", ascending=False).head(20)

cltv_df

df[df["Customer ID"] == 18139.0000].groupby("Invoice").agg({"InvoiceDate": lambda x: x})
##  1 VE 12 AYLIK CLTV Prediction

cltv_1_month = ggf.customer_lifetime_value(bgf,
                                           cltv_df['frequency'],
                                           cltv_df['recency'],
                                           cltv_df['T'],
                                           cltv_df['monetary'],
                                           time=1,  # 1 aylık
                                           freq="W",  # T'nin frekans bilgisi.
                                           discount_rate=0.01)

cltv_12_month = ggf.customer_lifetime_value(bgf,
                                            cltv_df['frequency'],
                                            cltv_df['recency'],
                                            cltv_df['T'],
                                            cltv_df['monetary'],
                                            time=12,  # 12 aylık
                                            freq="W",  # T'nin frekans bilgisi.
                                            discount_rate=0.01)

# Kıyaslama
cltv_1_month.sort_values(by="clv_1_month", ascending=False).head(10)
cltv_12_month.sort_values(by="clv_12_month", ascending=False).head(10)

# Kıyaslama yaptığımızda sıralama hemen hemen aynı ama 14088 Id li müşterini 1 aylık cltv değeri 13694 Id li
# müşterini 1 aylık ctlv değerinden fazlayken 12 aylık cltv değerlerine baktığımızda 13694 Id li
# müşterini değeri 14088 Id li müşteriden daha fazla olduğu görünüyor. Bunu sebebi müşterilerin satın alma
# sıklıkları ve satın alma örüntüleri olabilir.
cltv_df2 = cltv_df.reset_index()
cltv_df2[(cltv_df2["CustomerID"] == 14088) | (cltv_df2["CustomerID"] == 13694)]

# Bu 2 müşterinin lifetime value larını birlikte incelediğimizde incelediğimizde
# freqeuncy ve monetary değerleri durumu çok iyi açıklıyor. 14088 Id li müşterisin average order profiti daha yüksek
# ama frequency si az yani bu müşteri bizden az hizmet alıyor ama her geldiğinde iyi para bırakıyor
# diğer müşteri ise çok fazla hizmet alıyor ama az para bırakıyor. Frequency ve monetary değerlerini çarptığımızda
# 13694 Id li müşterinin toplam bıraktığı paranın daha fazla olduğunu görürüz.


## Segmentasyon ve Aksiyon Önerileri


cltv = cltv.reset_index()
cltv.head()

cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.groupby("segment").agg({"count", "mean", "sum"})

# 6 Ay içerisinde A ve B segmentlerinden kazanacağım tahmin edilen değer.

SEGMENT_A = cltv_final[cltv_final["segment"] == "A"]["clv"].sum()
SEGMENT_B = cltv_final[cltv_final["segment"] == "B"]["clv"].sum()
SEGMENT_A + SEGMENT_B

# Aksiyon önerileri ==>>  Bizi unutmamaları ve
# Tahmin edilen değerin altında bir kazanç elde etmememek için müşteriler alışkanlıklarını devam ettirmeliler.
# En değerli bulduğum A ve B segmentindeki müşterileri ayık tutulması gerek. Bizi unutmamaları için kampanyalarımızdan
# haberdar etmeliyiz. Daha önce aldığı hizmetlere benzer hizmetlerimizi önerebiliriz bu sayede tahmin edilen fazla kazancımız olabilir.





