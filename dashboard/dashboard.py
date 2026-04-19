import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data.csv")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

all_df = load_data()

with st.sidebar:
    st.title("E-Commerce Dashboard")
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=all_df["order_purchase_timestamp"].min(),
        max_value=all_df["order_purchase_timestamp"].max(),
        value=[all_df["order_purchase_timestamp"].min(), all_df["order_purchase_timestamp"].max()]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                 (all_df["order_purchase_timestamp"] <= str(end_date))]

st.header('E-Commerce Analysis Dashboard')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Orders", value=main_df.order_id.nunique())
with col2:
    st.metric("Total Revenue", value=f"BRL {main_df.price.sum():,.2f}")
with col3:
    st.metric("Total Customers", value=main_df.customer_unique_id.nunique())

st.divider()

# --- PERTANYAAN 1: Bagaimana performa pendapatan (revenue) dari berbagai kategori produk selama periode tahun 2018, dan kategori mana yang memberikan kontribusi paling signifikan? ---
st.subheader("Top 10 Product Categories by Revenue")
category_rev = main_df.groupby("product_category_name_english").price.sum().sort_values(ascending=False).head(10).reset_index()

fig_rev, ax_rev = plt.subplots(figsize=(10, 5))
sns.barplot(x="price", y="product_category_name_english", data=category_rev, palette="viridis", ax=ax_rev)
ax_rev.set_xlabel("Revenue (BRL)")
ax_rev.set_ylabel(None)
st.pyplot(fig_rev)

# --- PERTANYAAN 2: Bagaimana segmentasi profil pelanggan berdasarkan perilaku belanja menggunakan metode analisis RFM (Recency, Frequency, Monetary) untuk merancang strategi pemasaran yang lebih personal? ---
st.subheader("Best Customer Based on RFM Parameters")

snapshot_date = main_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
rfm_df = main_df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days, # Recency
    'order_id': 'nunique',                                               # Frequency
    'price': 'sum'                                                       # Monetary
}).reset_index()

rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]

# Plot Recency
sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Customer ID", fontsize=15)
ax[0].set_title("By Recency (days)", loc="center", fontsize=25)
ax[0].tick_params(axis='x', rotation=45, labelsize=12)

# Plot Frequency
sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Customer ID", fontsize=15)
ax[1].set_title("By Frequency", loc="center", fontsize=25)
ax[1].tick_params(axis='x', rotation=45, labelsize=12)

# Plot Monetary
sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("Customer ID", fontsize=15)
ax[2].set_title("By Monetary", loc="center", fontsize=25)
ax[2].tick_params(axis='x', rotation=45, labelsize=12)

plt.tight_layout()
st.pyplot(fig)

# --- PERTANYAAN 3: Bagaimana tren performa penjualan perusahaan dari bulan ke bulan selama tahun 2018? ---
st.subheader("Monthly Sales Trend")
monthly_trend = main_df.resample(rule='ME', on='order_purchase_timestamp').order_id.nunique().reset_index()
monthly_trend['order_month'] = monthly_trend['order_purchase_timestamp'].dt.strftime('%b %Y')

fig_trend, ax_trend = plt.subplots(figsize=(12, 5))
ax_trend.plot(monthly_trend['order_month'], monthly_trend['order_id'], marker='o', linewidth=2, color="#72BCD4")
plt.xticks(rotation=45)
st.pyplot(fig_trend)

st.markdown("#### Customer Segmentation Proportion")

rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*100
rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*100
rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*100

rfm_df['rfm_score'] = rfm_df[['r_rank_norm', 'f_rank_norm', 'm_rank_norm']].mean(axis=1)

rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

# Binning menjadi Segmen
rfm_df['customer_segment'] = pd.cut(rfm_df['rfm_score'],
                                    bins=[0, 20, 50, 80, 100],
                                    labels=['Hibernating', 'At Risk', 'Loyalist', 'Champions'],
                                    include_lowest=True)

segment_counts = rfm_df['customer_segment'].value_counts().sort_index()

fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
colors_pie = sns.color_palette("viridis", len(segment_counts))

ax_pie.pie(
    segment_counts,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_pie,
    pctdistance=0.85,
    explode=[0.05] * len(segment_counts)
)

centre_circle = plt.Circle((0,0), 0.70, fc='white')
ax_pie.add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
st.pyplot(fig_pie)

st.caption('Copyright © Fanidyasani Atantya 2026')