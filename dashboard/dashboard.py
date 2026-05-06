import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import numpy as np

st.set_page_config(page_title="E-Commerce Insights 🛍️", layout="wide")

@st.cache_data

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    path_data = os.path.join(base_dir, "main_data.csv")
    
    df = pd.read_csv(path_data)
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

all_df = load_data()

def format_big_number(num):
    if num >= 1_000_000:
        return f"BRL {num / 1_000_000:.2f}M"  # Jadi 13.22M
    elif num >= 1_000:
        return f"BRL {num / 1_000:.1f}K"   # Jadi 15.5K
    return f"BRL {num:.2f}"

with st.sidebar:
    st.title("E-Commerce Dashboard ✨")
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png")
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu 📅',
        min_value=all_df["order_purchase_timestamp"].min(),
        max_value=all_df["order_purchase_timestamp"].max(),
        value=[all_df["order_purchase_timestamp"].min(), all_df["order_purchase_timestamp"].max()]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) & 
                 (all_df["order_purchase_timestamp"] <= pd.to_datetime(end_date))]

st.header('E-Commerce Analysis Dashboard 📊')
col1, col2, col3 = st.columns(3)
with st.container():
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(
            label="📦 Total Orders", 
            value=f"{main_df.order_id.nunique():,}"
        )
    
    with m2:
        total_rev = main_df.price.sum()
        st.metric(
            label="💰 Total Revenue", 
            value=format_big_number(total_rev),
            help=f"Nilai asli: BRL {total_rev:,.2f}"
        )
    
    with m3:
        st.metric(
            label="👥 Total Customers", 
            value=f"{main_df.customer_unique_id.nunique():,}"
        )

st.divider()
tab_analysis, tab_eda = st.tabs(["📈 Performa Bisnis", "📍 Lokasi & Operasional"])
with tab_analysis:
    # --- PERTANYAAN 1: Bagaimana performa pendapatan (revenue) dari berbagai kategori produk selama periode tahun 2018, dan kategori mana yang memberikan kontribusi paling signifikan? ---
    st.subheader("Product Performance Analysis 📈")

    col_grafik, col_insight = st.columns([1.2, 0.8])

    with col_grafik:
        category_rev = main_df.groupby("product_category_name_english").price.sum().sort_values(ascending=False).head(10).reset_index()
        colors_category = ["#003f5c" if i == 0 else "#a2d2ff" for i in range(len(category_rev))]

        fig_rev, ax_rev = plt.subplots(figsize=(5, 3.5))
        sns.barplot(
            x="price", 
            y="product_category_name_english", 
            data=category_rev, 
            palette=colors_category,
            ax=ax_rev
        )

        # Styling
        ax_rev.set_title("Top 10 Categories by Revenue", fontsize=10)
        ax_rev.set_xlabel("Total Revenue (BRL)", fontsize=8)
        ax_rev.set_ylabel(None)
        ax_rev.tick_params(axis='y', labelsize=8)
        ax_rev.tick_params(axis='x', labelsize=7)
        
        sns.despine()
        plt.tight_layout()

        st.pyplot(fig_rev)

    with col_insight:
        st.write("#### 💡 Quick Insights")
        top_cat = category_rev.iloc[0]['product_category_name_english']
        top_val = category_rev.iloc[0]['price']
        
        st.info(f"""
        Kategori **{top_cat}** merupakan kontributor pendapatan terbesar dengan total 
        **BRL {top_val:,.2f}**. 
        
        Strategi stok dan promosi harus diprioritaskan pada 10 kategori ini karena 
        dominansinya terhadap total revenue perusahaan.
        """)
        
        st.metric("Top Category Share", value="High", delta="Significant")

    # --- PERTANYAAN 2: Bagaimana segmentasi profil pelanggan berdasarkan perilaku belanja menggunakan metode analisis RFM (Recency, Frequency, Monetary) untuk merancang strategi pemasaran yang lebih personal? ---
    st.subheader("Best Customer Based on RFM Parameters 🙋🏻‍♂️")

    snapshot_date = main_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm_df = main_df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',                                              
        'price': 'sum'                                                      
    }).reset_index()

    rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    colors_rfm = ["#003f5c" if i == 0 else "#a2d2ff" for i in range(5)]

    # Plot Recency
    rfm_recency = rfm_df.sort_values(by="recency", ascending=True).head(5)
    rfm_recency['customer_id_short'] = rfm_recency['customer_id'].str[-5:]
    sns.barplot(y="recency", x="customer_id_short", data=rfm_recency, palette=colors_rfm, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Customer ID (Suffix)", fontsize=12)
    ax[0].set_title("By Recency (days)", loc="center", fontsize=16)
    ax[0].tick_params(axis='x', labelsize=10)

    # Plot Frequency
    rfm_frequency = rfm_df.sort_values(by="frequency", ascending=False).head(5)
    rfm_frequency['customer_id_short'] = rfm_frequency['customer_id'].str[-5:]
    sns.barplot(y="frequency", x="customer_id_short", data=rfm_frequency, palette=colors_rfm, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Customer ID (Suffix)", fontsize=12)
    ax[1].set_title("By Frequency", loc="center", fontsize=16)
    ax[1].tick_params(axis='x', labelsize=10)

    # Plot Monetary
    rfm_monetary = rfm_df.sort_values(by="monetary", ascending=False).head(5)
    rfm_monetary['customer_id_short'] = rfm_monetary['customer_id'].str[-5:]
    sns.barplot(y="monetary", x="customer_id_short", data=rfm_monetary, palette=colors_rfm, ax=ax[2])
    ax[2].set_ylabel(None)
    ax[2].set_xlabel("Customer ID (Suffix)", fontsize=12)
    ax[2].set_title("By Monetary", loc="center", fontsize=16)
    ax[2].tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # --- PERTANYAAN 3: Bagaimana tren performa penjualan perusahaan dari bulan ke bulan selama tahun 2018? ---
    st.subheader("Monthly Sales Trend 🗓️")

    monthly_trend = main_df.resample(rule='ME', on='order_purchase_timestamp').order_id.nunique().reset_index()
    monthly_trend['order_month'] = monthly_trend['order_purchase_timestamp'].dt.strftime('%b %Y')

    col_chart, col_insight = st.columns([1.2, 0.8])

    with col_chart:
        fig_trend, ax_trend = plt.subplots(figsize=(6, 3.5))
        
        ax_trend.plot(
            monthly_trend['order_month'], 
            monthly_trend['order_id'], 
            marker='o', 
            linewidth=2, 
            color="#72BCD4",
            markersize=5
        )
        
        # Styling
        ax_trend.set_ylabel("Total Orders", fontsize=8)
        ax_trend.tick_params(axis='x', rotation=45, labelsize=8)
        ax_trend.tick_params(axis='y', labelsize=8)
        ax_trend.grid(axis='y', linestyle='--', alpha=0.4)
        sns.despine()
        
        plt.tight_layout()
        st.pyplot(fig_trend)

    with col_insight:
        st.write("#### 💡 Business Insight")

        total_orders = monthly_trend['order_id'].sum()
        peak_month = monthly_trend.loc[monthly_trend['order_id'].idxmax(), 'order_month']
        
        st.info(f"""
        - **Total Transaksi:** {total_orders:,} pesanan berhasil tercatat.
        - **Puncak Performa:** Penjualan tertinggi terjadi pada bulan **{peak_month}**.
        - **Saran:** Perhatikan tren musiman di bulan tersebut untuk mempersiapkan stok dan kampanye marketing di tahun berikutnya.
        """)

    st.divider()

    st.markdown("### Customer Segmentation Proportion 🧩")

    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

    rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*100

    rfm_df['rfm_score'] = rfm_df[['r_rank_norm', 'f_rank_norm', 'm_rank_norm']].mean(axis=1)

    # 2. BINNING MENJADI SEGMEN
    rfm_df['customer_segment'] = pd.cut(rfm_df['rfm_score'],
                                        bins=[0, 20, 50, 80, 100],
                                        labels=['Hibernating', 'At Risk', 'Loyalist', 'Champions'],
                                        include_lowest=True)

    segment_counts = rfm_df['customer_segment'].value_counts().sort_values(ascending=True)
    total_customers = segment_counts.sum()

    col_chart, col_info = st.columns([1.3, 0.7])

    with col_chart:
        fig_seg, ax_seg = plt.subplots(figsize=(6, 4))
        colors_segment = ["#a2d2ff", "#72BCD4", "#134e6f", "#003f5c"] 
        
        bars = ax_seg.barh(segment_counts.index, segment_counts.values, color=colors_segment)
        
        for bar in bars:
            width = bar.get_width()
            percentage = (width / total_customers) * 100
            ax_seg.text(
                width + (total_customers * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', 
                va='center', fontsize=9, fontweight='bold'
            )

        ax_seg.set_xlabel("Number of Customers", fontsize=8)
        ax_seg.set_title("Distribution by Segment", fontsize=10)
        ax_seg.spines['top'].set_visible(False)
        ax_seg.spines['right'].set_visible(False)
        ax_seg.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_seg)

    with col_info:
        st.write("#### 📋 Statistik")
        df_info = segment_counts.reset_index()
        df_info.columns = ['Segment', 'Total']
        df_info = df_info.sort_values(by='Total', ascending=False)
        st.dataframe(df_info, use_container_width=True, hide_index=True)
        
        top_seg = df_info.iloc[0]['Segment']
        st.success(f"Segmen terbanyak: **{top_seg}**")

    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank', 'r_rank_norm', 'f_rank_norm', 'm_rank_norm'], inplace=True)

    with st.expander("🔍 Lihat Detail & Definisi Segmen Pelanggan"):
        st.write("""
        Analisis segmentasi ini menggunakan teknik binning pada skor rata-rata RFM (skala 0-100). 
        Berikut adalah detail karakteristik untuk setiap kelompok:
        """)
        
        segment_summary = rfm_df.groupby('customer_segment', observed=False).agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'customer_id': 'count'
        }).rename(columns={'customer_id': 'Total Customers'}).reset_index()
        
        st.dataframe(segment_summary.style.format({
            'recency': '{:.1f} hari',
            'frequency': '{:.2f} order',
            'monetary': 'BRL {:.2f}'
        }), use_container_width=True)

        st.markdown("""
        - **Champions:** Pelanggan terbaik yang baru saja belanja, sering, dan nominal besar.
        - **Loyalist:** Pelanggan cukup setia dengan kontribusi revenue stabil.
        - **At Risk:** Dulu sering belanja tapi sudah lama tidak kembali.
        - **Hibernating:** Jarang belanja dan sudah sangat lama tidak aktif.
        """)
        pass
with tab_eda:
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.subheader("Customer Distribution by State")
        state_df = main_df.groupby("customer_state").customer_unique_id.nunique().sort_values(ascending=False).head(10).reset_index()
        colors_state = ["#003f5c" if i == 0 else "#a2d2ff" for i in range(len(state_df))]

        fig_state, ax_state = plt.subplots(figsize=(10, 6))
        sns.barplot(x="customer_unique_id", y="customer_state", data=state_df, palette=colors_state, ax=ax_state)
        ax_state.set_xlabel("Number of Unique Customers")
        ax_state.set_ylabel("State")
        st.pyplot(fig_state)
        
        st.write("""
        **Insight Geografis:**
        Negara bagian **SP (Sao Paulo)** mendominasi basis pelanggan secara signifikan. 
        Strategi pemasaran atau promo ongkir dapat difokuskan pada wilayah ini untuk efisiensi logistik.
        """)

    with col_eda2:
        st.subheader("Order Status Overview")
        status_df = main_df["order_status"].value_counts().reset_index()
        
        colors_status = ["#003f5c" if i == 0 else "#a2d2ff" for i in range(len(status_df))]
        
        fig_status, ax_status = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            x="count", 
            y="order_status", 
            data=status_df, 
            palette=colors_status,
            ax=ax_status
        )
        
        ax_status.set_xlabel("Number of Orders")
        ax_status.set_ylabel(None)
        st.pyplot(fig_status)
        
        st.write("""
        **Insight Operasional:**
        Mayoritas pesanan berstatus **'delivered'**. Hal ini menunjukkan bahwa proses fulfillment berjalan baik, 
        sehingga skor RFM yang dihasilkan valid karena didasarkan pada pesanan yang benar-benar sampai ke pelanggan.
        """)

    st.divider()
    
    st.subheader("Seasonality & Operational Note")
    st.info("""
    - Masalah operasional dan fluktuasi penjualan bersifat nasional, bukan spesifik wilayah tertentu.
    - Terdapat pola musiman di mana bulan-bulan tertentu mengalami penurunan (seperti Juni-Juli).
    - **Saran Bisnis:** Perusahaan dapat melakukan kampanye agresif pada bulan-bulan 'sepi' untuk menjaga pertumbuhan revenue tetap stabil sepanjang tahun.
    """)

st.caption('Copyright © Fanidyasani Atantya 2026')