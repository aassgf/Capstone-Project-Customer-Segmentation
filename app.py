# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
from io import StringIO, BytesIO

st.set_page_config(layout="wide", page_title="RFM Clusters Dashboard")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error membaca CSV: {e}")
        return pd.DataFrame()

def clean_rfm_df(df):
    df = df.copy()
    # Pastikan kolom penting ada
    for c in ['Cluster', 'MonetaryValue', 'Frequency', 'Recency']:
        if c not in df.columns:
            df[c] = np.nan
    # convert numeric
    df['Cluster'] = pd.to_numeric(df['Cluster'], errors='coerce')
    df['MonetaryValue'] = pd.to_numeric(df['MonetaryValue'], errors='coerce')
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    df['Recency'] = pd.to_numeric(df['Recency'], errors='coerce')
    df = df.dropna(subset=['Cluster', 'MonetaryValue', 'Frequency', 'Recency'])
    # convert cluster to int if whole numbers
    if pd.api.types.is_float_dtype(df['Cluster']):
        if (df['Cluster'] % 1 == 0).all():
            df['Cluster'] = df['Cluster'].astype(int)
    return df

def build_palette(cluster_vals, base_colors=None):
    """Return dict mapping cluster->hex color. base_colors can override."""
    if base_colors is None:
        base_colors = {}
    unique = sorted(cluster_vals)
    palette = {}
    # use base when exists
    for c in unique:
        if c in base_colors:
            palette[c] = base_colors[c]
    missing = [c for c in unique if c not in palette]
    if missing:
        # generate auto colors
        auto = sns.color_palette(n_colors=len(missing))
        for c, col in zip(missing, auto):
            palette[c] = mcolors.to_hex(col)
    # ensure deterministic order
    palette = {k: palette[k] for k in sorted(palette.keys())}
    return palette

def download_df_as_csv(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------------------------
# UI: Sidebar
# ---------------------------
st.sidebar.title("Pengaturan Data")
uploaded = st.sidebar.file_uploader("Upload CSV RFM (opsional)", type=['csv'])
use_example = st.sidebar.checkbox("Gunakan contoh dummy (jika tidak upload)", value=False)

# base palette sesuai yg kamu inginkan
base_cluster_colors = {
    0: '#1f77b4',  # Blue - Retain
    1: '#ff7f0e',  # Orange - Re-Engage
    2: '#2ca02c',  # Green - Nurture
    3: '#d62728'   # Red - Reward
}

# ---------------------------
# Data loading
# ---------------------------
if uploaded is not None:
    df_raw = load_csv(uploaded)
elif use_example:
    # create small example
    rng = np.random.default_rng(42)
    n = 300
    df_raw = pd.DataFrame({
        'Cluster': rng.integers(0, 4, size=n),
        'MonetaryValue': np.abs(rng.normal(200, 120, size=n)).round(2),
        'Frequency': np.abs(rng.normal(5, 3, size=n)).round(0),
        'Recency': np.abs(rng.normal(60, 45, size=n)).round(0)
    })
else:
    st.info("Upload CSV yang berisi kolom: Cluster, MonetaryValue, Frequency, Recency — atau centang 'Gunakan contoh dummy'.")
    df_raw = pd.DataFrame()  # empty

if df_raw.empty:
    st.stop()

df = clean_rfm_df(df_raw)

# Sidebar filters
st.sidebar.markdown("### Filter")
all_clusters = sorted(df['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Pilih cluster", options=all_clusters, default=all_clusters)
mv_min, mv_max = float(df['MonetaryValue'].min()), float(df['MonetaryValue'].max())
freq_min, freq_max = float(df['Frequency'].min()), float(df['Frequency'].max())
rec_min, rec_max = float(df['Recency'].min()), float(df['Recency'].max())

mv_range = st.sidebar.slider("MonetaryValue range", mv_min, mv_max, (mv_min, mv_max))
freq_range = st.sidebar.slider("Frequency range", freq_min, freq_max, (freq_min, freq_max))
rec_range = st.sidebar.slider("Recency range (days)", rec_min, rec_max, (rec_min, rec_max))

# Apply filters
df_filtered = df[
    (df['Cluster'].isin(selected_clusters)) &
    (df['MonetaryValue'] >= mv_range[0]) & (df['MonetaryValue'] <= mv_range[1]) &
    (df['Frequency'] >= freq_range[0]) & (df['Frequency'] <= freq_range[1]) &
    (df['Recency'] >= rec_range[0]) & (df['Recency'] <= rec_range[1])
].copy()

st.sidebar.markdown("---")
if st.sidebar.button("Reset filter"):
    st.experimental_rerun()

# ---------------------------
# Build palette for present clusters
# ---------------------------
palette = build_palette(sorted(df_filtered['Cluster'].unique()), base_colors=base_cluster_colors)

# ---------------------------
# Main layout
# ---------------------------
st.title("Dashboard Segmentasi Pelanggan (RFM + Clustering)")
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Ringkasan Data")
    st.write(f"Total baris (sebelum filter): {len(df)}")
    st.write(f"Total baris (setelah filter): {len(df_filtered)}")
    st.dataframe(df_filtered.head(10))

with c2:
    st.subheader("Deskripsi Cluster")
    st.markdown("""
    **Cluster 0 (Biru): Retain**  
    Fokus: menjaga pelanggan bernilai tinggi agar tetap aktif.  
    **Cluster 1 (Oranye): Re-Engage**  
    Fokus: mengaktifkan kembali pelanggan kurang aktif.  
    **Cluster 2 (Hijau): Nurture**  
    Fokus: mengembangkan pelanggan baru/berpotensi.  
    **Cluster 3 (Merah): Reward**  
    Fokus: beri reward pada pelanggan paling loyal.
    """)
    buf = download_df_as_csv(df_filtered)
    st.download_button("Unduh data filter (CSV)", data=buf, file_name="rfm_filtered.csv", mime="text/csv")

# ---------------------------
# Violin plots (matplotlib + seaborn) — R/F/M by cluster
# ---------------------------
st.markdown("## Visualisasi Distribusi RFM per Cluster")
fig = plt.figure(figsize=(12, 16))

order = sorted(df_filtered['Cluster'].unique())

# Monetary
ax1 = plt.subplot(3, 1, 1)
sns.violinplot(x='Cluster', y='MonetaryValue', data=df_filtered, hue='Cluster',
               palette=palette, legend=False, order=order)
ax1.set_title('Monetary Value Distribution by Cluster')
ax1.set_xlabel('')
ax1.set_ylabel('Monetary Value')

# Frequency
ax2 = plt.subplot(3, 1, 2)
sns.violinplot(x='Cluster', y='Frequency', data=df_filtered, hue='Cluster',
               palette=palette, legend=False, order=order)
ax2.set_title('Frequency Distribution by Cluster')
ax2.set_xlabel('')
ax2.set_ylabel('Frequency')

# Recency
ax3 = plt.subplot(3, 1, 3)
sns.violinplot(x='Cluster', y='Recency', data=df_filtered, hue='Cluster',
               palette=palette, legend=False, order=order)
ax3.set_title('Recency Distribution by Cluster')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Recency (days)')

plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# 3D interactive scatter (Plotly)
# ---------------------------
st.markdown("## Scatter 3D Interaktif (Monetary, Frequency, Recency)")
if len(df_filtered) > 0:
    # map colors for plotly
    color_map = {str(k): v for k, v in palette.items()}
    df_plot = df_filtered.copy()
    df_plot['Cluster_str'] = df_plot['Cluster'].astype(str)
    fig3d = px.scatter_3d(
        df_plot,
        x='MonetaryValue', y='Frequency', z='Recency',
        color='Cluster_str',
        color_discrete_map=color_map,
        hover_data=['MonetaryValue', 'Frequency', 'Recency', 'Cluster'],
        title='3D Customer Segmentation by RFM Clusters',
        width=900, height=600
    )
    fig3d.update_traces(marker=dict(size=4, line=dict(width=0.5, color='white')))
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("Tidak ada data untuk ditampilkan (setelah filter).")

# ---------------------------
# Cluster summary cards
# ---------------------------
st.markdown("## Insight & Rekomendasi per Cluster")
cols = st.columns(len(order) if len(order)>0 else 1)
cluster_texts = {
    0: ("Retain", "Pelanggan bernilai tinggi. Fokus mempertahankan melalui loyalitas & personalisasi."),
    1: ("Re-Engage", "Pelanggan kurang aktif. Kirim kampanye & diskon untuk mengaktifkan kembali."),
    2: ("Nurture", "Pelanggan baru/berpotensi. Bangun hubungan & tawarkan insentif."),
    3: ("Reward", "Pelanggan paling loyal. Beri reward & akses eksklusif.")
}
for i, c in enumerate(order):
    with cols[i]:
        title, desc = cluster_texts.get(int(c), (f"Cluster {c}", "Deskripsi tidak tersedia"))
        st.metric(label=f"Cluster {int(c)} — {title}", value=int((df['Cluster']==c).sum()))
        st.write(desc)

st.markdown("---")
st.write("Aplikasi ini bisa disesuaikan lagi: tambah chart KPI (LTV, churn rate), segmentasi lebih lanjut, atau koneksi DB.")

print("Hello Streamlit!")
