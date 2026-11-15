from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="white")
BLUE, LIGHT_BLUE, GREY, RED = "#1f6fb2", "#9bbfdc", "#c4c9cf", "#d9534f"


def clean(ax):
    ax.grid(False)
    sns.despine(ax=ax)
st.set_page_config(page_title="Dashboard E-Commerce", page_icon="🛒", layout="wide")

DATA_PATH = Path(__file__).parent / "main_data.csv"
GEO_PATH = Path(__file__).parent / "state_geo.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum"),
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    if len(rfm) < 8:
        rfm["segment"] = "Data terlalu sedikit"
        return rfm
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    def segment(row):
        if row["frequency"] >= 2:
            return "Pelanggan Setia (repeat)"
        if row["m_score"] >= 3:
            return "Baru Bernilai Tinggi" if row["r_score"] >= 3 else "Bernilai Tinggi - Berisiko Hilang"
        return "Baru Bernilai Rendah" if row["r_score"] >= 3 else "Tidak Aktif Bernilai Rendah"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


df = load_data()

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("Filter Data")
min_date, max_date = df["purchase_date"].min(), df["purchase_date"].max()
date_range = st.sidebar.date_input(
    "Rentang tanggal pembelian", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
all_states = sorted(df["customer_state"].unique())
selected_states = st.sidebar.multiselect("State pelanggan", all_states, default=all_states)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0] if isinstance(date_range, tuple) else date_range

filtered = df[
    (df["purchase_date"] >= start_date)
    & (df["purchase_date"] <= end_date)
    & (df["customer_state"].isin(selected_states))
]

st.title("🛒 Dashboard Analisis E-Commerce")
st.caption("Brazilian E-Commerce Public Dataset | pesanan berstatus delivered, Jan 2017 - Agu 2018")

if filtered.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah rentang tanggal atau state.")
    st.stop()

orders = filtered.drop_duplicates("order_id")

# ------------------------------------------------------------------ KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"{filtered['price'].sum():,.0f} BRL")
c2.metric("Jumlah Pesanan", f"{orders['order_id'].nunique():,}")
c3.metric("Rata-rata Review Score", f"{orders['review_score'].mean():.2f}")
c4.metric("Pesanan Terlambat", f"{orders['is_late'].mean() * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(
    ["Revenue Kategori & Tren", "Keterlambatan & Review", "Segmentasi RFM"]
)

# ------------------------------------------------------------------ Tab 1
with tab1:
    st.subheader("Kategori produk dengan revenue tertinggi dan terendah")
    n = st.slider("Jumlah kategori yang ditampilkan", 3, 15, 5)

    cat = (
        filtered[filtered["product_category"] != "unknown"]
        .groupby("product_category")
        .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    if cat.empty:
        st.info("Tidak ada data kategori pada filter ini.")
    else:
        cat["share_pct"] = cat["revenue"] / cat["revenue"].sum() * 100
        left, right = st.columns(2)
        with left:
            top = cat.head(n).reset_index()
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(top["product_category"], top["revenue"] / 1e6, color=[BLUE] + [LIGHT_BLUE] * (len(top) - 1))
            ax.invert_yaxis()
            for i, v in enumerate(top["revenue"]):
                ax.text(v / 1e6, i, f" {v/1e6:.2f} jt", va="center")
            ax.set_xlim(0, top["revenue"].max() / 1e6 * 1.2)
            ax.set_title(f"{len(top)} Kategori Revenue Tertinggi", loc="left")
            ax.set_xlabel("Revenue (juta BRL)"); ax.set_ylabel("")
            clean(ax)
            st.pyplot(fig)
            st.caption(f"Kontribusi {len(top)} kategori teratas: {top['share_pct'].sum():.1f}% dari total revenue")
        with right:
            bottom = cat.tail(n).reset_index().sort_values("revenue", ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(bottom["product_category"], bottom["revenue"], color=[GREY] * (len(bottom) - 1) + [RED])
            ax.invert_yaxis()
            for i, v in enumerate(bottom["revenue"]):
                ax.text(v, i, f" {v:,.0f}", va="center")
            ax.set_xlim(0, bottom["revenue"].max() * 1.2)
            ax.set_title(f"{len(bottom)} Kategori Revenue Terendah", loc="left")
            ax.set_xlabel("Revenue (BRL)"); ax.set_ylabel("")
            clean(ax)
            st.pyplot(fig)
            st.caption("Perhatian: skala sumbu kedua grafik berbeda.")

    st.subheader("Tren revenue bulanan")
    monthly = (
        filtered.groupby(filtered["order_purchase_timestamp"].dt.to_period("M"))["price"].sum().reset_index()
    )
    monthly["bulan"] = monthly["order_purchase_timestamp"].astype(str)
    monthly["ribu"] = monthly["price"] / 1000
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly["bulan"], monthly["ribu"], marker="o", linewidth=2, color=LIGHT_BLUE)
    peak = monthly.loc[monthly["ribu"].idxmax()]
    ax.plot(peak["bulan"], peak["ribu"], "o", color=RED, markersize=10)
    ax.annotate(f"Puncak {peak['bulan']}: {peak['ribu']:,.0f} ribu BRL", xy=(peak["bulan"], peak["ribu"]),
                xytext=(peak["bulan"], peak["ribu"] * 1.08), ha="center", color=RED, fontweight="bold")
    ax.set_ylim(0, monthly["ribu"].max() * 1.2)
    ax.set_xlabel("Bulan pembelian"); ax.set_ylabel("Revenue (ribu BRL)")
    plt.xticks(rotation=45)
    clean(ax)
    st.pyplot(fig)

# ------------------------------------------------------------------ Tab 2
with tab2:
    st.subheader("Keterlambatan pengiriman dan review score per state")
    top_n = st.slider("Tampilkan N state dengan pesanan terbanyak", 5, 15, 10)

    state_all = (
        orders.groupby("customer_state")
        .agg(orders=("order_id", "count"),
             late_pct=("is_late", lambda s: s.mean() * 100),
             avg_review=("review_score", "mean"))
    )
    state = (
        state_all.sort_values("orders", ascending=False)
        .head(top_n)
        .sort_values("late_pct", ascending=False)
        .reset_index()
    )
    bar_colors = [RED] * min(2, len(state)) + [GREY] * max(len(state) - 2, 0)
    overall_late = orders["is_late"].mean() * 100
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(state["customer_state"], state["late_pct"], color=bar_colors)
        for i, v in enumerate(state["late_pct"]):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(overall_late, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, max(state["late_pct"].max(), overall_late) * 1.2)
        ax.set_title(f"% Pesanan Terlambat (rata-rata {overall_late:.1f}%)", loc="left")
        ax.set_xlabel("State"); ax.set_ylabel("Pesanan terlambat (%)")
        clean(ax)
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(state["customer_state"], state["avg_review"], color=bar_colors)
        for i, v in enumerate(state["avg_review"]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 5)
        ax.set_title("Rata-rata Review Score (skala 1-5)", loc="left")
        ax.set_xlabel("State (urutan sama dengan grafik kiri)"); ax.set_ylabel("Review score")
        clean(ax)
        st.pyplot(fig)

    st.subheader("Review score: tepat waktu vs terlambat")
    labeled = orders.assign(status=orders["is_late"].map({False: "Tepat waktu", True: "Terlambat"}))
    timing = labeled.groupby("status").agg(
        avg_review=("review_score", "mean"),
        bad_pct=("review_score", lambda s: (s <= 2).sum() / max(s.notna().sum(), 1) * 100),
    )
    timing = timing.reindex([x for x in ["Tepat waktu", "Terlambat"] if x in timing.index])
    colors = [LIGHT_BLUE, RED][: len(timing)] if len(timing) == 2 else [LIGHT_BLUE]
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(timing.index, timing["avg_review"], color=colors)
        for i, v in enumerate(timing["avg_review"]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, 5); ax.set_ylabel("Rata-rata review score (skala 1-5)")
        ax.set_title("Rata-rata Review Score", loc="left")
        clean(ax)
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(timing.index, timing["bad_pct"], color=colors)
        for i, v in enumerate(timing["bad_pct"]):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
        ax.set_ylim(0, 100); ax.set_ylabel("Pesanan dengan review skor 1-2 (%)")
        ax.set_title("Persentase Review Buruk (skor 1-2)", loc="left")
        clean(ax)
        st.pyplot(fig)

    st.subheader("Peta sebaran pesanan dan keterlambatan per state")
    st.caption("Ukuran gelembung = jumlah pesanan, warna = persentase pesanan terlambat (koordinat = titik tengah state).")
    geo = pd.read_csv(GEO_PATH)[["customer_state", "lat", "lng"]].merge(state_all.reset_index(), on="customer_state")
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(geo["lng"], geo["lat"], s=geo["orders"] / geo["orders"].max() * 2000 + 40,
                    c=geo["late_pct"], cmap="Reds", edgecolor="grey", alpha=0.85)
    for _, r in geo.iterrows():
        ax.text(r["lng"], r["lat"], r["customer_state"], ha="center", va="center", fontsize=8)
    plt.colorbar(sc, ax=ax, shrink=0.7).set_label("Pesanan terlambat (%)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    clean(ax)
    st.pyplot(fig)

    st.subheader("Review score berdasarkan lama pengiriman")
    bins = pd.cut(orders["delivery_days"], [-1, 7, 14, 21, 1000], labels=["<= 7 hari", "8-14 hari", "15-21 hari", "> 21 hari"])
    b = orders.groupby(bins, observed=True).agg(n=("order_id", "count"), avg=("review_score", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(b.index.astype(str), b["avg"], color=[LIGHT_BLUE] * (len(b) - 1) + [RED] if len(b) else [])
    for i, (v, n_) in enumerate(zip(b["avg"], b["n"])):
        ax.text(i, v, f"{v:.2f}\n({n_:,} pesanan)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 5.5); ax.set_xlabel("Lama pengiriman"); ax.set_ylabel("Rata-rata review score (skala 1-5)")
    clean(ax)
    st.pyplot(fig)

# ------------------------------------------------------------------ Tab 3
with tab3:
    st.subheader("Segmentasi pelanggan (RFM)")
    rfm = build_rfm(filtered)
    repeat_pct = (rfm["frequency"] > 1).mean() * 100
    st.metric("Pelanggan yang membeli lebih dari sekali", f"{repeat_pct:.2f}%")

    seg = (
        rfm.groupby("segment")
        .agg(customers=("customer_unique_id", "count"),
             avg_recency=("recency", "mean"),
             avg_frequency=("frequency", "mean"),
             avg_monetary=("monetary", "mean"))
        .sort_values("customers", ascending=False)
    )
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        s = seg.sort_values("customers")
        ax.barh(s.index, s["customers"], color=[RED if "Berisiko" in x else LIGHT_BLUE for x in s.index])
        ax.set_title("Jumlah Pelanggan per Segmen", loc="left")
        ax.set_xlabel("Jumlah pelanggan unik")
        clean(ax)
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        s = seg.sort_values("avg_monetary")
        ax.barh(s.index, s["avg_monetary"], color=[RED if "Berisiko" in x else LIGHT_BLUE for x in s.index])
        ax.set_title("Rata-rata Belanja per Segmen", loc="left")
        ax.set_xlabel("Rata-rata monetary (BRL)")
        clean(ax)
        st.pyplot(fig)
    st.dataframe(seg.round(2))

st.caption("Proyek Analisis Data - E-Commerce Public Dataset")
