import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Retail Data Visualization Dashboard", page_icon="📊", layout="wide")
st.title("📊 Retail Sales – Data Visualization Dashboard")
st.caption("Task 3 | Explore sales, profit, categories and monthly performance")

@st.cache_data
def load_data():
    data = pd.read_csv("data/retail_sales.csv", parse_dates=["Order_Date"])
    data["Month"] = data["Order_Date"].dt.to_period("M").dt.to_timestamp()
    return data

df = load_data()
with st.sidebar:
    st.header("Dashboard Filters")
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    categories = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    min_date, max_date = df["Order_Date"].min().date(), df["Order_Date"].max().date()
    date_range = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[filtered["Order_Date"].dt.date.between(date_range[0], date_range[1])]
if filtered.empty:
    st.warning("No records match these filters. Please change your selections.")
    st.stop()

total_sales, total_profit = filtered["Sales"].sum(), filtered["Profit"].sum()
m1,m2,m3,m4 = st.columns(4)
m1.metric("Total Sales", f"${total_sales:,.0f}")
m2.metric("Total Profit", f"${total_profit:,.0f}")
m3.metric("Total Orders", f"{len(filtered):,}")
m4.metric("Average Order Value", f"${filtered['Sales'].mean():,.2f}")

left,right = st.columns(2)
with left:
    cat = filtered.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    st.plotly_chart(px.bar(cat, x="Category", y="Sales", color="Category", title="Sales by Product Category"), use_container_width=True)
with right:
    monthly = filtered.groupby("Month", as_index=False)["Sales"].sum()
    st.plotly_chart(px.line(monthly, x="Month", y="Sales", markers=True, title="Monthly Sales Trend"), use_container_width=True)

left,right = st.columns(2)
with left:
    pie = filtered.groupby("Category", as_index=False)["Sales"].sum()
    st.plotly_chart(px.pie(pie, names="Category", values="Sales", hole=.35, title="Category-wise Sales Share"), use_container_width=True)
with right:
    st.plotly_chart(px.histogram(filtered, x="Sales", nbins=28, marginal="box", title="Distribution of Order Values"), use_container_width=True)

st.plotly_chart(px.scatter(filtered, x="Sales", y="Profit", color="Category",
    hover_data=["Product","Region","Order_Date"], title="Profit vs. Sales"), use_container_width=True)
st.subheader("Filtered Data")
st.dataframe(filtered.drop(columns=["Month"]), use_container_width=True, hide_index=True)
st.download_button("Download filtered data as CSV",
    filtered.drop(columns=["Month"]).to_csv(index=False), "filtered_retail_sales.csv", "text/csv")

st.subheader("Key Insights")
st.markdown(f"- **Top sales category:** {filtered.groupby('Category')['Sales'].sum().idxmax()}")
st.markdown(f"- **Top-performing region:** {filtered.groupby('Region')['Sales'].sum().idxmax()}")
st.markdown(f"- **Profit margin:** {(total_profit / total_sales * 100):.1f}%" if total_sales else "- **Profit margin:** N/A")
