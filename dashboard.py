import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="CampusEats AI Dashboard", layout="wide")
st.title("🍽️ CampusEats AI Dashboard")
st.markdown("### Data-driven insights into student food patterns and demand forecasting.")

# -------------------------------
# Safe JSON Loader Function
# -------------------------------
def safe_load_json(filename, label):
    path = Path(filename)
    if not path.exists():
        st.warning(f"⚠️ {label} file not found: {filename}")
        return pd.DataFrame()
    return pd.read_json(path)

# -------------------------------
# Load Data
# -------------------------------
sentiments = safe_load_json("frontend_sentiments.json", "Sentiments")
predicted = safe_load_json("frontend_predicted_demand.json", "Predicted Demand")
top_items = safe_load_json("frontend_top_items.json", "Top Items")



# -------------------------------
# Tabs Layout
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Overview",
    "💬 Sentiment Insights",
    "📈 Demand Trends",
    "📚 Analysis",
    "ℹ️ About / Credits"
])

# -------------------------------------
# TAB 1: Dashboard Overview
# -------------------------------------
with tab1:
    st.header("📅 Top Predicted Items by Day")
    if not top_items.empty:
        fig1 = px.bar(
            top_items,
            x="item_name",
            y="predicted_quantity",
            color="day_of_week",
            barmode="group",
            title="Predicted Top Food Items by Day of the Week"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("⚠️ Top items data not available.")


# -------------------------------------
# TAB 2: Sentiment Insights
# -------------------------------------
with tab2:
    st.header("💬 Food Sentiment Insights")
    if not sentiments.empty:
        selected_item = st.selectbox("Select a Food Item:", sorted(sentiments["item_name"].unique()))
        item_data = sentiments[sentiments["item_name"] == selected_item].iloc[0]
        fig3 = px.pie(
            values=[item_data["Positive"], item_data["Negative"], item_data["Neutral"]],
            names=["Positive", "Negative", "Neutral"],
            title=f"Sentiment Distribution for {selected_item}"
        )
        st.plotly_chart(fig3, use_container_width=False)
        st.info(f"Final Sentiment: **{item_data['final_sentiment']}**")
    else:
        st.warning("⚠️ Sentiment data not found. Please run analysis.py first.")

# -------------------------------------
# TAB 3: Demand Trends
# -------------------------------------
with tab3:
    st.header("📈 Predicted Demand Trends")
    if not predicted.empty:
        selected_day = st.selectbox("Select Day:", sorted(predicted["day_of_week"].unique()))
        selected_time = st.selectbox("Select Time of Day:", sorted(predicted["time_of_day"].unique()))

        filtered = predicted[
            (predicted["day_of_week"] == selected_day)
            & (predicted["time_of_day"] == selected_time)
        ]
        fig4 = px.bar(
            filtered.sort_values(by="predicted_quantity", ascending=False).head(10),
            x="item_name",
            y="predicted_quantity",
            title=f"Predicted Demand on {selected_day} ({selected_time})"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("⚠️ Predicted demand data missing.")

# -------------------------------------
# TAB 4: Analysis (Static PNG Graphs)
# -------------------------------------
with tab4:
    st.header("📚 Analysis & Visual Reports")
    st.markdown("Visual insights derived from model performance and data exploration.")

    # Define each graph with its file path and caption
    analysis_graphs = [
        {"path": "pattern_chart.png", "caption": "Overall consumption trends across weekdays."},
        {"path": "sentiment_chart.png", "caption": "Sentiment insights across weekdays."},
        {"path": "demand_prediction_rf.png", "caption": ""},
        {"path": "top_items_by_day_subplots.png", "caption": "Subplot for Top items by Day."}
    ]

    # Display each image individually (no looping filenames)
    for g in analysis_graphs:
        path = Path(g["path"])
        if path.exists():
            st.image(str(path), caption=g["caption"], use_container_width=True)
        else:
            st.warning(f"⚠️ File not found: {g['path']}")


# -------------------------------------
# TAB 5: About / Credits
# -------------------------------------
with tab5:
    st.header("ℹ️ About CampusEats-AI")
    st.markdown("""
    CampusEats-AI is an intelligent analytics tool that leverages machine learning and NLP 
    to understand student food consumption patterns, predict meal demand, and optimize canteen operations.

    **Key Features:**
    - Day-wise and category-wise demand forecasting  
    - Sentiment analysis from student reviews  
    - Personalized recommendations  
    - Sustainable food management insights
    """)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: grey; font-size: 14px; margin-top: 20px;'>
        <b>~ by Sachin Chaudhary</b><br>
        <span style='font-size: 12px;'>Software Engineering Student, Delhi Technological University</span>
    </div>
    """, unsafe_allow_html=True)

