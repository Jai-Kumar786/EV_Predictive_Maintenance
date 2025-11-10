# Week 10 - Component 3: Fleet Health Dashboard using Streamlit (Expanded & Hardened)
#
# This script creates an interactive, professional web application to visualize fleet health.
#
# Expanded Features:
# -   **Hardened Data Loading:** Includes a data validation step to convert columns to numeric
#     types, preventing crashes from bad data in the database.
# -   **Summary Metrics:** Displays key performance indicators (KPIs) at the top of the page,
#     such as average fleet health and the number of vehicles requiring attention.
# -   **Improved Layout:** Uses columns for a cleaner, more organized presentation of metrics.
# -   **Informational Context:** Includes an expander to explain how to interpret the dashboard,
#     making it more useful for non-technical stakeholders.
# -   **Enhanced Table:** The ranked scorecard now uses progress bars for key metrics, making
#     it easier to visually compare vehicles.

import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Fleet Predictive Maintenance Dashboard",
    page_icon="🔋",
    layout="wide"
)

# --- Database Connection ---
DB_NAME = 'fleet_health.db'


@st.cache_data(ttl=60)  # Cache data for 60 seconds to improve performance
def get_latest_fleet_data():
    """
    Queries the DB for the most recent health record of each vehicle.
    Includes a data validation and type conversion step to prevent errors.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        query = """
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY vehicle_id ORDER BY timestamp DESC) as rn
                FROM health_records
            ) WHERE rn = 1
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        # --- Data Validation and Hardening ---
        # This is a defensive step. Even if the API saves data incorrectly, this
        # will attempt to convert the key columns to numbers, preventing crashes.
        if not df.empty:
            for col in ['predicted_soh', 'predicted_sop', 'health_score']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['predicted_soh', 'predicted_sop', 'health_score'], inplace=True)
        return df

    except Exception as e:
        st.error(f"Database connection or data processing failed: {e}")
        return pd.DataFrame()


# --- Main Dashboard UI ---
st.title("🔋 EV Fleet Predictive Maintenance Dashboard")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button('Refresh Data'):
    st.cache_data.clear()  # Clear the cache on refresh
    st.rerun()

fleet_df = get_latest_fleet_data()

if fleet_df.empty:
    st.warning("No data available in the fleet health database. Please process a trip file.")
else:
    st.markdown("---")

    # --- 1. Summary Metrics ---
    st.header("Fleet Summary")
    col1, col2, col3 = st.columns(3)

    avg_soh = fleet_df['predicted_soh'].mean()
    vehicles_at_risk = fleet_df[fleet_df['status'] != 'Healthy'].shape[0]
    total_vehicles = fleet_df.shape[0]

    col1.metric("Total Vehicles Monitored", f"{total_vehicles}")
    col2.metric("Fleet Average SoH", f"{avg_soh:.2%}")
    col3.metric("Vehicles Requiring Attention", f"{vehicles_at_risk}", delta=f"{vehicles_at_risk} at risk",
                delta_color="inverse")

    st.markdown("---")

    # --- 2. Fleet Health Risk Quadrant ---
    st.header("Fleet Health Risk Quadrant")

    with st.expander("How to interpret this chart"):
        st.info("""
            This chart provides an at-a-glance view of the entire fleet's health.
            -   **X-Axis (Predicted SoH):** Higher is better. This represents the long-term health and capacity of the battery.
            -   **Y-Axis (Health Score):** Lower is better. This is a unified risk score combining multiple factors.
            -   **Goal:** You want your vehicles in the **bottom-right** quadrant (high SoH, low risk score). Vehicles in the **top-left** are the highest priority for maintenance.
        """)

    median_soh = fleet_df['predicted_soh'].median()
    median_health_score = fleet_df['health_score'].median()

    chart = alt.Chart(fleet_df).mark_circle(size=200).encode(
        x=alt.X('predicted_soh:Q', title='Predicted SoH (Higher is Better)',
                scale=alt.Scale(domain=[fleet_df['predicted_soh'].min() - 0.05, 1.0])),
        y=alt.Y('health_score:Q', title='Health Score (Lower is Better)', scale=alt.Scale(zero=False)),
        color=alt.Color('status:N',
                        scale=alt.Scale(
                            domain=['Healthy', 'Monitor', 'Priority Maintenance'],
                            range=['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
                        ), legend=alt.Legend(title="Status")),
        tooltip=['vehicle_id', 'predicted_soh', 'predicted_sop', 'health_score', 'status']
    ).interactive()

    text = chart.mark_text(align='left', baseline='middle', dx=15).encode(text='vehicle_id:N')
    vline = alt.Chart(pd.DataFrame({'x': [median_soh]})).mark_rule(strokeDash=[5, 5], color='gray').encode(x='x')
    hline = alt.Chart(pd.DataFrame({'y': [median_health_score]})).mark_rule(strokeDash=[5, 5], color='gray').encode(
        y='y')

    st.altair_chart((chart + text + vline + hline), use_container_width=True)

    st.markdown("---")

    # --- 3. Ranked Health Scorecard ---
    st.header("Ranked Fleet Health Scorecard")


    def color_status(val):
        color = 'red' if val == 'Priority Maintenance' else 'orange' if val == 'Monitor' else 'green'
        return f'color: {color}'


    display_df = fleet_df[['vehicle_id', 'predicted_soh', 'predicted_sop', 'health_score', 'status', 'timestamp']]
    display_df = display_df.sort_values(by='health_score', ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df.style
        .applymap(color_status, subset=['status'])
        .format({
            'predicted_soh': '{:.2%}',
            'predicted_sop': '{:,.0f} W',
            'health_score': '{:.3f}'
        })
        .bar(subset=['predicted_soh'], align='left', color='#5fba7d')
        .bar(subset=['health_score'], align='left', color='#d65f5f'),
        use_container_width=True
    )