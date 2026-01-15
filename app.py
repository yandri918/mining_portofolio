import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from data_processor import load_and_process_data, get_mining_stats
from content import TRANSLATIONS

# --- Page Configuration ---
st.set_page_config(
    page_title="Mining Data Science Portfolio",
    page_icon="⛏️",
    layout="wide"
)

# --- Sidebar & Language Selection ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2558/2558944.png", width=100) # Placeholder icon
    lang_choice = st.radio("Language / Bahasa", ["EN", "ID"], horizontal=True)
    t = TRANSLATIONS[lang_choice]
    
    st.title(t['sidebar_title'])
    page = st.radio("Go to", [t['home_nav'], t['analysis_nav'], t['forecast_nav'], t['safety_nav']])
    
    st.markdown("---")
    # Contact info removed as requested

# --- Load Data ---
try:
    df = load_and_process_data('data.csv')
    mining_stats = get_mining_stats(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Page: Home & Profile ---
if page == t['home_nav']:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Placeholder for profile picture if user wants to add one later
        # st.image("profile.jpg", caption=t['role']) 
        st.markdown(f"### {t['role']}")
        st.info("📍 Data Scientist | 📍 Mining Focus | 📍 FIFO Ready")
        
    with col2:
        st.title(t['intro_title'])
        st.subheader(t['intro_subtitle'])
        st.markdown("---")
        
        st.markdown(f"#### {t['about_tit']}")
        st.markdown(t['about_text'])
        
        st.markdown(f"#### {t['skills_tit']}")
        cols = st.columns(3)
        for i, skill in enumerate(t['skills_list']):
            cols[i % 3].success(skill)

# --- Page: Analysis ---
elif page == t['analysis_nav']:
    st.title(t['charts_tit'])
    st.markdown(t['charts_desc'])
    
    # Metrics Table
    if mining_stats:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mining GDP (Latest)", f"{mining_stats['current_gdp']:,.1f}")
        m2.metric(t['metric_growth'], f"{mining_stats['total_growth_pct']:.1f}%")
        m3.metric(t['metric_cagr'], f"{mining_stats['cagr_pct']:.1f}%")
    
    st.divider()
    
    # 1. Comparison Chart
    fig_line = px.line(
        df, 
        x='Year', 
        y='GDP', 
        color='Sector',
        markers=True,
        title='GDP Trends by Economic Activity (2007-2014)',
        template="plotly_dark"
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # 2. Mining Specific Focus
    mining_data = df[df['Sector'] == 'Mining']
    fig_bar = px.bar(
        mining_data,
        x='Year',
        y='GDP',
        text_auto=True,
        title='Mining Sector Growth Focus',
        color='GDP',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Page: Forecast ---
elif page == t['forecast_nav']:
    st.title(t['forecast_tit'])
    st.markdown(t['forecast_desc'])
    
    mining_data = df[df['Sector'] == 'Mining'].copy()
    
    # Prepare data for ML
    X = mining_data[['Year']]
    y = mining_data['GDP']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Future predictions (Next 5 years)
    future_years = np.array(range(2015, 2020)).reshape(-1, 1)
    future_pred = model.predict(future_years)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'GDP': future_pred,
        'Type': t['predicted']
    })
    
    history_df = mining_data[['Year', 'GDP']].copy()
    history_df['Type'] = t['actual']
    
    combined_df = pd.concat([history_df, forecast_df])
    
    fig_forecast = px.line(
        combined_df, 
        x='Year', 
        y='GDP', 
        color='Type', 
        markers=True,
        symbol='Type',
        line_dash='Type',
        title='Mining GDP Linear Forecast (2015-2019 Projection)',
        template="plotly_dark"
    )
    
    # Highlight the projection area
    fig_forecast.add_vrect(
        x0=2014.5, x1=2019.5, 
        fillcolor="green", opacity=0.1, 
        layer="below", line_width=0,
        annotation_text="Forecast Zone", annotation_position="top left"
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.info("Note: This is a linear projection for demonstration purposes. Real-world mining forecasting would incorporate commodity prices, production volumes, and global demand indices.")

# --- Page: Safety Demos ---
elif page == t['safety_nav']:
    st.title(t['safety_tit'])
    st.markdown(t['safety_desc'])
    
    # 1. Incident Analysis
    st.subheader(t['sub_incidents'])
    st.markdown(t['desc_incidents'])
    
    # Generate Synthetic Incident Data
    incident_data = pd.DataFrame({
        'Month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Near_Miss': np.random.randint(5, 20, 12),
        'Minor_Incident': np.random.randint(0, 5, 12),
        'LTI': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0] # Lost Time Injuries
    })
    
    fig_inc = go.Figure()
    fig_inc.add_trace(go.Bar(x=incident_data['Month'], y=incident_data['Near_Miss'], name='Near Miss', marker_color='#F1C40F'))
    fig_inc.add_trace(go.Bar(x=incident_data['Month'], y=incident_data['Minor_Incident'], name='Minor Incident', marker_color='#E67E22'))
    fig_inc.add_trace(go.Scatter(x=incident_data['Month'], y=incident_data['LTI'], name='LTI (Lost Time)', mode='lines+markers', line=dict(color='#E74C3C', width=3)))
    
    fig_inc.update_layout(title="Monthly Safety Incidents (Synthetic)", barmode='stack', template='plotly_dark')
    st.plotly_chart(fig_inc, use_container_width=True)
    
    st.divider()
    
    # 2. Leading Indicators: Fatigue
    st.subheader(t['sub_indicators'])
    st.markdown(t['desc_indicators'])
    
    # Synthetic Fatigue Data
    n_workers = 50
    fatigue_data = pd.DataFrame({
        'Shift_Hour': np.random.uniform(0, 12, n_workers),
        'Hours_Slept': np.random.uniform(4, 9, n_workers),
    })
    # Fatigue Score = higher shift hour + lower sleep
    fatigue_data['Fatigue_Risk'] = (fatigue_data['Shift_Hour'] * 0.5) + ((9 - fatigue_data['Hours_Slept']) * 1.5)
    
    fig_fatigue = px.scatter(
        fatigue_data, 
        x='Shift_Hour', 
        y='Fatigue_Risk',
        size='Fatigue_Risk',
        color='Fatigue_Risk',
        color_continuous_scale='Redor',
        labels={'Shift_Hour': 'Hours into Shift', 'Fatigue_Risk': 'Fatigue Risk Score'},
        title="Worker Fatigue Risk Leading Indicator"
    )
    # Add safe threshold line
    fig_fatigue.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="Critical Risk Threshold")
    
    st.plotly_chart(fig_fatigue, use_container_width=True)
    
    st.divider()
    
    # 3. Predictive Maintenance
    st.subheader(t['sub_maint'])
    st.markdown(t['desc_maint'])
    
    # Synthetic Sensor Data
    time_series = pd.date_range(start='2024-01-01', periods=100, freq='H')
    # Normal operation + increasing vibration trend
    base_signal = np.random.normal(50, 2, 100)
    degradation = np.linspace(0, 20, 100) # Gradual wear
    vibration = base_signal + degradation
    
    # Threshold breach at index 80
    vibration[80:] += 10 # Sudden fault development
    
    sensor_df = pd.DataFrame({'Time': time_series, 'Vibration_Hz': vibration})
    
    fig_sensor = px.line(sensor_df, x='Time', y='Vibration_Hz', title="Haul Truck Engine Vibration Sensor")
    fig_sensor.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig_sensor.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Critical Failure Threshold")
    
    st.plotly_chart(fig_sensor, use_container_width=True)

st.markdown("---")
st.caption(t['footer'])
