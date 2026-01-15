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
    page = st.radio("Go to", [t['home_nav'], t['analysis_nav'], t['forecast_nav'], t['safety_nav'], t['prod_nav'], t['pbi_nav']])
    
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
    
    # 1. Incident Analysis (TRIFR)
    st.subheader(t['sub_incidents'])
    st.markdown(t['desc_incidents'])
    
    # Generate Synthetic Data with TRIFR Logic
    months = pd.date_range(start='2023-01-01', periods=12, freq='M')
    man_hours = np.random.normal(50000, 2000, 12) # ~50k hours/month
    lti_count = np.array([0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0]) # Lost Time Injuries
    
    # Calculate TRIFR: (LTI * 1,000,000) / Man Hours (Simplified to just LTI for demo)
    # Real world includes MTI + RWI usually
    trifr = (lti_count * 1_000_000) / man_hours
    
    incident_df = pd.DataFrame({
        'Month': months,
        'LTI': lti_count,
        'Man_Hours': man_hours,
        'TRIFR': trifr
    })
    
    fig_inc = go.Figure()
    # Bar for Raw Counts
    fig_inc.add_trace(go.Bar(
        x=incident_df['Month'], 
        y=incident_df['LTI'], 
        name='LTI Count', 
        marker_color='#E74C3C'
    ))
    
    # Line for Rate (TRIFR) on Secondary Axis
    fig_inc.add_trace(go.Scatter(
        x=incident_df['Month'], 
        y=incident_df['TRIFR'], 
        name='TRIFR', 
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#F1C40F', width=3)
    ))
    
    fig_inc.update_layout(
        title="Safety Performance: LTI vs TRIFR (12 Months)",
        xaxis_title="Month",
        yaxis=dict(title="LTI Count"),
        yaxis2=dict(
            title="TRIFR (per 1M hours)",
            overlaying='y',
            side='right',
            range=[0, 50]
        ),
        template='plotly_dark',
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_inc, use_container_width=True)
    
    st.divider()
    
    # 2. Leading Indicators: Fatigue (PVT Proxy)
    st.subheader(t['sub_indicators'])
    st.markdown(t['desc_indicators'])
    
    # Synthetic PVT Data
    n_tests = 100
    pvt_df = pd.DataFrame({
        'Shift_Hour': np.random.uniform(0, 12, n_tests),
        'Operator_ID': np.random.randint(100, 120, n_tests)
    })
    
    # Reaction time degrades (increases) as shift hour increases
    # Base reaction 250ms + random noise + fatigue factor
    pvt_df['Reaction_Time_ms'] = 250 + (pvt_df['Shift_Hour'] ** 2) * 1.5 + np.random.normal(0, 30, n_tests)
    
    # Risk Classification
    def get_risk(rt):
        if rt < 350: return 'Low'
        elif rt < 500: return 'Moderate'
        else: return 'Critical'
        
    pvt_df['Fatigue_Risk'] = pvt_df['Reaction_Time_ms'].apply(get_risk)
    
    fig_fatigue = px.scatter(
        pvt_df, 
        x='Shift_Hour', 
        y='Reaction_Time_ms',
        color='Fatigue_Risk',
        color_discrete_map={'Low': '#2ECC71', 'Moderate': '#F39C12', 'Critical': '#C0392B'},
        size='Reaction_Time_ms',
        title="Operator Fatigue Analysis: Reaction Time vs Shift Duration",
        labels={'Reaction_Time_ms': 'PVT Reaction Time (ms)', 'Shift_Hour': 'Hours into Shift'}
    )
    
    fig_fatigue.add_hline(y=500, line_dash="dash", line_color="red", annotation_text="Intervention Threshold")
    st.plotly_chart(fig_fatigue, use_container_width=True)
    
    st.divider()
    
    # 3. Condition Monitoring (Multivariate)
    st.subheader(t['sub_maint'])
    st.markdown(t['desc_maint'])
    
    # Generate Dual Sensor Data
    t_steps = 100
    time_points = pd.date_range(start='2024-06-01', periods=t_steps, freq='H')
    
    # Vibration: Stable then spikes
    vib = np.random.normal(4.5, 0.2, t_steps)           # Baseline 4.5 mm/s
    vib[70:] += np.linspace(0, 4.0, 30)                 # Gradual failure onset
    
    # Temperature: Leads vibration spike slightly (e.g. friction heat)
    temp = np.random.normal(85, 2, t_steps)             # Baseline 85°C
    temp[65:] += np.linspace(0, 25, 35)                 # Heat starts rising earlier
    
    cm_df = pd.DataFrame({'Time': time_points, 'Vibration': vib, 'Oil_Temp': temp})
    
    fig_cm = go.Figure()
    
    # Trace 1: Vibration
    fig_cm.add_trace(go.Scatter(
        x=cm_df['Time'], y=cm_df['Vibration'],
        name='Vibration (mm/s)',
        line=dict(color='#3498DB')
    ))
    
    # Trace 2: Temperature (Secondary Axis)
    fig_cm.add_trace(go.Scatter(
        x=cm_df['Time'], y=cm_df['Oil_Temp'],
        name='Oil Temp (°C)',
        yaxis='y2',
        line=dict(color='#E67E22', dash='dot')
    ))
    
    fig_cm.update_layout(
        title="Final Drive Diagnostics: Temp vs Vibration Correlation",
        yaxis=dict(title="Vibration (mm/s)"),
        yaxis2=dict(
            title="Oil Temperature (°C)",
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )
    
    fig_cm.add_vrect(x0=time_points[65], x1=time_points[75], fillcolor="yellow", opacity=0.2, annotation_text="Thermal Onset", annotation_position="top left")
    
    st.plotly_chart(fig_cm, use_container_width=True)

# --- Page: Production Optimization ---
elif page == t['prod_nav']:
    st.title(t['prod_tit'])
    st.markdown(t['prod_desc'])
    
    # 1. Processing Recovery Analysis
    st.subheader(t['sub_yield'])
    st.markdown(t['desc_yield'])
    
    # Synthetic Plant Data
    n_batches = 100
    plant_df = pd.DataFrame({
        'Feed_Grade': np.random.uniform(1.5, 4.5, n_batches), # g/t
        'Reagent_Dosage': np.random.uniform(200, 500, n_batches) # g/t
    })
    # Recovery is complex: Higher grade helps, optimum reagent helps
    plant_df['Recovery'] = 85 + (plant_df['Feed_Grade'] * 2) - ((plant_df['Reagent_Dosage'] - 350)**2 / 5000)
    plant_df['Recovery'] = plant_df['Recovery'].clip(80, 98) + np.random.normal(0, 0.5, n_batches)
    
    fig_rec = px.scatter(
        plant_df, x='Feed_Grade', y='Recovery', color='Reagent_Dosage',
        size='Reagent_Dosage',
        labels={'Feed_Grade': 'Feed Grade (g/t)', 'Recovery': 'Recovery (%)', 'Reagent_Dosage': 'Cyanide Dosage (ppm)'},
        title="Plant Recovery Optimization: Grade vs Reagent Impact",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_rec, use_container_width=True)
    
    st.divider()
    
    # 2. Drill & Blast: Fragmentation
    st.subheader(t['sub_drill'])
    st.markdown(t['desc_drill'])
    
    # Synthetic Blast Data
    n_blasts = 50
    blast_df = pd.DataFrame({
        'Blast_ID': range(1, 51),
        'Powder_Factor': np.random.uniform(0.6, 1.2, n_blasts), # kg/m3
    })
    # P80 Size (cm): Higher powder factor -> Smaller fragments
    blast_df['P80_Size_cm'] = 80 - (blast_df['Powder_Factor'] * 40) + np.random.normal(0, 5, n_blasts)
    # Dig Rate (t/h): Smaller fragments (lower P80) -> Faster digging
    blast_df['Dig_Rate'] = 2000 - (blast_df['P80_Size_cm'] * 15) + np.random.normal(0, 50, n_blasts)
    
    fig_blast = px.scatter(
        blast_df, x='P80_Size_cm', y='Dig_Rate', 
        size='Powder_Factor', color='Powder_Factor',
        labels={'P80_Size_cm': 'Fragmentation P80 (cm)', 'Dig_Rate': 'Excavator Dig Rate (t/h)'},
        title="Drill & Blast: Impact of Fragmentation on Dig Rates",
        trendline="ols"
    )
    st.plotly_chart(fig_blast, use_container_width=True)
    
    st.divider()
    
    # 3. Fleet Management
    st.subheader(t['sub_fleet'])
    st.markdown(t['desc_fleet'])
    
    # Synthetic Haul Cycle Data
    n_cycles = 200
    fleet_df = pd.DataFrame({
        'Truck_ID': np.random.choice(['DT101', 'DT102', 'DT103', 'DT104'], n_cycles),
        'Route': np.random.choice(['Pit A -> Crusher', 'Pit A -> Waste Dump', 'Pit B -> ROM'], n_cycles),
        'Shift': np.random.choice(['Day', 'Night'], n_cycles)
    })
    
    # Generate Cycle Times containing outliers
    fleet_df['Cycle_Time_min'] = np.random.normal(25, 3, n_cycles)
    # Add delays to specific truck (Maintenance issue?)
    fleet_df.loc[fleet_df['Truck_ID'] == 'DT104', 'Cycle_Time_min'] += 8 
    
    fig_fleet = px.box(
        fleet_df, x='Truck_ID', y='Cycle_Time_min', color='Shift',
        title="Fleet Cycle Time Distribution: Identifying Slow Trucks",
        labels={'Cycle_Time_min': 'Cycle Time (minutes)', 'Truck_ID': 'Haul Truck ID'}
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

# --- Page: Power BI + Python ---
elif page == t['pbi_nav']:
    st.title(t['pbi_tit'])
    st.markdown(t['pbi_desc'])
    
    st.subheader(t['pbi_sub'])
    st.info(t['pbi_exp'])
    
    # Synthetic Assay Data
    n_samples = 300
    assay_df = pd.DataFrame({
        'Au_gpt': np.concatenate([np.random.normal(0.5, 0.1, 100), np.random.normal(2.5, 0.5, 100), np.random.normal(5.0, 1.0, 100)]),
        'Cu_pct': np.concatenate([np.random.normal(0.1, 0.05, 100), np.random.normal(0.8, 0.2, 100), np.random.normal(0.2, 0.1, 100)]),
        'As_ppm': np.concatenate([np.random.normal(50, 10, 100), np.random.normal(200, 50, 100), np.random.normal(500, 100, 100)])
    })
    
    # K-Means Clustering (What we would do in Power BI)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    assay_df['Cluster'] = kmeans.fit_predict(assay_df[['Au_gpt', 'Cu_pct', 'As_ppm']])
    assay_df['Cluster_Label'] = assay_df['Cluster'].map({0: 'Waste/Low Grade', 1: 'Oxide Ore', 2: 'Refractory/High As'})
    
    # 3D Visualization (Simulating Power BI Python Visual)
    fig_3d = px.scatter_3d(
        assay_df, x='Au_gpt', y='Cu_pct', z='As_ppm',
        color='Cluster_Label',
        title="3D Geological Domain Clustering (Python in Power BI)",
        labels={'Au_gpt': 'Gold (g/t)', 'Cu_pct': 'Copper (%)', 'As_ppm': 'Arsenic (ppm)'},
        color_discrete_map={'Waste/Low Grade': 'gray', 'Oxide Ore': 'green', 'Refractory/High As': 'red'}
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.divider()
    
    st.markdown(f"#### {t['pbi_code_tit']}")
    st.code("""
# Power BI Python Script for Clustering:
# --------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Power BI automatically loads the selected table into 'dataset'
# Ensure you have 'pandas', 'scikit-learn', and 'matplotlib' installed in your Python environment.
df = dataset.copy()

# 1. Feature Selection
X = df[['Au_gpt', 'Cu_pct', 'As_ppm']]

# 2. Run K-Means Clustering (e.g. 3 Domains)
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# 3. Visualisation (Matplotlib is standard for PBI)
plt.figure(figsize=(10, 6))
# Scatter plot for 2 primary elements, colored by cluster
plt.scatter(df['Au_gpt'], df['Cu_pct'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Gold (g/t)')
plt.ylabel('Copper (%)')
plt.title('Automated Geological Domains (K-Means)')
plt.colorbar(label='Cluster ID')
plt.show()
    """, language="python")

st.markdown("---")
st.caption(t['footer'])
