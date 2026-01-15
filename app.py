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
    page = st.radio("Go to", [t['home_nav'], t['analysis_nav'], t['forecast_nav'], t['safety_nav'], t['prod_nav'], t['opt_nav'], t['mc_nav'], t['energy_nav'], t['pbi_nav']])
    
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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.info(t['ins_trifr'])
    
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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.warning(t['ins_fatigue'])
    
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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.success(t['ins_maint'])

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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.info(t['ins_yield'])
    
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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.success(t['ins_drill'])
    
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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.warning(t['ins_fleet'])

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
    
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.success(t['ins_pbi'])

# --- Page: Optimization (Linear Programming) ---
elif page == t['opt_nav']:
    st.title(t['opt_tit'])
    st.markdown(t['opt_desc'])
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader(t['opt_prob'])
        st.info(t['opt_obj'])
        for c in t['opt_cons']:
            st.text(c)
            
        # Interactive Inputs (Scenario Analysis)
        target_grade = st.slider("Target: Minimum Au Grade (g/t)", 1.0, 2.0, 1.4, 0.1)
        target_cap = st.number_input("Target: Mill Capacity (tpd)", 1000, 10000, 5000)

    # Solve Linear Programming Problem using Scipy
    from scipy.optimize import linprog

    # Available Stockpiles (Data)
    # SP A: High Grade (2.5 g/t), Low As (200 ppm), Expensive ($5/t)
    # SP B: Low Grade (1.0 g/t), Low As (100 ppm), Cheap ($2/t)
    # SP C: High Grade (2.0 g/t), High As (800 ppm), Medium ($3/t)
    
    # Variables: x0 (Tonnes A), x1 (Tonnes B), x2 (Tonnes C)
    # Objective: Minimize Cost = 5*x0 + 2*x1 + 3*x2
    c_obj = [5, 2, 3] 

    # Constraints (Inequalities are <=, so we flip > signs)
    # 1. Grade >= Target ->  -Grade*x <= -Target*Capacity 
    # (Since total tonnage is fixed equality constraint below, we can simplify linear constraint as weighted average sum)
    # Actually, simpler: Au*x0 + Au*x1 + Au*x2 >= Target * Total_Cap
    # Rearranged: -2.5x0 - 1.0x1 - 2.0x2 <= -Target * Total_Cap
    A_ub = [
        [-2.5, -1.0, -2.0],        # Grade Constraint (Negative for >=)
        [200, 100, 800]            # Arsenic Constraint (<= 480 ppm)
    ]
    b_ub = [
        -target_grade * target_cap, # Min Grade * Total Tonnage
        480 * target_cap            # Max As * Total Tonnage
    ]

    # Equality Constraint: x0 + x1 + x2 = Total_Cap
    A_eq = [[1, 1, 1]]
    b_eq = [target_cap]

    # Bounds: Tonnage cannot be negative
    bounds = [(0, None), (0, None), (0, None)]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    with col2:
        if res.success:
            st.success("**Solver Status:** Optimal Solution Found! ✅")
            
            # Results
            tonnes = res.x
            total_tons = sum(tonnes)
            final_grade = (tonnes[0]*2.5 + tonnes[1]*1.0 + tonnes[2]*2.0) / total_tons
            final_as = (tonnes[0]*200 + tonnes[1]*100 + tonnes[2]*800) / total_tons
            total_cost = res.fun
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Grade (Au)", f"{final_grade:.2f} g/t", delta=f"{final_grade - target_grade:.2f}")
            m2.metric("Final Arsenic", f"{final_as:.0f} ppm", delta=f"{480 - final_as:.0f} margin", delta_color="normal")
            m3.metric("Total Cost", f"${total_cost:,.0f}")
            
            # Pie Chart Visualization of the Blend
            blend_df = pd.DataFrame({
                'Stockpile': ['SP A (High Grade)', 'SP B (Low Grade)', 'SP C (High As)'],
                'Tonnes': tonnes
            })
            
            fig_opt = px.pie(blend_df, values='Tonnes', names='Stockpile', 
                             title=f"Optimal Blend Ratio (Total: {total_tons:,.0f} t)",
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_opt, use_container_width=True)
            
        else:
            st.error(f"Solver Failed: {res.message}")
            st.warning("Try lowering the Target Grade or increasing Allowable Arsenic.")
            
    # Insight Box
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.info(t['ins_opt'])

# --- Page: Monte Carlo Simulation ---
elif page == t['mc_nav']:
    st.title(t['mc_tit'])
    st.markdown(t['mc_desc'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        n_sim = st.slider("Iterations (N)", 1000, 10000, 5000, 1000)
        
        st.markdown("**Gold Price ($/oz)**")
        price_min = st.number_input("Min Price", 1800, 2200, 2000)
        price_mode = st.number_input("Most Likely", 2000, 2500, 2300)
        price_max = st.number_input("Max Price", 2400, 3000, 2700)
        
        st.markdown("**Cost Parameters**")
        cost_base = st.number_input("Base AISC ($/oz)", 1000, 1500, 1200)
        volatility = st.slider("Cost Volatility (+/- %)", 5, 20, 10) / 100
        
    # Run Simulation
    # 1. Generate Price Dist (Triangular)
    prices = np.random.triangular(price_min, price_mode, price_max, n_sim)
    
    # 2. Generate Cost Dist (Uniform)
    costs = np.random.uniform(cost_base * (1 - volatility), cost_base * (1 + volatility), n_sim)
    
    # 3. Calculate Profit Margin ($/oz)
    margins = prices - costs
    
    # 4. Total Annual Profit at 100k oz production
    annual_profit_m = margins * 100_000 / 1_000_000 # In Million USD
    
    # Stats
    p10 = np.percentile(annual_profit_m, 10)
    p50 = np.percentile(annual_profit_m, 50)
    p90 = np.percentile(annual_profit_m, 90)
    prob_loss = np.mean(annual_profit_m < 0) * 100
    
    with col2:
        st.subheader(t['mc_res'])
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("P90 (Conservative)", f"${p10:.1f} M") # P10 is low case
        m2.metric("P50 (Expected)", f"${p50:.1f} M")
        m3.metric("P10 (Optimistic)", f"${p90:.1f} M")   # P90 is high upside
        
        # Visualization
        fig_mc = px.histogram(annual_profit_m, nbins=50, 
                              title="Probabilistic Distribution of Annual Profit",
                              labels={'value': 'Net Profit ($ Million)'},
                              color_discrete_sequence=['#2ECC71'])
        
        fig_mc.add_vline(x=p10, line_dash="dash", line_color="white", annotation_text="P90 Risk")
        fig_mc.add_vline(x=p50, line_dash="solid", line_color="white", annotation_text="P50 Median")
        fig_mc.add_vline(x=p90, line_dash="dash", line_color="white", annotation_text="P10 Upside")
        
        fig_mc.update_layout(showlegend=False, template="plotly_dark")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        if prob_loss > 0:
            st.error(f"⚠️ Risk Alert: There is a {prob_loss:.1f}% probability of losing money.")
        else:
            st.success("✅ Robust Economics: 0% Probability of Loss in these scenarios.")

    # Insight Box
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.info(t['ins_mc'])

# --- Page: Energy Sector (Coal & Oil) ---
elif page == t['energy_nav']:
    st.title(t['energy_tit'])
    st.markdown(t['energy_desc'])
    
    # --- Tab 1: Coal Trading ---
    st.header(t['coal_tit'])
    st.markdown(t['coal_desc'])
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Contract Specs (PLTU)")
        spec_gar = st.number_input("Target GAR (kcal/kg)", 3800, 5000, 4200)
        spec_sulfur = st.number_input("Max Sulfur (%)", 0.5, 2.0, 1.0, 0.1)
        penalty_rate = st.number_input("Sulfur Penalty (USD/ton per 0.1%)", 0.5, 5.0, 2.0)
        
        st.markdown("---")
        st.subheader("Blending Ratio")
        # Simple slider for 2-product blend
        blend_pct = st.slider("% Low CV / Low Sulfur Coal", 0, 100, 60)
        
    with col2:
        # COAL DATA:
        # A: Low CV (4000), Low S (0.4%) - "Eco Coal"
        # B: High CV (4800), High S (1.8%) - "Dirty Coal"
        
        pct_a = blend_pct / 100
        pct_b = 1 - pct_a
        
        final_gar = (pct_a * 4000) + (pct_b * 4800)
        final_sulfur = (pct_a * 0.4) + (pct_b * 1.8)
        
        # Calculate Penalty
        # Usually checking if Sulfur > Spec
        sulfur_excess = max(0, final_sulfur - spec_sulfur)
        # Penalty is per 0.1% excess
        penalty_per_ton = (sulfur_excess / 0.1) * penalty_rate
        
        st.metric("Blended GAR", f"{final_gar:.0f} kcal/kg", delta=f"{final_gar - spec_gar:.0f}")
        st.metric("Blended Sulfur", f"{final_sulfur:.2f} %", delta=f"{spec_sulfur - final_sulfur:.2f}", delta_color="normal")
        
        if penalty_per_ton > 0:
            st.error(f"⚠️ SULFUR PENALTY APPLIED: ${penalty_per_ton:.2f} per ton")
        else:
            st.success("✅ NO PENALTY: Coal meets Sulfur specs.")
            
        # Chart
        coal_df = pd.DataFrame({
            'Metric': ['Calorific Value (kcal/kg)', 'Sulfur Content (%)'],
            'Value': [final_gar, final_sulfur],
            'Limit': [spec_gar, spec_sulfur]
        })
        
        # Visualize Composition
        fig_coal = go.Figure()
        fig_coal.add_trace(go.Bar(name='Actual', x=coal_df['Metric'], y=coal_df['Value'], marker_color='#F39C12'))
        fig_coal.add_trace(go.Bar(name='Contract Limit', x=coal_df['Metric'], y=coal_df['Limit'], marker_color='#7F8C8D'))
        fig_coal.update_layout(barmode='group', title="Blended Quality vs Contract Spec", template='plotly_dark')
        st.plotly_chart(fig_coal, use_container_width=True)

    with st.expander(t['ins_safe_tit'], expanded=True):
        st.info(t['ins_coal'])
        
    st.divider()
    
    # --- Tab 2: Oil & Gas (DCA) ---
    st.header(t['oil_tit'])
    st.markdown(t['oil_desc'])
    
    dca_col1, dca_col2 = st.columns([1, 2])
    
    with dca_col1:
        st.subheader("Reservoir Params")
        
        # Added Tooltips for Education
        qi = st.number_input("Initial Rate (qi) - bbl/day", 100, 5000, 1000, help=t['oil_help_qi'])
        di = st.slider("Initial Decline (Di) - %/year", 10, 90, 40, help=t['oil_help_di']) / 100
        b_factor = st.slider("Arps b-factor", 0.0, 1.0, 0.4, 0.1, help=t['oil_help_b'])
        eco_limit = st.number_input("Economic Limit (bbl/day)", 10, 100, 50, help="Production level where cost > revenue. Below this, the well is shut down.")
        
    with dca_col2:
        # Time vector (Months)
        t_months = np.arange(0, 60, 1) # 5 Years
        t_years = t_months / 12
        
        # Arps Equation: q(t) = qi / (1 + b * Di * t)^(1/b)
        # Handle b=0 (Exponential) separately
        if b_factor == 0:
            qt = qi * np.exp(-di * t_years)
        else:
            qt = qi / ((1 + b_factor * di * t_years) ** (1/b_factor))
            
        # Calculate Cumulative Production (Approx integration)
        cum_prod = np.cumsum(qt * 30.4) # Monthly sum
        
        # Find Economic Limit Cutoff
        profitable_months = qt >= eco_limit
        eur = np.sum(qt[profitable_months] * 30.4)
        remaining_life = np.sum(profitable_months)
        
        st.metric("Estimated Ultimate Recovery (EUR)", f"{eur/1000:.1f} k bbl")
        st.metric("Remaining Economic Life", f"{remaining_life:.0f} Months")
        
        # Chart
        fig_oil = go.Figure()
        fig_oil.add_trace(go.Scatter(x=t_months, y=qt, mode='lines', name='Production Rate', line=dict(color='#2ECC71', width=3)))
        fig_oil.add_hline(y=eco_limit, line_dash="dash", line_color="red", annotation_text="Economic Limit")
        
        fig_oil.update_layout(
            title="Production Forecast (Decline Curve)",
            xaxis_title="Months",
            yaxis_title="Oil Rate (bbl/day)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_oil, use_container_width=True)
        
    with st.expander(t['ins_safe_tit'], expanded=True):
        st.warning(t['ins_oil'])

st.markdown("---")
st.caption(t['footer'])
