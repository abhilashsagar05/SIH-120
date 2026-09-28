# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objs as go
import requests
import time

# ===========================================================================
# Configuration & Layout Setup
# ===========================================================================
st.set_page_config(
    page_title="WellTwin AI | Baghewala",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic the Bolt UI Theme
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0b1121;
        color: #e2e8f0;
    }
    
    /* Top Navigation Bar */
    .top-nav {
        background-color: #0f172a;
        padding: 10px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 20px;
        border-radius: 8px;
    }
    .nav-item {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .nav-value {
        color: #38bdf8;
        font-weight: bold;
    }
    .status-online {
        color: #10b981;
        font-weight: bold;
    }
    
    /* Panel Cards */
    .panel {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        height: 100%;
    }
    .panel h4 {
        margin-top: 0;
        font-size: 0.95rem;
        color: #cbd5e1;
        display: flex;
        align-items: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #f8fafc;
        margin: 10px 0 5px 0;
    }
    .metric-unit {
        font-size: 1rem;
        color: #94a3b8;
    }
    .metric-trend-down {
        color: #ef4444;
        font-size: 0.8rem;
    }
    .metric-trend-up {
        color: #10b981;
        font-size: 0.8rem;
    }
    
    /* Action Button */
    .action-btn {
        background-color: #0ea5e9;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        border: none;
        width: 100%;
        font-weight: bold;
        margin-top: 10px;
        cursor: pointer;
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #080f1e !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Hide default header */
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000"

# ===========================================================================
# Sidebar Navigation (Mimicking the UI)
# ===========================================================================
with st.sidebar:
    st.markdown("### 📈 WellTwin AI\n<span style='color:#64748b; font-size:0.8rem'>Baghewala Field</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation Links
    menu_items = [
        "⊞ Overview", "💠 Digital Twin", "🔥 CSS Optimizer", "⚙️ SRP Optimizer",
        "🧠 AI Predictions", "🛡️ Predictive Maintenance", "🧪 What-If Simulator",
        "⚖️ Scenario Comparison", "🔔 Alerts & Events", "📄 Reports"
    ]
    for item in menu_items:
        if "Digital Twin" in item:
            st.markdown(f"<div style='background:#0f172a; padding:8px; border-radius:4px; color:#38bdf8; border-left:3px solid #38bdf8'><b>{item}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:8px; color:#94a3b8;'>{item}</div>", unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("<span style='color:#64748b; font-size:0.7rem; letter-spacing:1px'>ENGINEER ONLY</span>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px; color:#94a3b8;'>🗄️ Historical Data</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px; color:#94a3b8;'>📊 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px; color:#94a3b8;'>⚙️ Settings</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<span style='color:#64748b; font-size:0.7rem'>ACTIVE ROLE</span><br><div style='background:#0ea5e9; padding:4px 8px; border-radius:4px; display:inline-block; font-size:0.8rem'>Operator</div> <span style='color:#64748b; font-size:0.8rem; margin-left:10px'>Engineer</span>", unsafe_allow_html=True)

# ===========================================================================
# Top Navigation Bar
# ===========================================================================
top_bar = """
<div class="top-nav">
    <div>
        <b>Baghewala WellTwin AI</b> <span style='color:#64748b; font-size:0.8rem'>Well-to-Surface Digital Twin</span>
    </div>
    <div style='display:flex; gap:20px; align-items:center'>
        <span class="status-online">● DIGITAL TWIN ONLINE</span>
        <span class="nav-item">Well: <span class="nav-value">BGW-001</span></span>
        <span class="nav-item">Data: <span class="nav-value">SYNTHETIC</span></span>
        <span class="nav-item">Sim: <span style='color:#f1c40f'>PAUSED</span></span>
        <span class="nav-item">T: 3h</span>
        <div style='background:#1e293b; padding:4px 8px; border-radius:4px; display:flex; gap:10px'>
            <span style='color:#10b981'>▶ Start</span>
            <span style='color:#f1c40f'>⏸ Pause</span>
            <span style='color:#94a3b8'>↺ Reset</span>
        </div>
        <div style='background:#1e293b; padding:4px 8px; border-radius:4px;'>
            <span style='color:#38bdf8'>1x</span> <span style='color:#94a3b8'>5x 10x</span>
        </div>
        <div style='background:#ea580c; color:white; padding:4px 8px; border-radius:4px; font-weight:bold'>SIH Demo</div>
    </div>
</div>
"""
st.markdown(top_bar, unsafe_allow_html=True)

# Title Row
st.markdown("#### ⚡ Command Centre — Well BGW-001\n<span style='color:#64748b; font-size:0.9rem'>Baghewala Field, Rajasthan, India • Heavy Crude 17-19° API</span>", unsafe_allow_html=True)

# ===========================================================================
# Metric Cards (Top Row)
# ===========================================================================
def fetch_api_data(endpoint: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=1)
        if resp.status_code == 200: return resp.json()
    except: pass
    return {}

css_data = fetch_api_data("css_status")
diag_data = fetch_api_data("diagnostics")

temp = css_data.get('reservoir_temperature_c', 85.0) if css_data else 85.0
visc = css_data.get('crude_viscosity_cP', 1689.0) if css_data else 1689.0

m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(f"""
    <div class="panel" style="border-top: 3px solid #10b981">
        <h4>🌡️ RESERVOIR TEMP</h4>
        <div class="metric-value">{temp:.1f} <span class="metric-unit">°C</span></div>
        <div class="metric-trend-down">↓ 0.0°C</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="panel" style="border-top: 3px solid #38bdf8">
        <h4>💧 VISCOSITY</h4>
        <div class="metric-value">{visc:,.0f} <span class="metric-unit">cP</span></div>
        <div style="color:#64748b; font-size:0.8rem">Evaluating...</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="panel" style="border-top: 3px solid #ef4444">
        <h4>📈 OIL PRODUCTION</h4>
        <div class="metric-value">4.4 <span class="metric-unit">bbl/d</span></div>
        <div class="metric-trend-down">↓ 0.0%</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="panel" style="border-top: 3px solid #f59e0b">
        <h4>⚖️ SOR</h4>
        <div class="metric-value">24.13</div>
        <div class="metric-trend-down">↓ 0.00</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown("""
    <div class="panel" style="border-top: 3px solid #10b981">
        <h4>⏱️ PUMP LOAD</h4>
        <div class="metric-value">38 <span class="metric-unit">%</span></div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown("""
    <div class="panel" style="border-top: 3px solid #10b981">
        <h4>⚠️ ROD FAILURE RISK</h4>
        <div class="metric-value">0 <span class="metric-unit">%</span></div>
        <div class="metric-trend-down">↓ 0%</div>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# AI Analysis Panels (Middle Row)
# ===========================================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="panel">
        <h4>⚡ What is happening?</h4>
        <span style='color:#64748b; font-size:0.8rem'>Current well status</span><br><br>
        <table style="width:100%; color:#cbd5e1; font-size:0.9rem">
            <tr><td style="padding:4px 0">Reservoir</td><td style="text-align:right; color:#10b981; font-weight:bold">Stable</td></tr>
            <tr><td style="padding:4px 0">Viscosity</td><td style="text-align:right; color:#10b981; font-weight:bold">Moderate</td></tr>
            <tr><td style="padding:4px 0">Production</td><td style="text-align:right; color:#ef4444; font-weight:bold">Declining</td></tr>
            <tr><td style="padding:4px 0">SRP Load</td><td style="text-align:right; color:#10b981; font-weight:bold">Normal</td></tr>
            <tr><td style="padding:4px 0">Failure Risk</td><td style="text-align:right; color:#10b981; font-weight:bold">LOW</td></tr>
            <tr><td style="padding:4px 0">CSS Stage</td><td style="text-align:right; color:#38bdf8; font-weight:bold">INJECTION</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="panel">
        <h4>🤖 What will happen?</h4>
        <span style='color:#64748b; font-size:0.8rem'>AI prediction</span><br><br>
        
        <div style="background:#1e293b; padding:10px; border-radius:6px; margin-bottom:10px">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase">Production Forecast (next 24h)</div>
            <div style="font-size:1.5rem; font-weight:bold; color:white">9.7 <span style="font-size:0.9rem; color:#94a3b8">bbl/d</span></div>
            <div style="color:#10b981; font-size:0.8rem">↑ 5.3 bbl/d predicted change</div>
        </div>
        
        <div style="background:#1e293b; padding:10px; border-radius:6px; margin-bottom:10px">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase">Risk Forecast</div>
            <div style="color:#10b981; font-weight:bold; font-size:0.9rem">● LOW - 1%</div>
        </div>
        
        <div style="background:#1e293b; padding:10px; border-radius:6px;">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase">SOR Forecast</div>
            <div style="font-size:1.2rem; font-weight:bold; color:white">0.77</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="panel">
        <h4>⚙️ What should we do?</h4>
        <span style='color:#64748b; font-size:0.8rem'>AI recommendation</span><br><br>
        <div style="font-weight:bold; color:white; margin-bottom:10px">Optimize CSS + SRP</div>
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.4; margin-bottom:15px">
            Adjust steam volume from 3200 to 1500 t/cycle. Change injection pressure from 95 to 140 bar. Modify soak time from 72 to 48 hours. Reduce SPM from 5.5 to 6. Adjust stroke length from 3 to 4 m. Change VFD speed from 45 to 35 Hz.
        </div>
        
        <table style="width:100%; color:#cbd5e1; font-size:0.85rem; margin-bottom:15px">
            <tr><td style="padding:2px 0">Production</td><td style="text-align:right; color:#10b981">+5.32 bbl/d</td></tr>
            <tr><td style="padding:2px 0">Energy</td><td style="text-align:right; color:#10b981">-158.82 kWh/bbl</td></tr>
            <tr><td style="padding:2px 0">Failure Risk</td><td style="text-align:right; color:#ef4444">+0.56 %</td></tr>
            <tr><td style="padding:2px 0">SOR</td><td style="text-align:right; color:#10b981">-23.36</td></tr>
        </table>
        
        <button class="action-btn">Apply Recommendation</button>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# Charts (Bottom Row)
# ===========================================================================
ch1, ch2 = st.columns([1.5, 1])

# Placeholder charts matching the dark blue theme
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=[1,2,3,4,5], y=[3.8, 3.9, 4.1, 4.0, 4.4], fill='tozeroy', line=dict(color='#10b981'), fillcolor='rgba(16, 185, 129, 0.1)'))
fig1.update_layout(
    title="<span style='color:white; font-size:16px'>📈 Production & Pump Load</span><br><span style='color:#64748b; font-size:12px'>Live simulation trend</span>",
    plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
    margin=dict(l=10, r=10, t=60, b=10),
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#64748b'))
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=[1,2,3,4,5], y=[94.2, 85.0, 72.6, 65.0, 52.5], fill='tozeroy', line=dict(color='#f97316'), fillcolor='rgba(249, 115, 22, 0.1)'))
fig2.update_layout(
    title="<span style='color:white; font-size:16px'>🌡️ Reservoir Temperature</span><br><span style='color:#64748b; font-size:12px'>Thermal response</span>",
    plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
    margin=dict(l=10, r=10, t=60, b=10),
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#64748b'))
)

with ch1:
    st.plotly_chart(fig1, use_container_width=True)
with ch2:
    st.plotly_chart(fig2, use_container_width=True)

# Auto-refresh mechanism
time.sleep(2)
st.rerun()
