# -*- coding: utf-8 -*-
"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : dashboard/app.py
  Author : Digital Twin Engineering Team (SIH-DS026)
  Version: 1.0.0

  Purpose
  -------
  The interactive Streamlit presentation layer. This dashboard serves as the 
  command center for the digital twin, polling the FastAPI backend and rendering
  real-time physics-informed predictions and autonomous control actions.
  
  Views:
  - Thermodynamic: Tracks Boberg-Lantz cooling and exponential viscosity spike.
  - Elastodynamic: Renders the PINN-generated downhole dynamometer card.
  - VFD Control: Displays the CNN diagnostic output and the asymmetric 
    Strokes Per Minute (SPM) commands dynamically mitigating "Rod Floating".
================================================================================
"""

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
import time
import pandas as pd

# ===========================================================================
# Configuration & Layout Setup
# ===========================================================================
st.set_page_config(
    page_title="Baghewala Digital Twin | SIH-DS026",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme hackathon aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .alert-text-healthy {
        color: #2ECC71;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .alert-text-warning {
        color: #F1C40F;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .alert-text-critical {
        color: #E74C3C;
        font-weight: bold;
        font-size: 1.2rem;
        animation: blink 1s linear infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session State Initialization (For Historical Trending)
# ===========================================================================
if "time_history" not in st.session_state:
    st.session_state.time_history = []
if "temp_history" not in st.session_state:
    st.session_state.temp_history = []
if "visc_history" not in st.session_state:
    st.session_state.visc_history = []

API_BASE_URL = "http://localhost:8000"


# ===========================================================================
# Sidebar Controls
# ===========================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Pump_Jack_in_Texas.jpg/320px-Pump_Jack_in_Texas.jpg", caption="SRP System")
    st.title("Control Panel")
    st.markdown("---")
    
    st.header("Simulation Control")
    run_simulation = st.checkbox("🟢 Start Live Simulation", value=False)
    
    st.markdown("---")
    st.header("VFD Baseline Config")
    baseline_spm = st.slider("Nominal Target SPM", min_value=1.0, max_value=10.0, value=6.0, step=0.5)
    
    st.markdown("---")
    st.markdown("**Team SIH-DS026**\n\n*Smart India Hackathon 2026*")


# ===========================================================================
# Main Dashboard Layout
# ===========================================================================
st.title("🛢️ Baghewala Well-to-Surface Digital Twin")
st.markdown("Real-time telemetry, thermodynamics, and autonomous CNN-VFD control for extreme heavy oil extraction.")

# Layout Columns
col_metrics = st.columns(4)
col_charts_top = st.columns((1.2, 1))

# Placeholders for dynamic rendering
metric_placeholders = [col.empty() for col in col_metrics]
thermo_placeholder = col_charts_top[0].empty()
mech_placeholder = col_charts_top[1].empty()
control_placeholder = st.empty()


# ===========================================================================
# Real-Time Rendering Loop
# ===========================================================================
def fetch_api_data(endpoint: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return {}

while run_simulation:
    # 1. Fetch live data from FastAPI
    css_data = fetch_api_data("css_status")
    dyn_data = fetch_api_data("dynamometer")
    diag_data = fetch_api_data("diagnostics")
    tele_data = fetch_api_data("telemetry")

    if not (css_data and dyn_data and diag_data and tele_data):
        st.warning("⚠️ Cannot connect to Backend API. Please ensure `python -m src.api` is running on port 8000.")
        time.sleep(2)
        continue

    # 2. Update Historical State
    current_time = css_data.get("simulation_day", 0)
    st.session_state.time_history.append(current_time)
    st.session_state.temp_history.append(css_data.get("reservoir_temperature_c", 0))
    st.session_state.visc_history.append(css_data.get("crude_viscosity_cP", 0))

    # Keep only recent 200 points to prevent memory overflow
    if len(st.session_state.time_history) > 200:
        st.session_state.time_history.pop(0)
        st.session_state.temp_history.pop(0)
        st.session_state.visc_history.pop(0)

    # 3. Render Top Metrics
    vfd_cmd = diag_data.get("vfd_command", {})
    probs = diag_data.get("cnn_probabilities", {})
    is_floating = diag_data.get("rod_floating_alert", False)
    
    with metric_placeholders[0].container():
        st.metric(label="🌡️ Heated Zone Temp", value=f"{css_data.get('reservoir_temperature_c', 0):.1f} °C")
    with metric_placeholders[1].container():
        st.metric(label="💧 Crude Viscosity", value=f"{css_data.get('crude_viscosity_cP', 0):,.0f} cP")
    with metric_placeholders[2].container():
        st.metric(label="⬆️ Upstroke SPM", value=f"{vfd_cmd.get('upstroke_spm', 0):.1f}")
    with metric_placeholders[3].container():
        st.metric(label="⬇️ Downstroke SPM", value=f"{vfd_cmd.get('downstroke_spm', 0):.1f}")

    # 4. Render Thermodynamic Plot (Boberg-Lantz)
    fig_thermo = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Temperature Line
    fig_thermo.add_trace(
        go.Scatter(x=st.session_state.time_history, y=st.session_state.temp_history, 
                   name="Reservoir Temp (°C)", line=dict(color="#E74C3C", width=3)),
        secondary_y=False,
    )
    # Viscosity Line
    fig_thermo.add_trace(
        go.Scatter(x=st.session_state.time_history, y=st.session_state.visc_history, 
                   name="Viscosity (cP)", line=dict(color="#3498DB", width=3, dash="dot")),
        secondary_y=True,
    )
    
    # Mark Economic Cut-off threshold (~8000 cP)
    fig_thermo.add_hline(y=8000, secondary_y=True, line_dash="dash", line_color="yellow", 
                         annotation_text="Economic Cut-off (8000 cP)", annotation_position="bottom right")

    fig_thermo.update_layout(
        title="<b>Reservoir Thermodynamics (Boberg-Lantz)</b>",
        xaxis_title="Simulation Time (Days)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_thermo.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig_thermo.update_yaxes(title_text="Viscosity (cP)", type="log", secondary_y=True) # Log scale for massive viscosity

    thermo_placeholder.plotly_chart(fig_thermo, use_container_width=True, key=f"thermo_{current_time}")

    # 5. Render Mechanical Plot (PINN Dynamometer Card)
    fig_mech = go.Figure()
    
    pos_m = dyn_data.get("position_m", [])
    load_kn = dyn_data.get("load_kN", [])
    
    fig_mech.add_trace(go.Scatter(
        x=pos_m, y=load_kn, 
        mode='lines+markers',
        marker=dict(color="#2ECC71" if not is_floating else "#E74C3C", size=6),
        line=dict(color="#2ECC71" if not is_floating else "#E74C3C", width=2),
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.2)" if not is_floating else "rgba(231, 76, 60, 0.3)"
    ))
    
    # Baseline 0 kN to visualize compressive buckling
    fig_mech.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig_mech.update_layout(
        title="<b>PINN Downhole Dynamometer Card</b>",
        xaxis_title="Plunger Displacement [m]",
        yaxis_title="Plunger Load [kN]",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[-10, 80]), # Fixed scale to show dropping to zero
        xaxis=dict(range=[-0.2, 3.2])
    )
    
    mech_placeholder.plotly_chart(fig_mech, use_container_width=True, key=f"mech_{current_time}")

    # 6. Render Autonomous VFD Control Banner
    # Determine the status text based on highest probability
    if probs:
        dom_class = max(probs, key=probs.get)
    else:
        dom_class = "Healthy"

    if dom_class == "Rod Floating":
        status_html = f"<div class='alert-text-critical'>🚨 CRITICAL: ROD FLOATING ({probs.get('Rod Floating', 0)*100:.0f}%) — Asymmetric Profiling Activated</div>"
    elif dom_class == "Fluid Pound":
        status_html = f"<div class='alert-text-warning'>⚠️ WARNING: Fluid Pound ({probs.get('Fluid Pound', 0)*100:.0f}%) — Slowing Operations</div>"
    else:
        status_html = f"<div class='alert-text-healthy'>✅ System Nominal ({probs.get('Healthy', 0)*100:.0f}%)</div>"

    with control_placeholder.container():
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("### CNN Diagnostics & VFD Actuation")
        st.markdown(status_html, unsafe_allow_html=True)
        
        # Display the Asymmetric speeds being commanded
        up = vfd_cmd.get('upstroke_spm', 0)
        down = vfd_cmd.get('downstroke_spm', 0)
        torque = vfd_cmd.get('motor_torque_limit_pct', 0)
        
        st.markdown(
            f"> **Command Sent to VFD:** Modulate Upstroke to **{up:.1f} SPM**, "
            f"Downstroke to **{down:.1f} SPM**. Torque Limit set to **{torque:.1f}%**."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # 7. Regulate refresh rate
    time.sleep(0.3)
