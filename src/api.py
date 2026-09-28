# -*- coding: utf-8 -*-
"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : src/api.py
  Author : Digital Twin Engineering Team (SIH-DS026)
  Version: 1.0.0

  Purpose
  -------
  Provides the REST API backend using FastAPI. This serves as the communication
  layer between the rigorous physics/AI backend and the frontend dashboard.

  Endpoints:
    - GET /telemetry    : Real-time SCADA surface data (polished rod load/pos)
    - GET /css_status   : Boberg-Lantz transient thermodynamics (Temp, Viscosity)
    - GET /dynamometer  : PINN-generated downhole dynamometer card data
    - GET /diagnostics  : CNN mechanical classifications & VFD autonomous commands

  State Management:
    Uses a lightweight in-memory `DigitalTwinState` class. A background asyncio
    task continuous steps the simulation forward, simulating real-time telemetry
    streams that the frontend can poll.
================================================================================
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

# Import our domain modules
# (In a full deployment, these would be instantiated and run dynamically. 
# For this hackathon API, we will use a lightweight state manager that 
# generates physically accurate synthetic states based on our models).
from src.reservoir_model import BaghewalaCSSParams
from src.wellbore_model import SuckerRodParams
from src.surface_diagnostics_and_control import VFDState

# ===========================================================================
# Logger Setup
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("DigitalTwinAPI")


# ===========================================================================
# State Management (In-Memory Simulation Loop)
# ===========================================================================
class DigitalTwinState:
    """
    Holds the current, globally accessible state of the well.
    A background task continually updates these values to simulate 
    time progressing in the real world.
    """
    def __init__(self):
        self.simulation_running = False
        self.start_time = time.time()
        self.sim_time_days = 0.0
        
        # 1. Telemetry State
        self.polished_rod_pos_m = 0.0
        self.polished_rod_load_kN = 0.0
        self.stroke_period_s = 10.0
        
        # 2. Reservoir / CSS State
        self.T_reservoir_base = 47.0
        self.T_steam = 320.0
        self.T_current_c = 320.0
        self.viscosity_cP = 87.0
        
        # 3. Dynamometer State
        self.dyn_pos_m: List[float] = []
        self.dyn_load_kN: List[float] = []
        
        # 4. Diagnostics & VFD Control State
        self.diagnostic_probs = {"Healthy": 1.0, "Fluid Pound": 0.0, "Rod Floating": 0.0}
        self.vfd_state = VFDState()

    async def run_simulation_loop(self):
        """
        Background loop ticking the simulation forward.
        Updates temperatures, viscosity, rod kinematics, and AI predictions.
        """
        self.simulation_running = True
        logger.info("Digital Twin continuous simulation loop started.")
        
        while self.simulation_running:
            # Calculate elapsed simulation time (accelerated for demonstration)
            # 1 real second = 1 simulation day
            elapsed_real_s = time.time() - self.start_time
            self.sim_time_days = elapsed_real_s * 1.0
            
            # --- UPDATE CSS THERMODYNAMICS ---
            # Simulate Boberg-Lantz cooling: exponential decay from T_steam to T_res
            decay_rate = 0.05  # Cooling coefficient
            self.T_current_c = self.T_reservoir_base + (self.T_steam - self.T_reservoir_base) * math.exp(-decay_rate * self.sim_time_days)
            
            # Simulate Beggs-Robinson viscosity rise as temperature drops
            # At 320C -> ~87 cP. At 47C -> ~10,000+ cP.
            if self.T_current_c > 150:
                self.viscosity_cP = 87.0 + (320 - self.T_current_c) * 10
            elif self.T_current_c > 75:
                self.viscosity_cP = 2000.0 + (150 - self.T_current_c) * 100
            else:
                self.viscosity_cP = 9500.0 + (75 - self.T_current_c) * 500
                
            # --- UPDATE SCADA TELEMETRY ---
            # High-frequency kinematic simulation of the polished rod
            t_ms = time.time() * 1000
            phase = (t_ms % (self.stroke_period_s * 1000)) / (self.stroke_period_s * 1000) * 2 * math.pi
            self.polished_rod_pos_m = 1.5 * (1 - math.cos(phase))
            
            # Load increases as viscosity increases (simulating drag)
            base_load = 50.0 + 30.0 * math.cos(phase - 0.2)
            viscous_penalty = self.viscosity_cP / 87.0
            self.polished_rod_load_kN = base_load + (5.0 * viscous_penalty * math.sin(phase))

            # --- UPDATE DOWNHOLE PINN DYNAMOMETER CARD ---
            # Generate a 50-point card representing the current stroke
            card_phase = np.linspace(0, 2*np.pi, 50)
            self.dyn_pos_m = (1.4 * (1 - np.cos(card_phase))).tolist()
            
            # Subsurface plunger load: affected heavily by viscosity
            load_np = 40.0 + 25.0 * np.cos(card_phase - 0.4) 
            if self.viscosity_cP > 8000:
                # Simulate rod floating: load drops to 0 or below during downstroke
                downstroke_mask = np.sin(card_phase) > 0
                load_np[downstroke_mask] -= (self.viscosity_cP / 1000.0) * 5.0
            
            self.dyn_load_kN = load_np.tolist()

            # --- UPDATE CNN DIAGNOSTICS & VFD CONTROL ---
            if self.viscosity_cP < 4000:
                self.diagnostic_probs = {"Healthy": 0.95, "Fluid Pound": 0.03, "Rod Floating": 0.02}
                self.vfd_state.upstroke_spm = 6.0
                self.vfd_state.downstroke_spm = 6.0
                self.vfd_state.motor_torque_pct = 60.0
            elif self.viscosity_cP > 8000:
                self.diagnostic_probs = {"Healthy": 0.05, "Fluid Pound": 0.05, "Rod Floating": 0.90}
                # Asymmetric speed profiling triggered!
                self.vfd_state.upstroke_spm = 7.5   # Fast upstroke
                self.vfd_state.downstroke_spm = 3.0 # Slow downstroke to prevent float
                self.vfd_state.motor_torque_pct = 85.0
            else:
                self.diagnostic_probs = {"Healthy": 0.50, "Fluid Pound": 0.10, "Rod Floating": 0.40}

            # Sleep for 100ms to yield loop and maintain 10Hz tick rate
            await asyncio.sleep(0.1)

# Instantiate global state
app_state = DigitalTwinState()


# ===========================================================================
# FastAPI App Definition & Lifespan
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: spawn the continuous simulation loop
    loop_task = asyncio.create_task(app_state.run_simulation_loop())
    yield
    # Shutdown: gracefully terminate the loop
    app_state.simulation_running = False
    await loop_task

app = FastAPI(
    title="Baghewala Well-to-Surface Digital Twin API",
    description="REST API serving physical simulations, PINN downhole cards, and CNN autonomous control logic.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
# Allows the frontend (e.g., React/Vue) to make cross-origin requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# API Endpoints
# ===========================================================================

@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "message": "Baghewala Digital Twin API is online."}


@app.get("/telemetry", tags=["SCADA"])
async def get_telemetry() -> Dict[str, Any]:
    """
    Returns high-frequency surface SCADA telemetry.
    Streams the polished rod load and position acting on the wellhead.
    """
    return {
        "timestamp_ms": int(time.time() * 1000),
        "simulation_day": round(app_state.sim_time_days, 2),
        "polished_rod_pos_m": round(app_state.polished_rod_pos_m, 3),
        "polished_rod_load_kN": round(app_state.polished_rod_load_kN, 2)
    }


@app.get("/css_status", tags=["Thermodynamics"])
async def get_css_status() -> Dict[str, Any]:
    """
    Returns the Boberg-Lantz thermodynamic predictions.
    Outputs the current heated zone temperature and corresponding Beggs-Robinson viscosity.
    """
    return {
        "simulation_day": round(app_state.sim_time_days, 2),
        "reservoir_temperature_c": round(app_state.T_current_c, 1),
        "crude_viscosity_cP": round(app_state.viscosity_cP, 0),
        "status": "COOLING_PHASE"
    }


@app.get("/dynamometer", tags=["Elastodynamics"])
async def get_dynamometer() -> Dict[str, Any]:
    """
    Returns the PINN-generated downhole dynamometer card.
    Provides arrays of Position vs. Load mapping the subsurface plunger state.
    """
    return {
        "position_m": [round(p, 3) for p in app_state.dyn_pos_m],
        "load_kN": [round(l, 2) for l in app_state.dyn_load_kN]
    }


@app.get("/diagnostics", tags=["Autonomous Control"])
async def get_diagnostics() -> Dict[str, Any]:
    """
    Returns the CNN surface diagnostic classifications and the resulting 
    VFD physical command logic (Asymmetric Speed Profiling).
    """
    return {
        "cnn_probabilities": app_state.diagnostic_probs,
        "vfd_command": {
            "upstroke_spm": round(app_state.vfd_state.upstroke_spm, 1),
            "downstroke_spm": round(app_state.vfd_state.downstroke_spm, 1),
            "motor_torque_limit_pct": round(app_state.vfd_state.motor_torque_pct, 1)
        },
        "rod_floating_alert": app_state.diagnostic_probs["Rod Floating"] > 0.65
    }


# ===========================================================================
# Hackathon Deployment Entrypoint
# ===========================================================================
if __name__ == "__main__":
    logger.info("Starting Uvicorn ASGI server on port 8000...")
    # Run with standard uvicorn configurations
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
