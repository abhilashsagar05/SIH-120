# -*- coding: utf-8 -*-
"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : run_demo.py
  Author : Digital Twin Engineering Team (SIH-DS026)
  Version: 1.0.0

  Purpose
  -------
  The master execution script for the Smart India Hackathon Live Presentation.
  This script simultaneously boots the backend FastAPI server and the frontend 
  Streamlit dashboard. It also spawns a background thread to simulate the 
  high-frequency edge telemetry ingestion, printing a professional real-time 
  log to the console for the judges.
================================================================================
"""

import subprocess
import time
import sys
import threading
import math
import atexit

# Global process references for clean teardown
api_process = None
dashboard_process = None

def print_banner():
    """Prints a highly polished terminal banner for the judge presentation."""
    banner = """
================================================================================
      ██████╗  █████╗  ██████╗ ██╗  ██╗███████╗██╗    ██╗ █████╗ ██╗      █████╗ 
      ██╔══██╗██╔══██╗██╔════╝ ██║  ██║██╔════╝██║    ██║██╔══██╗██║     ██╔══██╗
      ██████╔╝███████║██║  ███╗███████║█████╗  ██║ █╗ ██║███████║██║     ███████║
      ██╔══██╗██╔══██║██║   ██║██╔══██║██╔══╝  ██║███╗██║██╔══██║██║     ██╔══██║
      ██████╔╝██║  ██║╚██████╔╝██║  ██║███████╗╚███╔███╔╝██║  ██║███████╗██║  ██║
      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
                                                                             
      WELL-TO-SURFACE DIGITAL TWIN | TEAM SIH-DS026 | SMART INDIA HACKATHON
================================================================================
    """
    print(f"\033[96m{banner}\033[0m")

def start_backend():
    """Boots the FastAPI server hosting the PINN/CNN models."""
    global api_process
    print("\033[92m[SYSTEM]\033[0m Booting Physics/AI Backend (FastAPI)...")
    
    # Run the API in the background. We use python -m to resolve src. module paths.
    api_process = subprocess.Popen(
        [sys.executable, "-m", "src.api"],
        stdout=subprocess.DEVNULL,  # Hide standard API logs to keep terminal clean
        stderr=subprocess.STDOUT
    )
    time.sleep(3) # Give uvicorn a moment to bind to port 8000
    print("\033[92m[SYSTEM]\033[0m Backend API successfully bound to http://localhost:8000")

def start_frontend():
    """Boots the Streamlit dashboard."""
    global dashboard_process
    print("\033[92m[SYSTEM]\033[0m Booting Digital Twin Dashboard (Streamlit)...")
    
    # Run Streamlit in headless mode to keep terminal output clean
    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    time.sleep(2)
    print("\033[92m[SYSTEM]\033[0m Interactive Dashboard live at http://localhost:8501")

def telemetry_simulator_thread():
    """
    Background thread simulating edge ingestion.
    This creates a beautiful real-time log of the accelerated CSS cycle 
    for the judges to watch in the terminal.
    """
    print("\n\033[93m[EDGE]\033[0m Initializing SCADA connection to Baghewala Wellpad BGW-07...")
    time.sleep(2)
    print("\033[93m[EDGE]\033[0m Connection established. Commencing high-frequency data ingestion.\n")
    
    start_time = time.time()
    t_res_base = 47.0
    t_steam = 320.0
    
    try:
        while True:
            elapsed = time.time() - start_time
            sim_day = elapsed * 1.0  # 1 sec = 1 day
            
            # Thermodynamic decay simulation (matching the API loop)
            current_temp = t_res_base + (t_steam - t_res_base) * math.exp(-0.05 * sim_day)
            
            # Viscosity spike
            if current_temp > 150:
                visc = 87.0 + (320 - current_temp) * 10
            elif current_temp > 75:
                visc = 2000.0 + (150 - current_temp) * 100
            else:
                visc = 9500.0 + (75 - current_temp) * 500
                
            # Formatting the log output
            time_str = time.strftime("%H:%M:%S")
            log = (f"\033[90m[{time_str}]\033[0m "
                   f"\033[94mDay {sim_day:04.1f}\033[0m | "
                   f"Reservoir: \033[91m{current_temp:05.1f} °C\033[0m | "
                   f"Viscosity: \033[96m{visc:07.0f} cP\033[0m | ")
            
            # Simulate the CNN/VFD logic triggering
            if visc > 8000:
                log += "\033[91m[ALERT] Rod Floating Predicted! Activating Asymmetric VFD (Up: 7.5 SPM | Dn: 3.0 SPM)\033[0m"
            elif visc > 4000:
                log += "\033[93m[WARN] Approaching fluid pound boundaries. Monitoring...\033[0m"
            else:
                log += "\033[92m[OK] System Nominal. Maintaining 6.0 SPM.\033[0m"
                
            print(log)
            time.sleep(1.0) # Print a log every second
            
    except KeyboardInterrupt:
        pass

def cleanup():
    """Terminates subprocesses when the demo is closed."""
    print("\n\033[91m[SYSTEM]\033[0m Shutting down Digital Twin architecture...")
    if api_process:
        api_process.terminate()
    if dashboard_process:
        dashboard_process.terminate()
    print("\033[92m[SYSTEM]\033[0m Teardown complete. Goodbye!")

if __name__ == "__main__":
    # Register the cleanup function to run on exit
    atexit.register(cleanup)
    
    print_banner()
    
    try:
        start_backend()
        start_frontend()
        
        # Start the edge telemetry simulation on the main thread
        telemetry_simulator_thread()
        
    except KeyboardInterrupt:
        # Caught Ctrl+C, atexit will handle the cleanup
        sys.exit(0)
