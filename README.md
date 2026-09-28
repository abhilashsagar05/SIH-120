# Baghewala Heavy Oil Field — Well-to-Surface Digital Twin

## Overview
A modular Python digital twin for the Baghewala field (14-19 API heavy crude,
Jodhpur Sandstone), integrating CSS thermal modelling, SRP dynamometer simulation,
heavy-oil rheology (Power-Law shear-thinning), and SCADA telemetry synthesis.

## Quick Start
```bash
python setup_project.py          # scaffold (run once)
pip install -r requirements.txt
python -m src.data_ingestion     # smoke-test telemetry simulator
python -m src.fluid_dynamics     # smoke-test rheology engine
```
