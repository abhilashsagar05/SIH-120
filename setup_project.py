"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  File   : setup_project.py
  Purpose: One-shot project scaffold generator.
           Run this script ONCE from the repository root to create the full
           directory tree, stub __init__.py packages, and placeholder files.
  Usage  : python setup_project.py
================================================================================
"""

import os
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical directory / file manifest
# ---------------------------------------------------------------------------
# Each tuple is  (relative_path, file_content_or_None)
#   - If content is None  → create the directory only.
#   - If content is ""    → create an empty file (e.g. __init__.py stubs).
#   - Otherwise           → write the content string verbatim.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()

MANIFEST: list[tuple[str, str | None]] = [
    # ── Source package ───────────────────────────────────────────────────────
    ("src/__init__.py",         "# Baghewala Digital Twin — source package\n"),
    ("src/data_ingestion.py",   None),   # populated separately
    ("src/fluid_dynamics.py",   None),   # populated separately
    ("src/thermal_model.py",    "# Placeholder: CSS thermal-front propagation model\n"),
    ("src/srp_model.py",        "# Placeholder: Sucker-Rod Pump dynamometer & rod-string model\n"),
    ("src/production_model.py", "# Placeholder: IPR / VLP nodal-analysis integration\n"),
    ("src/utils.py",            "# Placeholder: shared helpers (unit conversion, logging, I/O)\n"),

    # ── Data directories ─────────────────────────────────────────────────────
    ("data/raw/.gitkeep",       ""),
    ("data/processed/.gitkeep", ""),
    ("data/synthetic/.gitkeep", ""),

    # ── Trained / serialised models ──────────────────────────────────────────
    ("models/.gitkeep",         ""),

    # ── Interactive dashboard ────────────────────────────────────────────────
    ("dashboard/__init__.py",   "# Baghewala Digital Twin — dashboard package\n"),
    ("dashboard/app.py",        "# Placeholder: Dash / Streamlit real-time dashboard entry-point\n"),
    ("dashboard/layouts.py",    "# Placeholder: page layout definitions\n"),
    ("dashboard/callbacks.py",  "# Placeholder: reactive callback functions\n"),

    # ── Tests ────────────────────────────────────────────────────────────────
    ("tests/__init__.py",       ""),
    ("tests/test_rheology.py",  "# Placeholder: pytest suite for fluid_dynamics.py\n"),
    ("tests/test_scada.py",     "# Placeholder: pytest suite for data_ingestion.py\n"),

    # ── Notebooks ────────────────────────────────────────────────────────────
    ("notebooks/.gitkeep",      ""),

    # ── Config & environment ─────────────────────────────────────────────────
    ("config/field_params.yaml", textwrap.dedent("""\
        # Baghewala Field Parameters — override defaults in source modules
        reservoir:
          api_gravity_min: 14.0          # degrees API
          api_gravity_max: 19.0          # degrees API
          bht_min_c: 46.0                # Bottom-hole temperature, Celsius
          bht_max_c: 48.0                # Bottom-hole temperature, Celsius
          viscosity_insitu_min_cp: 10000 # In-situ dynamic viscosity, cP
          viscosity_insitu_max_cp: 13000 # In-situ dynamic viscosity, cP
          formation: Jodhpur Sandstone

        srp:
          strokes_per_minute: 4.0        # SPM nominal
          stroke_length_m: 3.0           # Polished-rod stroke length, m
          rod_string_od_m: 0.022         # Rod OD, m

        css:
          steam_injection_temp_c: 300.0  # Injected steam temperature, Celsius
          cycle_duration_days: 45        # Total CSS cycle duration
          soak_duration_days: 7          # Soak period within cycle
    """)),

    ("requirements.txt", textwrap.dedent("""\
        numpy>=1.26
        pandas>=2.2
        scipy>=1.13
        matplotlib>=3.9
        pyyaml>=6.0
        pytest>=8.0
    """)),

    ("README.md", textwrap.dedent("""\
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
    """)),
]


# ---------------------------------------------------------------------------
# Scaffold engine
# ---------------------------------------------------------------------------

def create_structure(manifest: list[tuple[str, str | None]], root: Path) -> None:
    created_dirs: set[Path] = set()
    created_files: int = 0
    skipped_files: int = 0

    for rel_path_str, content in manifest:
        target = root / rel_path_str

        parent = target.parent
        if parent not in created_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            created_dirs.add(parent)

        if content is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [DIR ]  {target.parent.relative_to(root)}/")
            continue

        if target.exists() and target.stat().st_size > 0:
            existing = target.read_text(encoding="utf-8")
            if not existing.strip().startswith("# Placeholder"):
                print(f"  [SKIP]  {rel_path_str}  (exists — will not overwrite)")
                skipped_files += 1
                continue

        target.write_text(content, encoding="utf-8")
        print(f"  [FILE]  {rel_path_str}")
        created_files += 1

    print()
    print(f"Scaffold complete: {created_files} file(s) written, {skipped_files} skipped.")


if __name__ == "__main__":
    print("=" * 72)
    print("  Baghewala Digital Twin - Project Scaffold")
    print(f"  Root : {ROOT}")
    print("=" * 72)
    create_structure(MANIFEST, ROOT)
