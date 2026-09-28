# -*- coding: utf-8 -*-
"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : src/reservoir_model.py
  Classes: BaghewalaCSSParams, ReservoirThermodynamics
  Author : Digital Twin Engineering Team
  Version: 1.0.0

  Purpose
  -------
  Implements the Boberg-Lantz (1966) analytical CSS thermal model for the
  Baghewala Jodhpur Sandstone formation. This module governs the complete
  Cyclic Steam Stimulation "Huff and Puff" cycle:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Phase 1 — INJECTION  : Steam (300–350°C, x=0.75) injected at      │
    │                         ~80 t/day for 10–20 days. The Marx-         │
    │                         Langenheim (1959) model calculates the       │
    │                         geometry of the heated zone.                 │
    │                                                                     │
    │  Phase 2 — SOAK       : Injection shut-in. Steam condenses, heat   │
    │                         equilibrates radially, viscosity drops       │
    │                         from ~12,000 cP to < 200 cP at T_peak.     │
    │                                                                     │
    │  Phase 3 — PRODUCTION : Mobile oil produced. Heated zone cools via │
    │                         three simultaneous loss mechanisms modelled  │
    │                         by the Boberg-Lantz ODE (see §3.5).        │
    └─────────────────────────────────────────────────────────────────────┘

  Boberg-Lantz Energy Balance ODE (the core of this module)
  ----------------------------------------------------------
  Let θ(t) = (T̄(t) − T_r) / (T̄_0 − T_r)  [dimensionless temperature excess]

  The energy balance on the heated zone of area A_h, thickness h, and
  volumetric heat capacity M_R yields the ODE:

      dθ/dt = −θ(t) × [Γ_vert(t) + Γ_rad(t) + Γ_fluid]          … (BL)

  where the three decay-rate coefficients are:

      Γ_vert(t) = 2λ_cap / (M_R × h × √(π × α_cap × t))
          [Vertical conduction to overburden + underburden; singular at t=0,
           integrable; accounts for heat loss from both faces of the pay zone]

      Γ_rad(t)  = 2πλ_res / (M_R × A_h × ln(r_front(t) / r_h))
          [Radial conduction beyond heated-zone edge; r_front = √(4α_res × t)
           is the logarithmic thermal diffusion front radius]

      Γ_fluid   = q_liq × M_fluid / (M_R × A_h × h)
          [Heat carried away by produced oil + water; quasi-constant for
           fixed liquid withdrawal rate q_liq]

  Heated Zone Geometry (Marx-Langenheim, 1959)
  ---------------------------------------------
  The area A_h heated by steam injection is calculated from the G-function:

      G(t_D) = e^{t_D} × erfc(√t_D) + 2√(t_D / π) − 1           … (ML)
      t_D    = 4 × α_cap × t_inj / h²                   [dimensionless time]

      A_h = (q_inj × t_inj × G(t_D)) / (M_R × h × ΔT_s)          … (A_h)

  SOR and GHG Cutoff Logic
  ------------------------
  evaluate_css_cutoff() evaluates five criteria in parallel and recommends
  the EARLIEST (most conservative) cutoff day:

      1. T̄ < T_cutoff              → viscosity-driven pumping penalty
      2. viscous_penalty > factor   → SRP overload / Scope 1 GHG waste
      3. SOR_cumulative > limit     → economic steam threshold
      4. SOR_rate > warning         → SOR deterioration rate
      5. η_energy < minimum         → thermodynamic efficiency floor

  References
  ----------
  [1] Boberg, T.C. & Lantz, R.B. (1966). "Calculation of the Production Rate
      of a Thermally Stimulated Well." JPT, Dec 1966, pp. 1613–1623. SPE-1772.
  [2] Marx, J.W. & Langenheim, R.H. (1959). "Reservoir Heating by Hot Fluid
      Injection." Trans. AIME, 216, pp. 312–315.
  [3] Beggs, H.D. & Robinson, J.R. (1975). "Estimating the Viscosity of Crude
      Oil Systems." JPT, Sep 1975, pp. 1140–1141. SPE-5434.
================================================================================
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import erfc  # complementary error function: erfc(x) = 1-erf(x)

# ===========================================================================
# Module-level Logger — UTF-8 enforced for Windows cp1252 terminals
# ===========================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(stream=sys.stderr)
    try:
        _handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 — stream.reconfigure unavailable
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ===========================================================================
# Physical Constants (SI)
# ===========================================================================
SEC_PER_DAY   : float = 86_400.0         # seconds per day
DARCY_TO_M2   : float = 9.869_233e-13   # 1 milliDarcy = 9.869e-13 m²
BAR_TO_PA     : float = 1.0e5            # 1 bar = 100,000 Pa
RHO_WATER_STP : float = 998.2            # kg/m³  pure water at 20°C, 1 atm
BBL_TO_M3     : float = 0.158_987        # 1 US barrel = 0.159 m³
M3_TO_BBL     : float = 6.289_811        # 1 m³ = 6.29 bbl


# ===========================================================================
# §1  Steam Table Interpolation
#     (Tabulated from IAPWS-IF97 standard, valid 100–374°C saturation line)
# ===========================================================================

# Saturation temperatures [°C] — the breakpoints for linear interpolation
_SAT_T_C  = np.array([100,  120,  140,  160,  180,  200,  220,  240,
                        260,  280,  300,  320,  340,  360,  374.14],
                       dtype=np.float64)

# Latent heat of vaporisation h_fg [kJ/kg] at each saturation temperature
_H_FG_KJ = np.array([2256.7, 2203.0, 2145.0, 2083.0, 2015.0, 1941.0,
                       1859.0, 1768.0, 1662.0, 1545.0, 1404.0, 1238.0,
                       1027.0, 720.0,  0.0],
                      dtype=np.float64)

# Saturated liquid enthalpy h_f [kJ/kg] at each saturation temperature
_H_F_KJ   = np.array([419.1,  503.7,  589.1,  675.6,  763.2,  852.4,
                        943.7, 1037.5, 1134.9, 1236.0, 1345.0, 1462.0,
                       1596.0, 1761.0, 2099.3],
                       dtype=np.float64)


def _steam_h_fg(T_sat_c: float) -> float:
    """
    Return the latent heat of vaporisation h_fg [kJ/kg] at saturation
    temperature T_sat_c [°C] using IAPWS-IF97 table interpolation.
    Valid range: 100°C – 374°C (critical point of water).
    """
    return float(np.interp(T_sat_c, _SAT_T_C, _H_FG_KJ))


def _steam_h_f(T_sat_c: float) -> float:
    """
    Return the saturated liquid specific enthalpy h_f [kJ/kg] at T_sat_c [°C].
    """
    return float(np.interp(T_sat_c, _SAT_T_C, _H_F_KJ))


def _steam_total_enthalpy(T_sat_c: float, quality: float) -> float:
    """
    Return total specific enthalpy of wet steam [kJ/kg]:

        h_ws = h_f + x × h_fg

    where x = steam quality (dryness fraction) ∈ [0, 1].
    x = 0 → saturated liquid;  x = 1 → dry saturated steam.
    """
    return _steam_h_f(T_sat_c) + quality * _steam_h_fg(T_sat_c)


# ===========================================================================
# §2  CSS Parameter Dataclass
# ===========================================================================

@dataclass
class BaghewalaCSSParams:
    """
    Comprehensive parameter container for one CSS-operated Baghewala well.

    All parameters use SI base units internally. Temperatures are specified
    in °C for field convenience and converted to Kelvin inside formulas.
    Defaults are calibrated to the Jodhpur Sandstone (Bikaner-Nagaur basin)
    at ~620 m TVD with 14–19°API heavy crude.

    Sections
    --------
    A. Well identification
    B. Reservoir petrophysics (permeability, geometry, saturation)
    C. Reservoir thermal properties (volumetric heat capacity, conductivity)
    D. Overburden/underburden (cap rock) thermal properties
    E. Produced fluid thermal properties
    F. CSS injection operational parameters
    G. CSS production operational parameters
    H. Economic and GHG cutoff thresholds
    I. Derived quantities (computed in __post_init__)
    """

    # ── A. Well identifier ────────────────────────────────────────────────────
    well_id: str = "BGW-07"

    # ── B. Reservoir petrophysics ─────────────────────────────────────────────
    T_reservoir_c:    float = 47.0    # °C   baseline bottom-hole temperature (BHT)
    k_md:             float = 850.0   # mD   horizontal permeability, Jodhpur Sst
    h_net_m:          float = 18.0    # m    net pay thickness (gross ~25 m, NTG~0.72)
    phi:              float = 0.30    # -    effective porosity
    S_o:              float = 0.72    # -    initial oil saturation
    r_e_m:            float = 150.0   # m    drainage radius (250 m well spacing grid)
    r_w_m:            float = 0.108   # m    wellbore radius (6-5/8" casing OD → r_w)
    P_reservoir_bar:  float = 38.0    # bar  average reservoir pore pressure
    P_wellbore_bar:   float = 12.0    # bar  average flowing bottomhole pressure
    oil_api:          float = 16.5    # °API crude gravity (Baghewala mid-range)

    # ── C. Reservoir thermal properties ───────────────────────────────────────
    # M_R = φ×S_o×(ρc_p)_oil + φ×(1-S_o)×(ρc_p)_water + (1-φ)×(ρc_p)_rock
    # For Jodhpur Sandstone with ~30% porosity and 72% oil saturation:
    M_R_kJ_m3_K:       float = 2100.0 # kJ/(m³·K) bulk volumetric heat capacity
    lambda_res_W_mK:    float = 1.40   # W/(m·K)   effective thermal conductivity

    # ── D. Cap rock / base rock thermal properties ────────────────────────────
    # Cap rock (shale/mudstone) governs VERTICAL heat loss — the dominant loss
    # mechanism in thin pay zones like Baghewala (~18 m net pay).
    M_cap_kJ_m3_K:     float = 2200.0 # kJ/(m³·K) volumetric heat capacity, cap rock
    lambda_cap_W_mK:    float = 1.60   # W/(m·K)   thermal conductivity, cap rock

    # ── E. Produced fluid thermal properties ──────────────────────────────────
    rho_oil_kg_m3:      float = 962.0  # kg/m³   live-oil density at reservoir P,T
    cp_oil_J_kgK:       float = 2050.0 # J/(kg·K) specific heat capacity of crude
    rho_water_kg_m3:    float = 1005.0 # kg/m³   formation water density
    cp_water_J_kgK:     float = 4182.0 # J/(kg·K) specific heat of formation water
    water_cut_frac:     float = 0.30   # -        water cut (WOR = 0.43 bbl/bbl)

    # ── F. CSS Injection parameters ───────────────────────────────────────────
    T_steam_c:          float = 320.0    # °C    injection steam temperature
    steam_quality:      float = 0.75     # -     dryness fraction at wellhead
    m_dot_steam_kg_d:   float = 80_000.0 # kg/day injection rate
    t_injection_days:   float = 15.0     # days  injection phase duration
    t_soak_days:        float = 7.0      # days  soak phase (shut-in duration)

    # ── G. CSS Production parameters ──────────────────────────────────────────
    q_liquid_m3_d:      float = 45.0    # m³/day  total liquid withdrawal rate
    B_o:                float = 1.05    # -       oil formation volume factor

    # ── H. Economic & GHG cutoff thresholds ───────────────────────────────────
    SOR_economic:       float = 5.0     # bbl CWE steam / bbl oil — cumulative limit
    SOR_rate_warning:   float = 0.50    # bbl/bbl per day — SOR deterioration rate
    T_cutoff_c:         float = 72.0    # °C  minimum T̄ for economic pumped production
    eta_energy_min:     float = 0.12    # -   minimum thermal energy return ratio
    srp_overload_factor:float = 3.5     # -   viscous power penalty trigger multiplier

    # ── I. Derived quantities (set in __post_init__) ──────────────────────────
    alpha_res_m2_s:     float = field(init=False)  # m²/s  reservoir thermal diffusivity
    alpha_cap_m2_s:     float = field(init=False)  # m²/s  cap rock thermal diffusivity
    M_fluid_J_m3_K:     float = field(init=False)  # J/(m³·K) produced fluid vol. heat cap.

    def __post_init__(self) -> None:
        # ── Thermal diffusivities: α = λ / M_vol ─────────────────────────────
        # M_vol is volumetric heat capacity [J/(m³·K)] = M_kJ × 1000
        self.alpha_res_m2_s = self.lambda_res_W_mK / (self.M_R_kJ_m3_K * 1e3)
        self.alpha_cap_m2_s = self.lambda_cap_W_mK / (self.M_cap_kJ_m3_K * 1e3)

        # ── Volumetric heat capacity of the produced fluid mixture ────────────
        # M_fluid = oil_frac × ρ_oil × c_p_oil  +  water_frac × ρ_water × c_p_water
        # Units: (kg/m³) × (J/(kg·K)) = J/(m³·K)
        oil_frac             = 1.0 - self.water_cut_frac
        self.M_fluid_J_m3_K  = (
            oil_frac             * self.rho_oil_kg_m3   * self.cp_oil_J_kgK
            + self.water_cut_frac * self.rho_water_kg_m3 * self.cp_water_J_kgK
        )

        # ── Validation ────────────────────────────────────────────────────────
        assert self.T_steam_c > self.T_reservoir_c, (
            f"Steam temperature ({self.T_steam_c}°C) must exceed "
            f"reservoir temperature ({self.T_reservoir_c}°C)."
        )
        assert 0.0 < self.steam_quality <= 1.0, "Steam quality must be in (0, 1]."
        assert self.T_cutoff_c > self.T_reservoir_c, (
            "Cutoff temperature must exceed reservoir temperature."
        )

        logger.debug(
            "CSSParams | α_res=%.2e m²/s | α_cap=%.2e m²/s | "
            "M_fluid=%.0f J/(m³·K)",
            self.alpha_res_m2_s, self.alpha_cap_m2_s, self.M_fluid_J_m3_K,
        )


# ===========================================================================
# §3  ReservoirThermodynamics — Boberg-Lantz CSS Engine
# ===========================================================================

class ReservoirThermodynamics:
    """
    Implements the full Boberg-Lantz (1966) analytical model for CSS thermal
    dynamics in the Baghewala heavy oil field.

    The class is organised into five functional groups:
    ───────────────────────────────────────────────────
    §3.1  steam_injection_properties()   — IAPWS steam thermodynamics
    §3.2  _marx_langenheim_g() + compute_heated_zone()  — injection zone geometry
    §3.3  _beggs_robinson_viscosity_cP() — dead oil viscosity (Beggs-Robinson 1975)
    §3.4  _darcy_oil_rate_m3_d()         — temperature-coupled Darcy flow
    §3.5  _boberg_lantz_ode()            — ODE RHS for BL energy balance
    §3.6  simulate_production_phase()    — scipy RK45 ODE integration
    §3.7  evaluate_css_cutoff()          — multi-criterion cutoff recommendation
    §3.8  generate_cycle_report()        — end-to-end convenience wrapper
    """

    def __init__(self, params: Optional[BaghewalaCSSParams] = None) -> None:
        self.p = params if params is not None else BaghewalaCSSParams()
        logger.info(
            "ReservoirThermodynamics init | well=%s | T_res=%.1f°C | "
            "T_steam=%.1f°C (x=%.2f) | k=%.0f mD | h=%.1f m",
            self.p.well_id, self.p.T_reservoir_c,
            self.p.T_steam_c, self.p.steam_quality,
            self.p.k_md, self.p.h_net_m,
        )

    # =========================================================================
    # §3.1  Steam Injection Thermodynamics
    # =========================================================================

    def steam_injection_properties(self) -> Dict[str, float]:
        """
        Compute thermodynamic quantities for the injected steam package.

        The NET usable thermal energy delivered to the reservoir per unit mass
        of steam is the total steam enthalpy minus the "baseline" enthalpy of
        water already at reservoir temperature:

            Δh_useful = h_ws − h_w(T_res)   [kJ/kg]

        where h_ws = h_f(T_s) + x·h_fg(T_s) is the wet-steam enthalpy.

        The reference enthalpy of water at reservoir temperature uses the
        approximation h_w(T) ≈ c_p_water × T_c [kJ/kg] (valid for T < 100°C).

        Cold Water Equivalent (CWE) volume — industry-standard steam measure:
            V_CWE = m_steam_total / ρ_water_STP   [m³]

        Returns
        -------
        dict with keys:
            h_fg_kJ_kg          Latent heat at T_steam [kJ/kg]
            h_f_kJ_kg           Saturated liquid enthalpy at T_steam [kJ/kg]
            h_ws_kJ_kg          Total wet-steam enthalpy [kJ/kg]
            h_ref_kJ_kg         Water enthalpy at T_reservoir [kJ/kg]
            delta_h_kJ_kg       Net usable enthalpy (h_ws − h_ref) [kJ/kg]
            q_inj_W             Heat injection rate [W]
            m_steam_total_kg    Total steam mass injected [kg]
            Q_total_GJ          Total heat injected [GJ]
            V_CWE_m3            Steam as cold-water equivalent [m³]
            V_CWE_bbl           Steam CWE in barrels [bbl]
        """
        p = self.p

        h_fg  = _steam_h_fg(p.T_steam_c)                            # kJ/kg
        h_f   = _steam_h_f(p.T_steam_c)                             # kJ/kg
        h_ws  = h_f + p.steam_quality * h_fg                        # kJ/kg  wet steam

        # Reservoir-temperature water enthalpy (baseline reference)
        # h_w(T_res) = c_p_water × T_res ≈ 4.18 × 47 ≈ 196.5 kJ/kg
        c_p_liq    = 4.18                                            # kJ/(kg·K)
        h_ref      = c_p_liq * p.T_reservoir_c                      # kJ/kg

        delta_h    = h_ws - h_ref                                    # kJ/kg net

        # Total mass and heat injected
        m_total    = p.m_dot_steam_kg_d * p.t_injection_days        # kg
        Q_total_GJ = m_total * delta_h * 1e-6                       # GJ

        # Heat injection RATE [W] = (mass/day × delta_h_kJ/kg × 1000 J/kJ) / 86400 s/day
        q_inj_W = (p.m_dot_steam_kg_d * delta_h * 1e3) / SEC_PER_DAY  # W

        # Cold Water Equivalent volume
        V_CWE_m3  = m_total / RHO_WATER_STP                         # m³
        V_CWE_bbl = V_CWE_m3 * M3_TO_BBL                           # bbl

        result = {
            "h_fg_kJ_kg"        : h_fg,
            "h_f_kJ_kg"         : h_f,
            "h_ws_kJ_kg"        : h_ws,
            "h_ref_kJ_kg"       : h_ref,
            "delta_h_kJ_kg"     : delta_h,
            "q_inj_W"           : q_inj_W,
            "m_steam_total_kg"  : m_total,
            "Q_total_GJ"        : Q_total_GJ,
            "V_CWE_m3"          : V_CWE_m3,
            "V_CWE_bbl"         : V_CWE_bbl,
        }
        logger.info(
            "Steam: h_fg=%.1f kJ/kg | h_ws=%.1f kJ/kg | Δh=%.1f kJ/kg | "
            "Q_total=%.2f GJ | V_CWE=%.1f m³ (%.0f bbl)",
            h_fg, h_ws, delta_h, Q_total_GJ, V_CWE_m3, V_CWE_bbl,
        )
        return result

    # =========================================================================
    # §3.2  Marx-Langenheim Heated Zone Model
    # =========================================================================

    def _marx_langenheim_g(self, t_D: np.ndarray) -> np.ndarray:
        """
        Compute the Marx-Langenheim (1959) G-function.

        The G-function is the dimensionless heated-area growth function that
        accounts for progressive heat loss to the overburden during injection.
        It was derived by Marx & Langenheim (1959) by solving the transient
        heat conduction problem for a growing steam zone with continuous heat
        loss to a semi-infinite overburden.

        Definition (equation 10 from Marx & Langenheim, 1959):

            G(t_D) = e^{t_D} · erfc(√t_D) + 2√(t_D/π) − 1        … (ML-G)

        Physical interpretation:
        ──────────────────────────────────────────────────────────────────
        ┌───────────┬───────────────────────────────────────────────────┐
        │   t_D     │  Behaviour                                        │
        ├───────────┼───────────────────────────────────────────────────┤
        │   → 0     │  G → 0  (no area heated at injection start)      │
        │   small   │  G ≈ 2√(t_D/π) − t_D  (losses dominate)         │
        │   large   │  G ≈ 2√(t_D/π)        (linear in √injection_time)│
        │   → ∞     │  G grows without bound (ever-expanding steam zone)│
        └───────────┴───────────────────────────────────────────────────┘

        Dimensionless time:  t_D = 4 · α_cap · t_inj / h²
        where α_cap = cap rock thermal diffusivity [m²/s].

        Parameters
        ----------
        t_D : np.ndarray  Dimensionless injection time(s), ≥ 0.

        Returns
        -------
        G : np.ndarray  Marx-Langenheim G values ≥ 0 (clipped for stability).
        """
        t_D = np.asarray(t_D, dtype=np.float64)

        # Numerical guard: at t_D = 0 the formula returns −1 which is wrong;
        # we clamp small values and handle analytically G(0) = 0.
        safe = np.maximum(t_D, 1e-10)

        # G(t_D) = exp(t_D) · erfc(√t_D) + 2·√(t_D/π) − 1
        G = (np.exp(safe) * erfc(np.sqrt(safe))
             + 2.0 * np.sqrt(safe / np.pi)
             - 1.0)

        # Clip to [0, ∞) — tiny negative values arise from floating-point cancellation
        return np.maximum(G, 0.0)

    def compute_heated_zone(self) -> Dict[str, float]:
        """
        Calculate the heated zone geometry and initial temperature after CSS
        injection and soak phases using the Marx-Langenheim (1959) model.

        Algorithm
        ---------
        Step 1 — Dimensionless injection time:
                    t_D_inj = 4 · α_cap · t_inj_s / h²

        Step 2 — Marx-Langenheim G-function at t_D_inj:
                    G_inj = G(t_D_inj)

        Step 3 — Net heat injection rate [W]:
                    q_inj = ṁ_steam · Δh_useful  [W]

        Step 4 — Heat stored in the heated cylinder after accounting for
                 vertical overburden losses (Marx-Langenheim heat balance):
                    Q_stored = q_inj · t_inj_s · G_inj  [J]

        Step 5 — Heated zone area A_h from the thermal storage equation:
                 The Marx-Langenheim model states that the steam zone volume
                 is filled uniformly at T_steam, so:
                    A_h = Q_stored / (M_R · h · ΔT_s)  [m²]
                 where ΔT_s = T_steam − T_reservoir

        Step 6 — Heated zone radius:
                    r_h = √(A_h / π)  [m]

        Step 7 — Initial average temperature at start of production.
                 During the soak phase, the steam condenses and heat
                 re-distributes within the pay zone. The initial average
                 temperature is estimated from the energy stored per unit
                 reservoir volume:
                    T̄_0 = T_reservoir + Q_stored / (M_R · A_h · h)
                 Physically T̄_0 ≤ T_steam (clamped for numerical safety).

        Returns
        -------
        dict with keys:
            t_D_inj      Dimensionless injection time [-]
            G_inj        Marx-Langenheim G at end of injection [-]
            q_inj_W      Heat injection rate [W]
            A_h_m2       Heated zone area [m²]
            r_h_m        Heated zone radius [m]
            V_h_m3       Heated zone bulk volume [m³]
            T_bar_0_c    Initial average temperature at production start [°C]
            theta_0      Initial dimensionless temperature excess θ₀ [-]
            Q_stored_GJ  Thermal energy stored in heated zone [GJ]
        """
        p = self.p
        sp = self.steam_injection_properties()

        # ── Step 1: Dimensionless injection time ──────────────────────────────
        # t_D = 4 · α_cap · t_inj / h²   [-]
        t_inj_s  = p.t_injection_days * SEC_PER_DAY    # s
        t_D_inj  = 4.0 * p.alpha_cap_m2_s * t_inj_s / (p.h_net_m ** 2)

        # ── Step 2: Marx-Langenheim G-function ───────────────────────────────
        G_inj = float(self._marx_langenheim_g(np.array([t_D_inj]))[0])

        # ── Step 3: Heat injection rate [W] ───────────────────────────────────
        q_inj_W = sp["q_inj_W"]

        # ── Step 4: Thermal energy stored in heated zone [J] ─────────────────
        # Q_stored = q_inj[W] × t_inj[s] × G(t_D)
        # The factor G < 1 represents the fraction of injected heat that
        # actually heats the reservoir (the rest is lost to overburden).
        Q_stored_J  = q_inj_W * t_inj_s * G_inj
        Q_stored_GJ = Q_stored_J * 1e-9

        # ── Step 5: Heated zone area (Marx-Langenheim corrected formula) ─────
        # The correct ML (1959) heated-area equation (their Eq. 7):
        #
        #   A_h = (Q_inj_total / (M_R × h × ΔT_s)) × G(t_D)
        #
        # Derivation:
        #   Q_inj_total = total heat poured into the system via steam [J]
        #   Without overburden losses, steam would heat area = Q_total/(M_R·h·ΔT).
        #   G(t_D) is the areal efficiency factor: only fraction G of that
        #   theoretical area actually reaches T_steam temperature.
        #   → Q_stored = M_R × A_h × h × ΔT_s  (heat retained in heated zone)
        #
        M_R_J      = p.M_R_kJ_m3_K * 1e3               # J/(m³·K)
        delta_T_s  = p.T_steam_c - p.T_reservoir_c      # K
        Q_inj_total_J = q_inj_W * t_inj_s               # J  total injected heat

        A_h_m2 = (Q_inj_total_J / (M_R_J * p.h_net_m * delta_T_s)) * G_inj

        # Minimum area floor: at least the heated annulus out to 5×r_w
        # (prevents degenerate near-zero geometry at very small t_D)
        A_h_floor_m2 = np.pi * (5.0 * p.r_w_m) ** 2
        A_h_m2       = max(A_h_m2, A_h_floor_m2)

        # Recompute Q_stored from the actual A_h used (for energy bookkeeping)
        Q_stored_J  = M_R_J * A_h_m2 * p.h_net_m * delta_T_s
        Q_stored_GJ = Q_stored_J * 1e-9

        # ── Step 6: Heated zone radius ────────────────────────────────────────
        r_h_m  = np.sqrt(A_h_m2 / np.pi)
        V_h_m3 = A_h_m2 * p.h_net_m                    # bulk heated volume [m³]

        # ── Step 7: Initial average temperature ───────────────────────────────
        # Reverse-compute T̄_0 from stored energy (handles partial heating):
        # T̄_0 = T_r + Q_stored / (M_R × V_h)
        T_bar_0_c = p.T_reservoir_c + Q_stored_J / (M_R_J * V_h_m3)
        T_bar_0_c = min(T_bar_0_c, p.T_steam_c)  # physical upper bound

        # Dimensionless initial temperature excess θ₀ ∈ (0, 1]
        theta_0 = (T_bar_0_c - p.T_reservoir_c) / delta_T_s

        logger.info(
            "Heated zone: t_D=%.4f | G=%.4f | A_h=%.1f m² | r_h=%.1f m | "
            "T̄_0=%.1f°C | θ₀=%.4f | Q_stored=%.2f GJ",
            t_D_inj, G_inj, A_h_m2, r_h_m, T_bar_0_c, theta_0, Q_stored_GJ,
        )
        return {
            "t_D_inj"     : t_D_inj,
            "G_inj"       : G_inj,
            "q_inj_W"     : q_inj_W,
            "A_h_m2"      : A_h_m2,
            "r_h_m"       : r_h_m,
            "V_h_m3"      : V_h_m3,
            "T_bar_0_c"   : T_bar_0_c,
            "theta_0"     : theta_0,
            "Q_stored_GJ" : Q_stored_GJ,
        }

    # =========================================================================
    # §3.3  Beggs-Robinson (1975) Dead Oil Viscosity Correlation
    # =========================================================================

    def _beggs_robinson_viscosity_cP(
        self,
        T_c: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Estimate dead-oil dynamic viscosity [cP] using the Beggs-Robinson (1975)
        correlation, the industry standard for heavy crude API correlations.

        Correlation development (Beggs & Robinson, SPE-5434, 1975):
        ─────────────────────────────────────────────────────────────
        Regression fitted on 5,008 measurements from 600 crude systems.
        Range: API = 16–58°,  T = 70–295°F (21–146°C).
        For Baghewala (API ≈ 16.5°), results above 146°C are extrapolated.

        Step 1 — API-dependent scaling coefficient:
                    Z = 3.0324 − 0.02023 × API_gravity
                    X = 10^Z

        Step 2 — Dead oil viscosity at temperature T_F [°F]:
                    log₁₀(log₁₀(μ_od + 1)) = X × T_F^(−1.163)

                 Solving for μ_od:
                    inner_exp = X × T_F^(−1.163)
                    μ_od      = 10^(10^inner_exp) − 1   [cP]

        Physical check at baseline conditions (API=16.5, T=47°C=116.6°F):
            Z = 3.0324 − 0.02023 × 16.5 = 2.6986
            X = 10^2.6986 = 499.4
            inner = 499.4 × 116.6^(−1.163) = 499.4 × 0.003667 = 1.831
            μ_od = 10^(10^1.831) − 1 = 10^67.8 − 1 ≈ 6.3e67 cP
            → This extreme value confirms the fluid is near-solid at BHT
              without CSS. We clip to 100,000 cP (physical maximum for flow).

        Parameters
        ----------
        T_c : Temperature [°C]. Scalar or NumPy array.

        Returns
        -------
        mu_cP : Dead oil dynamic viscosity [cP], clipped to [0.5, 100,000].
        """
        API = self.p.oil_api
        T_c = np.asarray(T_c, dtype=np.float64)

        # Convert °C → °F  (Beggs-Robinson requires Fahrenheit)
        T_F = T_c * 9.0 / 5.0 + 32.0
        T_F = np.maximum(T_F, 70.0)  # floor at 70°F to prevent log underflow

        # API-dependent constant Z and its power X
        Z = 3.0324 - 0.02023 * API     # intercept in double-log space
        X = np.power(10.0, Z)           # temperature scaling factor

        # Double-logarithmic viscosity correlation:
        #   log10(log10(μ+1)) = X × T_F^(−1.163)
        inner_exp = X * np.power(T_F, -1.163)       # log10(log10(μ+1))

        # Guard: for very low T_F (heavy crude at low T), inner_exp can be huge.
        # Cap to avoid overflow: μ is physically bounded by gel-point viscosity.
        # At T=47°C (116.6°F), API=16.5: inner_exp ≈ 1.83, log10(μ+1) ≈ 67
        # → μ → ∞; field data shows ~10,000-13,000 cP. We cap at 3.0 (outer exp)
        # so that μ_max = 10^(10^3.0) − 1 = 10^1000 − 1 → clipped to 100,000.
        inner_exp = np.minimum(inner_exp, 3.0)       # log10(μ+1) ≤ 10^3 = 1000

        log10_mu_p1 = np.power(10.0, inner_exp)     # log10(μ+1)
        log10_mu_p1 = np.minimum(log10_mu_p1, 5.0)  # μ+1 ≤ 10^5 = 100,000

        mu_cP = np.power(10.0, log10_mu_p1) - 1.0   # cP

        # Physical bounds [0.5, 100,000] cP
        return np.clip(mu_cP, 0.5, 100_000.0)

    # =========================================================================
    # §3.4  Temperature-Coupled Darcy Radial Flow
    # =========================================================================

    def _darcy_oil_rate_m3_d(
        self,
        T_bar_c: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Compute instantaneous oil production rate [m³/day at surface conditions]
        using the steady-state Darcy radial flow equation with temperature-
        dependent viscosity from the Beggs-Robinson correlation.

        Darcy Radial Flow (Productivity Index form)
        ─────────────────────────────────────────────
        For a circular drainage area with well at centre, the flow rate is:

            q_res = (2π × k × h × ΔP) / (μ_o(T̄) × B_o × [ln(r_e/r_w) − ¾])

            [m³/s] = [m²] × [m] × [Pa] / ([Pa·s] × [-] × [-])

        The Dupuit correction −¾ accounts for:
        - ½  → average pressure in the drainage area (vs. boundary pressure)
        - ¼  → skin effect assumed zero for a CSS-stimulated well

        Conversion to surface oil rate [m³/day]:
            q_oil_surface = q_res × (1 − WC) / B_o × SEC_PER_DAY
            (divide by B_o to convert reservoir → surface volume,
             multiply by (1−WC) to get oil fraction of produced liquids)

        Physical coupling with CSS:
        ────────────────────────────
        As T̄ drops (heated zone cools), μ_o(T̄) increases dramatically,
        reducing q_oil via the 1/μ dependence. This is the fundamental
        economic driver for the CSS cutoff: shrinking oil rate + fixed steam
        cost = deteriorating SOR.

        Parameters
        ----------
        T_bar_c : Average heated-zone temperature [°C]. Scalar or array.

        Returns
        -------
        q_oil_m3_d : Surface oil production rate [m³/day].
        """
        p = self.p

        # Permeability [m²] (1 mD = 9.869e-13 m²)
        k_m2 = p.k_md * DARCY_TO_M2

        # Pressure drawdown [Pa]
        dP_Pa = (p.P_reservoir_bar - p.P_wellbore_bar) * BAR_TO_PA

        # Oil viscosity at current average temperature [Pa·s]
        mu_Pa_s = self._beggs_robinson_viscosity_cP(T_bar_c) * 1e-3

        # Dupuit steady-state radial flow logarithmic term [-]
        # The −¾ is the standard Dupuit correction for pseudo-steady-state flow
        ln_term = np.log(p.r_e_m / p.r_w_m) - 0.75

        # Reservoir volumetric flow rate [m³/s]
        # q = (2π × k × h × ΔP) / (μ × B_o × ln_term)
        q_res_m3_s = (2.0 * np.pi * k_m2 * p.h_net_m * dP_Pa) / (
            mu_Pa_s * p.B_o * ln_term
        )

        # Convert to surface oil rate [m³/day]
        # q_oil_surface = q_res × (1 − WC) / B_o × SEC_PER_DAY
        q_oil_m3_d = q_res_m3_s * (1.0 - p.water_cut_frac) / p.B_o * SEC_PER_DAY

        return np.maximum(q_oil_m3_d, 0.0)  # physical lower bound

    # =========================================================================
    # §3.5  Boberg-Lantz ODE Right-Hand Side
    # =========================================================================

    def _boberg_lantz_ode(
        self,
        t_s: float,
        y: np.ndarray,
        A_h_m2: float,
        r_h_m: float,
        T_bar_0_c: float,
    ) -> np.ndarray:
        """
        Right-hand side of the Boberg-Lantz (1966) energy balance ODE.

        This function is passed directly to scipy.integrate.solve_ivp and
        is called thousands of times during the RK45 numerical integration.
        All equations are derived from the energy balance on the heated zone.

        State Vector
        ─────────────
        y[0] = θ(t)  :  Dimensionless temperature excess ∈ [0, 1]
                          θ = (T̄ − T_r) / (T̄_0 − T_r)
                          θ = 1 → T̄ = T̄_0 (initial post-soak temperature)
                          θ = 0 → T̄ = T_r  (reservoir baseline, fully cooled)

        y[1] = V_oil :  Cumulative oil produced [m³ at surface conditions]

        Derivation of the ODE
        ─────────────────────
        Full energy balance on the heated cylinder (volume V_h = A_h × h):

            M_R × V_h × dT̄/dt = − q̇_vert(t) − q̇_rad(t) − q̇_fluid(t)

        Substituting T̄ = T_r + θ × ΔT_0  (where ΔT_0 = T̄_0 − T_r = const):

            M_R × A_h × h × ΔT_0 × dθ/dt = −θ × ΔT_0 × [Σ rate terms]

        Dividing by M_R × A_h × h × ΔT_0:

            dθ/dt = −θ × [Γ_vert(t) + Γ_rad(t) + Γ_fluid]             … (BL)

        ──────────────────────────────────────────────────────────────────────
        Γ_VERT(t): Vertical Conduction Loss Coefficient [s⁻¹]
        ──────────────────────────────────────────────────────────────────────
        The heated zone (thickness h) is bounded above and below by cap rock
        and base rock. Heat conducts outward from both faces into semi-infinite
        solid half-spaces, each initially at T_r.

        For a suddenly-heated surface in a semi-infinite solid, the heat flux
        is given by the half-space Green's function:
            q̇_face = λ_cap × A_h × (T̄ − T_r) / √(π × α_cap × t)

        Including both top and bottom faces (factor of 2):
            q̇_vert = 2 × λ_cap × A_h × θ × ΔT_0 / √(π × α_cap × t)

        Therefore:
            Γ_vert(t) = q̇_vert / (M_R × A_h × h × ΔT_0 × θ)
                       = 2λ_cap / (M_R × h × √(π × α_cap × t))

        Note: Γ_vert → ∞ as t → 0 (integrable singularity: ∫₀ᵗ Γ_vert dτ = finite).
              Guard: use t_safe = max(t, 1 s) to avoid numerical blow-up at start.

        ──────────────────────────────────────────────────────────────────────
        Γ_RAD(t): Radial Conduction Loss Coefficient [s⁻¹]
        ──────────────────────────────────────────────────────────────────────
        Heat conducts radially outward from the hot cylinder (radius r_h) into
        the cooler, unsteamed reservoir. As time progresses, the thermal
        diffusion front r_front(t) = √(4 × α_res × t) expands outward.

        The steady-state logarithmic approximation (valid for large t) gives:
            q̇_rad = 2π × h × λ_res × (T̄ − T_r) / ln(r_front / r_h)

        Therefore:
            Γ_rad(t) = 2π × λ_res / (M_R × A_h × ln(r_front(t) / r_h))

        Guard: ln(r_front / r_h) ≥ ε_floor to avoid division by zero at small t.

        ──────────────────────────────────────────────────────────────────────
        Γ_FLUID: Produced Fluid Heat Removal Coefficient [s⁻¹]
        ──────────────────────────────────────────────────────────────────────
        Produced oil and water leaving the heated zone carry their sensible
        heat above reservoir temperature. Volumetric heat removal rate:
            q̇_fluid = q_liq × (ρc_p)_mix × (T̄ − T_r)
                     = q_liq × M_fluid × θ × ΔT_0

        Therefore:
            Γ_fluid = q_liq_m3_s × M_fluid / (M_R × A_h × h)

        This coefficient is constant for a fixed liquid withdrawal rate.

        Parameters
        ----------
        t_s       : Current time from production phase start [seconds].
        y         : State vector [θ, V_oil_m3].
        A_h_m2    : Heated zone planar area [m²] (constant after injection).
        r_h_m     : Heated zone equivalent radius [m].
        T_bar_0_c : Initial average temperature at start of production [°C].

        Returns
        -------
        dydt : [dθ/dt [s⁻¹], dV_oil/dt [m³/s]]
        """
        p = self.p

        # ── Unpack state vector ───────────────────────────────────────────────
        # Clamp θ to [0, 1]: cannot exceed initial conditions or go negative.
        theta  = float(np.clip(y[0], 0.0, 1.0))

        # Current absolute temperature [°C]
        T_bar_c = p.T_reservoir_c + theta * (T_bar_0_c - p.T_reservoir_c)

        # ── Precompute shared constants ───────────────────────────────────────
        M_R_J = p.M_R_kJ_m3_K * 1e3   # J/(m³·K) volumetric heat cap reservoir
        h     = p.h_net_m               # m  net pay thickness
        A_h   = A_h_m2                  # m²
        r_h   = r_h_m                   # m

        # Numerical guard against t=0 singularity in Γ_vert
        t_safe = max(t_s, 1.0)          # 1-second minimum (≈ 0.0 production time)

        # ─────────────────────────────────────────────────────────────────────
        # Γ_VERT: Vertical conduction rate coefficient
        # Γ_vert = 2λ_cap / (M_R × h × √(π × α_cap × t))
        # ─────────────────────────────────────────────────────────────────────
        Gamma_vert = (
            2.0 * p.lambda_cap_W_mK
            / (M_R_J * h * np.sqrt(np.pi * p.alpha_cap_m2_s * t_safe))
        )

        # ─────────────────────────────────────────────────────────────────────
        # Γ_RAD: Radial conduction rate coefficient
        # r_front(t) = √(4 × α_res × t)    [thermal diffusion front radius]
        # Γ_rad = 2π × λ_res / (M_R × A_h × ln(r_front / r_h))
        # ─────────────────────────────────────────────────────────────────────
        r_front      = np.sqrt(4.0 * p.alpha_res_m2_s * t_safe)
        # Ensure r_front ≥ r_h (thermal front must be at least at heated zone edge)
        r_front      = max(r_front, r_h * 1.005)
        ln_rf_rh     = max(np.log(r_front / r_h), 0.05)  # floor avoids 1/0

        Gamma_rad = (
            2.0 * np.pi * p.lambda_res_W_mK
            / (M_R_J * A_h * ln_rf_rh)
        )

        # ─────────────────────────────────────────────────────────────────────
        # Γ_FLUID: Produced fluid heat removal rate coefficient
        # q_liq_m3_s = liquid rate [m³/s] = q_liq[m³/day] / 86400
        # Γ_fluid = q_liq × M_fluid / (M_R × A_h × h)
        # ─────────────────────────────────────────────────────────────────────
        q_liq_m3_s = p.q_liquid_m3_d / SEC_PER_DAY
        Gamma_fluid = (
            q_liq_m3_s * p.M_fluid_J_m3_K
            / (M_R_J * A_h * h)
        )

        # ─────────────────────────────────────────────────────────────────────
        # Boberg-Lantz ODE:  dθ/dt = −θ × (Γ_vert + Γ_rad + Γ_fluid)
        # ─────────────────────────────────────────────────────────────────────
        dtheta_dt = -theta * (Gamma_vert + Gamma_rad + Gamma_fluid)

        # ─────────────────────────────────────────────────────────────────────
        # Oil production rate ODE:  dV_oil/dt = q_oil_surface [m³/s]
        # The oil rate is a function of T̄(t) via Beggs-Robinson viscosity.
        # ─────────────────────────────────────────────────────────────────────
        q_oil_m3_d = self._darcy_oil_rate_m3_d(T_bar_c)    # m³/day (surface)
        dV_oil_dt  = q_oil_m3_d / SEC_PER_DAY               # m³/s

        return np.array([dtheta_dt, dV_oil_dt])

    # =========================================================================
    # §3.6  Production Phase Simulation — ODE Integration
    # =========================================================================

    def simulate_production_phase(
        self,
        t_prod_max_days: float = 60.0,
        n_eval_points:   int   = 600,
    ) -> pd.DataFrame:
        """
        Numerically integrate the Boberg-Lantz ODE over the CSS production
        phase using the scipy RK45 adaptive-step solver.

        The simulation produces a high-resolution time series of all key
        thermodynamic, rheological, and economic state variables.

        Simulation State Variables
        ─────────────────────────────────────────────────────────────────
        Primary (ODE states):
          θ(t)       : Dimensionless temperature excess
          V_oil(t)   : Cumulative surface oil production [m³]

        Derived (computed post-solution):
          T̄(t)       : Average heated-zone temperature [°C]
          μ_oil(t)   : Dead oil viscosity at T̄ [cP]
          q_oil(t)   : Instantaneous surface oil rate [m³/d]
          SOR(t)     : Cumulative Steam-Oil Ratio [bbl CWE / bbl oil]
          ΔSOR/Δt    : SOR deterioration rate [bbl/bbl/day]
          P_penalty  : Viscous power penalty factor μ(T̄)/μ(T̄_0) [-]
          Γ_vert     : Vertical decay coefficient [μs⁻¹] (diagnostic)
          Γ_rad      : Radial decay coefficient [μs⁻¹] (diagnostic)
          Γ_fluid    : Fluid decay coefficient [μs⁻¹] (diagnostic)

        Parameters
        ----------
        t_prod_max_days : Maximum production phase duration to simulate [days].
        n_eval_points   : Number of output time points (resolution of result DF).

        Returns
        -------
        pd.DataFrame indexed by 'time_days' with columns listed above.

        Raises
        ------
        RuntimeError : If the scipy ODE solver fails to converge.
        """
        p = self.p

        # ── Step 1: Compute heated zone geometry from injection + soak ────────
        hz      = self.compute_heated_zone()
        A_h     = hz["A_h_m2"]
        r_h     = hz["r_h_m"]
        T_bar_0 = hz["T_bar_0_c"]
        theta_0 = hz["theta_0"]

        logger.info(
            "Starting BL ODE integration | T̄_0=%.1f°C | θ₀=%.4f | "
            "A_h=%.1f m² | r_h=%.1f m | t_max=%.0f days | n=%d",
            T_bar_0, theta_0, A_h, r_h, t_prod_max_days, n_eval_points,
        )

        # ── Step 2: Time domain setup ─────────────────────────────────────────
        t_start  = 1.0                                    # 1 s (avoid t=0 singularity)
        t_end    = t_prod_max_days * SEC_PER_DAY          # s
        t_eval   = np.linspace(t_start, t_end, n_eval_points)

        # ── Step 3: Initial conditions ────────────────────────────────────────
        # y = [θ₀, V_oil_0=0 m³]
        y0 = np.array([theta_0, 0.0])

        # ── Step 4: Solve the BL ODE ──────────────────────────────────────────
        # RK45 with tight tolerances for accurate SOR bookkeeping
        sol = solve_ivp(
            fun          = self._boberg_lantz_ode,
            t_span       = (t_start, t_end),
            y0           = y0,
            method       = "RK45",
            t_eval       = t_eval,
            args         = (A_h, r_h, T_bar_0),
            rtol         = 1.0e-7,
            atol         = 1.0e-10,
            dense_output = False,
        )

        if not sol.success:
            logger.error("Boberg-Lantz ODE solver failed: %s", sol.message)
            raise RuntimeError(
                f"Boberg-Lantz ODE integration failed for well {p.well_id}: "
                f"{sol.message}"
            )

        # ── Step 5: Unpack ODE solution ───────────────────────────────────────
        t_days   = sol.t / SEC_PER_DAY
        theta    = np.clip(sol.y[0], 0.0, 1.0)      # clamp to [0, 1]
        V_oil_m3 = np.maximum(sol.y[1], 0.0)         # non-negative production

        # ── Step 6: Compute derived channels ─────────────────────────────────
        delta_T0 = T_bar_0 - p.T_reservoir_c          # °C — initial temp excess

        # Average heated-zone temperature
        T_bar_c = p.T_reservoir_c + theta * delta_T0  # °C

        # Beggs-Robinson dead oil viscosity at each T̄
        mu_cP = self._beggs_robinson_viscosity_cP(T_bar_c)

        # Surface oil production rate [m³/day]
        q_oil_m3_d = self._darcy_oil_rate_m3_d(T_bar_c)

        # Steam CWE volume (fixed at injection; all steam injected before production)
        sp          = self.steam_injection_properties()
        V_steam_bbl = sp["V_CWE_bbl"]                  # bbl CWE — constant

        # Cumulative oil in barrels
        V_oil_bbl = V_oil_m3 * M3_TO_BBL

        # ── Cumulative SOR [bbl steam / bbl oil] ──────────────────────────────
        # Guard against zero oil production in first few time-steps
        SOR_cumulative = np.where(
            V_oil_bbl > 0.05,               # threshold: ≥ 0.05 bbl produced
            V_steam_bbl / V_oil_bbl,
            np.full_like(V_oil_bbl, np.inf),
        )

        # ── SOR deterioration rate [bbl/bbl per day] ─────────────────────────
        # Measures HOW FAST the SOR is worsening.
        # We compute the numerical gradient only on FINITE SOR values to prevent
        # false triggers from inf-filled values at the very start of production.
        SOR_for_gradient = np.where(np.isinf(SOR_cumulative), np.nan, SOR_cumulative)
        # Forward-fill NaN with the first finite value so gradient is well-defined
        first_finite = np.nanmin(SOR_for_gradient) if not np.all(np.isnan(SOR_for_gradient)) else 0.0
        SOR_for_gradient = np.where(np.isnan(SOR_for_gradient), first_finite, SOR_for_gradient)
        SOR_rate_per_d = np.abs(np.gradient(SOR_for_gradient, t_days))
        # Zero-out the rate for points where SOR is still inf (no real signal yet)
        SOR_rate_per_d = np.where(np.isinf(SOR_cumulative), 0.0, SOR_rate_per_d)

        # ── Viscous power penalty factor ──────────────────────────────────────
        # The SRP must overcome viscous drag proportional to μ × q.
        # Penalty = μ(T̄(t)) / μ(T̄_0)  — how many times harder the pump works
        # compared to the initial warm-oil conditions at the start of production.
        mu_initial      = float(self._beggs_robinson_viscosity_cP(T_bar_0))
        viscous_penalty = mu_cP / mu_initial

        # ── Boberg-Lantz decay coefficients (diagnostic channels) ─────────────
        # Re-compute vectorially for all t_eval points
        t_s_vec    = np.maximum(sol.t, 1.0)           # [s] — safe for sqrt

        Gamma_vert = (
            2.0 * p.lambda_cap_W_mK
            / ((p.M_R_kJ_m3_K * 1e3) * p.h_net_m
               * np.sqrt(np.pi * p.alpha_cap_m2_s * t_s_vec))
        )  # [1/s]

        r_front_vec  = np.sqrt(4.0 * p.alpha_res_m2_s * t_s_vec)
        r_front_vec  = np.maximum(r_front_vec, r_h * 1.005)
        ln_rf_vec    = np.maximum(np.log(r_front_vec / r_h), 0.05)
        Gamma_rad    = (
            2.0 * np.pi * p.lambda_res_W_mK
            / ((p.M_R_kJ_m3_K * 1e3) * A_h * ln_rf_vec)
        )  # [1/s]

        Gamma_fluid  = np.full_like(t_s_vec,
            (p.q_liquid_m3_d / SEC_PER_DAY) * p.M_fluid_J_m3_K
            / ((p.M_R_kJ_m3_K * 1e3) * A_h * p.h_net_m)
        )  # [1/s]

        # ── Thermal energy in produced oil above T_reservoir ──────────────────
        # Approximation: oil cools from T̄ to T_r on the way to surface.
        # Sensible heat extracted per m³: ρ × c_p × (T̄ − T_r)
        q_thermal_kW = (
            q_oil_m3_d / SEC_PER_DAY
            * p.rho_oil_kg_m3 * p.cp_oil_J_kgK
            * np.maximum(T_bar_c - p.T_reservoir_c, 0.0)
        ) * 1e-3  # kW

        # ── Assemble output DataFrame ─────────────────────────────────────────
        df = pd.DataFrame({
            "time_days"            : t_days,
            "theta"                : theta,
            "T_bar_c"              : T_bar_c,
            "mu_cP"                : mu_cP,
            "q_oil_m3_d"           : q_oil_m3_d,
            "V_oil_m3_cumulative"  : V_oil_m3,
            "V_oil_bbl_cumulative" : V_oil_bbl,
            "SOR_cumulative"       : SOR_cumulative,
            "SOR_rate_bbl_bbl_d"   : SOR_rate_per_d,
            "viscous_power_penalty": viscous_penalty,
            "q_thermal_kW"         : q_thermal_kW,
            "Gamma_vert_us"        : Gamma_vert * 1e6,   # μs⁻¹ (micro-per-second)
            "Gamma_rad_us"         : Gamma_rad  * 1e6,
            "Gamma_fluid_us"       : Gamma_fluid * 1e6,
        }).set_index("time_days")

        logger.info(
            "BL simulation done | T̄: %.1f→%.1f°C | μ: %.0f→%.0f cP | "
            "V_oil=%.1f bbl | SOR_final=%.2f | viscous_penalty_final=%.1fx",
            float(T_bar_c[0]),    float(T_bar_c[-1]),
            float(mu_cP[0]),      float(mu_cP[-1]),
            float(V_oil_bbl[-1]),
            float(SOR_cumulative[-1]) if not np.isinf(SOR_cumulative[-1]) else 999.0,
            float(viscous_penalty[-1]),
        )
        return df

    # =========================================================================
    # §3.7  Autonomous CSS Production Cutoff Evaluator
    # =========================================================================

    def evaluate_css_cutoff(
        self,
        sim_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyse the production phase simulation and autonomously recommend
        the optimal CSS production cutoff, minimising Scope 1 GHG emissions
        while maximising economic recovery.

        Five Cutoff Criteria
        ─────────────────────────────────────────────────────────────────────
        Each criterion is evaluated independently at every simulation time
        point. The FIRST (earliest) trigger determines the recommended cutoff.

        ┌───┬─────────────────────────┬──────────────────────────────────────┐
        │ # │ Criterion               │ Physical / Economic Rationale         │
        ├───┼─────────────────────────┼──────────────────────────────────────┤
        │ 1 │ T̄ < T_cutoff           │ HIGHEST PRIORITY. Below this T̄, crude│
        │   │ (Temperature floor)     │ viscosity exceeds the SRP design load.│
        │   │                         │ The pump fights viscous drag, burning  │
        │   │                         │ excess electricity → Scope 1 CO₂     │
        ├───┼─────────────────────────┼──────────────────────────────────────┤
        │ 2 │ viscous_penalty >       │ Even before T̄ hits the floor, the    │
        │   │ srp_overload_factor     │ power penalty multiple reveals that   │
        │   │ (SRP overload)          │ ≥3.5× baseline power is wasted on    │
        │   │                         │ viscosity → rod fatigue + GHG waste   │
        ├───┼─────────────────────────┼──────────────────────────────────────┤
        │ 3 │ SOR_cumulative >        │ Each barrel of oil has consumed more  │
        │   │ SOR_economic            │ steam (CWE) than the economic limit.  │
        │   │ (Economic SOR)          │ Revenue cannot cover steam generation │
        │   │                         │ cost (diesel/gas boiler emissions).   │
        ├───┼─────────────────────────┼──────────────────────────────────────┤
        │ 4 │ SOR_rate >              │ Marginal SOR is deteriorating faster  │
        │   │ SOR_rate_warning        │ than acceptable → near the economic   │
        │   │ (SOR deterioration)     │ cliff. Early warning to plan shut-in. │
        ├───┼─────────────────────────┼──────────────────────────────────────┤
        │ 5 │ η_energy < eta_min      │ Thermodynamic efficiency floor. Even  │
        │   │ (Energy efficiency)     │ if SOR looks acceptable, the cycle is │
        │   │                         │ returning < minimum thermal fraction.  │
        └───┴─────────────────────────┴──────────────────────────────────────┘

        Scope 1 GHG Linkage
        ─────────────────────
        The SRP pumping power scales with viscous drag:
            P_pump(t) ∝ μ(T̄(t)) × q_liquid
        The GHG emission rate from the diesel prime mover or grid power:
            GHG_rate(t) ∝ P_pump(t) × EF_CO2   [kgCO₂/kWh]
        The viscous_power_penalty column directly quantifies this multiplication
        factor. Recommending cutoff before the penalty exceeds the threshold
        avoids exponential CO₂ emission blow-up as crude approaches gel point.

        Parameters
        ----------
        sim_df : Output DataFrame from simulate_production_phase().

        Returns
        -------
        dict with keys:
            recommended_cutoff_day    float — optimal cutoff day (production days)
            cutoff_trigger            str   — name of the criterion that fired first
            cutoff_priority           int   — priority number (1=highest)
            all_trigger_days          dict  — first trigger day for ALL 5 criteria
            V_oil_at_cutoff_m3        float — cumulative oil at cutoff [m³]
            V_oil_at_cutoff_bbl       float — cumulative oil at cutoff [bbl]
            SOR_at_cutoff             float — cumulative SOR at cutoff [bbl/bbl]
            T_bar_at_cutoff_c         float — average heated zone T at cutoff [°C]
            mu_at_cutoff_cP           float — crude viscosity at cutoff [cP]
            viscous_penalty_at_cutoff float — viscous power factor at cutoff [-]
            eta_energy_at_cutoff      float — energy efficiency ratio at cutoff [-]
            ghg_excess_avoided_frac   float — fraction of excess pump GHG avoided [%]
            css_cycle_summary         dict  — high-level cycle economics
            recommendation            str   — human-readable action message
            all_triggers_df           pd.DataFrame — full timeline of all criteria
        """
        p  = self.p
        df = sim_df.copy()
        t  = df.index.values                          # time_days array

        # ── Compute thermal energy return ratio η_energy ─────────────────────
        sp          = self.steam_injection_properties()
        Q_steam_J   = sp["Q_total_GJ"] * 1e9          # J — total steam heat

        # Thermal energy extracted by produced oil above T_reservoir:
        #   Q_oil(t) ≈ V_oil × ρ_oil × c_p_oil × ΔT_produced
        # ΔT_produced is approximated as half the heated-zone excess temperature
        # (assumes linear cooling from T̄ at wellhead to T_r at surface separator)
        dT_prod   = np.maximum(df["T_bar_c"].values - p.T_reservoir_c, 0.0) * 0.5
        Q_oil_J   = (df["V_oil_m3_cumulative"].values
                     * p.rho_oil_kg_m3 * p.cp_oil_J_kgK * dT_prod)
        eta_energy = np.where(Q_steam_J > 0.0, Q_oil_J / Q_steam_J, 0.0)
        df["eta_energy"] = eta_energy

        # ── Evaluate each criterion as a boolean mask ─────────────────────────
        # Criteria 4 and 5 (ratio-based) only fire after >= 100 bbl produced
        # to prevent false t=0 triggers from division-by-zero artifacts.
        has_min_oil = df["V_oil_bbl_cumulative"].values >= 100.0

        mask_T    = df["T_bar_c"].values < p.T_cutoff_c
        mask_visc = df["viscous_power_penalty"].values > p.srp_overload_factor
        mask_SOR  = (df["SOR_cumulative"].values < 900.0) & (
                     df["SOR_cumulative"].values > p.SOR_economic)
        mask_rate = has_min_oil & (df["SOR_rate_bbl_bbl_d"].values > p.SOR_rate_warning)
        mask_eta  = has_min_oil & (eta_energy < p.eta_energy_min)

        # ── Find first trigger day for each criterion ─────────────────────────
        def _first_day(mask: np.ndarray) -> Optional[float]:
            """Return the time (days) of the first True entry, or None."""
            idxs = np.where(mask)[0]
            return float(t[idxs[0]]) if len(idxs) > 0 else None

        all_trigger_days: Dict[str, Optional[float]] = {
            "T_bar_below_cutoff"   : _first_day(mask_T),
            "viscous_srp_overload" : _first_day(mask_visc),
            "SOR_cumulative"       : _first_day(mask_SOR),
            "SOR_deterioration_rate": _first_day(mask_rate),
            "energy_efficiency"    : _first_day(mask_eta),
        }

        # Priority order (lower number = higher importance)
        priority_map: Dict[str, int] = {
            "T_bar_below_cutoff"    : 1,
            "viscous_srp_overload"  : 2,
            "SOR_cumulative"        : 3,
            "SOR_deterioration_rate": 4,
            "energy_efficiency"     : 5,
        }

        # Select the earliest-firing criterion (ties broken by priority)
        valid_triggers = {
            k: (day, priority_map[k])
            for k, day in all_trigger_days.items()
            if day is not None
        }

        if valid_triggers:
            recommended_key, (recommended_day, cutoff_priority) = min(
                valid_triggers.items(),
                key=lambda kv: (kv[1][0], kv[1][1]),
            )
        else:
            # No criterion triggered — recommend end of simulation window
            recommended_key   = "no_trigger_in_window"
            recommended_day   = float(t[-1])
            cutoff_priority   = 0

        # ── Extract all state variables at the cutoff day ─────────────────────
        co_idx    = int(np.searchsorted(t, recommended_day, side="left"))
        co_idx    = min(co_idx, len(t) - 1)

        V_oil_co_m3  = float(df["V_oil_m3_cumulative"].iloc[co_idx])
        V_oil_co_bbl = float(df["V_oil_bbl_cumulative"].iloc[co_idx])
        SOR_co       = float(df["SOR_cumulative"].iloc[co_idx])
        T_bar_co     = float(df["T_bar_c"].iloc[co_idx])
        mu_co        = float(df["mu_cP"].iloc[co_idx])
        penalty_co   = float(df["viscous_power_penalty"].iloc[co_idx])
        eta_co       = float(eta_energy[co_idx])

        # ── GHG excess emissions avoided by timely cutoff ─────────────────────
        # The "excess" power consumed above baseline (penalty > 1) maps to
        # excess CO₂. We compute the fraction of total excess power that
        # would have occurred AFTER the recommended cutoff.
        all_excess    = np.maximum(df["viscous_power_penalty"].values - 1.0, 0.0)
        excess_after  = all_excess[co_idx + 1:]
        total_excess  = float(np.sum(all_excess))
        avoided_excess = float(np.sum(excess_after))
        ghg_avoided   = (avoided_excess / total_excess) if total_excess > 1e-6 else 0.0
        ghg_avoided   = float(np.clip(ghg_avoided, 0.0, 1.0))

        # ── CSS cycle economic summary ────────────────────────────────────────
        total_oil_bbl   = V_oil_co_bbl
        total_oil_m3    = V_oil_co_m3
        steam_bbl       = sp["V_CWE_bbl"]
        cycle_SOR       = steam_bbl / max(total_oil_bbl, 0.01)
        # Approximate recovery factor (fraction of oil-in-place recovered):
        V_h_m3   = float(df.attrs.get("V_h_m3", 0.0))   # heated zone volume
        if V_h_m3 > 0:
            OOIP_m3  = V_h_m3 * p.phi * p.S_o
            RF_frac  = total_oil_m3 / max(OOIP_m3, 1.0)
        else:
            OOIP_m3, RF_frac = 0.0, 0.0

        css_cycle_summary = {
            "injection_days"       : p.t_injection_days,
            "soak_days"            : p.t_soak_days,
            "production_days"      : recommended_day,
            "total_cycle_days"     : p.t_injection_days + p.t_soak_days + recommended_day,
            "steam_CWE_bbl"        : steam_bbl,
            "oil_produced_bbl"     : total_oil_bbl,
            "cycle_SOR"            : cycle_SOR,
            "T_initial_c"          : float(df["T_bar_c"].iloc[0]),
            "T_at_cutoff_c"        : T_bar_co,
            "mu_initial_cP"        : float(df["mu_cP"].iloc[0]),
            "mu_at_cutoff_cP"      : mu_co,
            "ghg_excess_avoided_pct": ghg_avoided * 100.0,
        }

        # ── Human-readable recommendation message ─────────────────────────────
        _messages: Dict[str, str] = {
            "T_bar_below_cutoff": (
                f"[PRIORITY 1 — TEMPERATURE CUTOFF]\n"
                f"The average heated-zone temperature T̄ = {T_bar_co:.1f}°C has fallen "
                f"below the viscosity-driven cutoff of {p.T_cutoff_c}°C. "
                f"At this temperature, Baghewala crude viscosity reaches {mu_co:.0f} cP. "
                f"The SRP rod string is fighting {penalty_co:.1f}× its baseline viscous "
                f"drag, consuming disproportionate electrical energy and generating "
                f"excess Scope 1 CO₂ via the diesel steam generators.\n"
                f"ACTION: Shut-in well BGW-{p.well_id}. Plan next CSS injection cycle. "
                f"Estimated {ghg_avoided*100:.1f}% of cycle excess pump GHG is avoided "
                f"by cutting off on day {recommended_day:.1f} vs. running to depletion."
            ),
            "viscous_srp_overload": (
                f"[PRIORITY 2 — SRP VISCOUS OVERLOAD]\n"
                f"Viscous power penalty = {penalty_co:.2f}× baseline (threshold: "
                f"{p.srp_overload_factor:.1f}×). The SRP prime mover is consuming "
                f"{(penalty_co-1)*100:.0f}% MORE power than at initial warm-oil conditions "
                f"to lift {mu_co:.0f} cP crude. This translates directly to excess Scope 1 "
                f"CO₂ from the diesel generators and accelerated rod fatigue.\n"
                f"ACTION: Immediate shut-in on day {recommended_day:.1f}. Initiate "
                f"next CSS steam injection to restore reservoir mobility."
            ),
            "SOR_cumulative": (
                f"[PRIORITY 3 — ECONOMIC SOR BREACH]\n"
                f"Cumulative Steam-Oil Ratio = {SOR_co:.2f} bbl/bbl exceeds the "
                f"economic limit of {p.SOR_economic:.1f} bbl/bbl. "
                f"Total steam injected: {steam_bbl:.0f} bbl CWE. "
                f"Oil produced: {V_oil_co_bbl:.0f} bbl. "
                f"Continued production is unprofitable — steam cost exceeds oil revenue "
                f"at current commodity prices.\n"
                f"ACTION: Shut-in and evaluate next cycle injection parameters "
                f"(reduce t_inj or increase steam quality to improve SOR)."
            ),
            "SOR_deterioration_rate": (
                f"[PRIORITY 4 — RAPID SOR DETERIORATION]\n"
                f"SOR deterioration rate = {float(df['SOR_rate_bbl_bbl_d'].iloc[co_idx]):.3f} "
                f"bbl/bbl/day (threshold: {p.SOR_rate_warning:.2f}). "
                f"Current SOR = {SOR_co:.2f}. The SOR is worsening faster than the "
                f"economic tolerance — an exponential deterioration phase is beginning "
                f"as crude approaches near-solid viscosity.\n"
                f"ACTION: Proactive shut-in on day {recommended_day:.1f} to prevent "
                f"SOR from breaching hard economic limit."
            ),
            "energy_efficiency": (
                f"[PRIORITY 5 — THERMODYNAMIC EFFICIENCY FLOOR]\n"
                f"Thermal energy return ratio η = {eta_co:.4f} < minimum {p.eta_energy_min:.2f}. "
                f"The production phase is returning less than {p.eta_energy_min*100:.0f}% "
                f"of the injected steam energy as recoverable oil heat. "
                f"The cycle has passed its thermodynamically efficient window.\n"
                f"ACTION: Shut-in. Begin planning next CSS cycle to restore η above minimum."
            ),
            "no_trigger_in_window": (
                f"[NO CUTOFF TRIGGERED]\n"
                f"All five cutoff criteria satisfied throughout the {t[-1]:.0f}-day "
                f"simulation window. Production remains economic and thermodynamically "
                f"efficient. Extend the simulation window or continue monitoring."
            ),
        }

        recommendation = _messages.get(
            recommended_key,
            f"[UNKNOWN TRIGGER: {recommended_key}] — Evaluate manually.",
        )

        # ── Build full trigger timeline DataFrame ─────────────────────────────
        triggers_df = pd.DataFrame({
            "T_bar_c"                  : df["T_bar_c"].values,
            "mu_cP"                    : df["mu_cP"].values,
            "SOR_cumulative"           : df["SOR_cumulative"].values,
            "viscous_power_penalty"    : df["viscous_power_penalty"].values,
            "eta_energy"               : eta_energy,
            "trigger_T_cutoff"         : mask_T,
            "trigger_visc_overload"    : mask_visc,
            "trigger_SOR_cumulative"   : mask_SOR,
            "trigger_SOR_rate"         : mask_rate,
            "trigger_eta_efficiency"   : mask_eta,
        }, index=df.index)

        logger.info(
            "CUTOFF RECOMMENDATION | trigger='%s' (P%d) | day=%.1f | "
            "oil=%.0f bbl | SOR=%.2f | T̄=%.1f°C | μ=%.0f cP | GHG_avoided=%.1f%%",
            recommended_key, cutoff_priority, recommended_day,
            V_oil_co_bbl, SOR_co if not np.isinf(SOR_co) else 999.9,
            T_bar_co, mu_co, ghg_avoided * 100.0,
        )

        return {
            "recommended_cutoff_day"   : recommended_day,
            "cutoff_trigger"           : recommended_key,
            "cutoff_priority"          : cutoff_priority,
            "all_trigger_days"         : all_trigger_days,
            "V_oil_at_cutoff_m3"       : V_oil_co_m3,
            "V_oil_at_cutoff_bbl"      : V_oil_co_bbl,
            "SOR_at_cutoff"            : SOR_co,
            "T_bar_at_cutoff_c"        : T_bar_co,
            "mu_at_cutoff_cP"          : mu_co,
            "viscous_penalty_at_cutoff": penalty_co,
            "eta_energy_at_cutoff"     : eta_co,
            "ghg_excess_avoided_frac"  : ghg_avoided,
            "css_cycle_summary"        : css_cycle_summary,
            "recommendation"           : recommendation,
            "all_triggers_df"          : triggers_df,
        }

    # =========================================================================
    # §3.8  Full CSS Cycle Report (Convenience Wrapper)
    # =========================================================================

    def generate_cycle_report(
        self,
        t_prod_max_days: float = 60.0,
        n_points:        int   = 600,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute the complete end-to-end CSS cycle analysis in a single call.

        Runs:
          1. steam_injection_properties()  — injection thermodynamics
          2. compute_heated_zone()         — Marx-Langenheim geometry
          3. simulate_production_phase()   — Boberg-Lantz ODE integration
          4. evaluate_css_cutoff()         — multi-criterion cutoff logic

        Parameters
        ----------
        t_prod_max_days : Maximum production phase window [days].
        n_points        : ODE output resolution (number of time points).

        Returns
        -------
        (sim_df, cutoff_report) :
            sim_df        pd.DataFrame from simulate_production_phase()
            cutoff_report dict from evaluate_css_cutoff()
        """
        sim_df        = self.simulate_production_phase(t_prod_max_days, n_points)
        cutoff_report = self.evaluate_css_cutoff(sim_df)
        return sim_df, cutoff_report


# ===========================================================================
# §4  Module Smoke Test  (python -m src.reservoir_model)
# ===========================================================================

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  ReservoirThermodynamics (Boberg-Lantz) -- Smoke Test")
    print("  Baghewala Well-to-Surface Digital Twin | src/reservoir_model.py")
    print("=" * 72)

    # ── Initialise with Baghewala field parameters ────────────────────────────
    params = BaghewalaCSSParams(
        well_id           = "BGW-07",
        T_steam_c         = 320.0,
        steam_quality     = 0.75,
        m_dot_steam_kg_d  = 80_000.0,
        t_injection_days  = 15.0,
        t_soak_days       = 7.0,
        T_cutoff_c        = 72.0,
        SOR_economic      = 5.0,
        srp_overload_factor = 3.5,
    )
    engine = ReservoirThermodynamics(params)

    # ── [1] Steam injection thermodynamics ────────────────────────────────────
    print("\n[1] Steam Injection Properties (IAPWS-IF97 Tables)")
    sp = engine.steam_injection_properties()
    for k, v in sp.items():
        print(f"    {k:<28}: {v:.3f}")

    # ── [2] Marx-Langenheim heated zone ───────────────────────────────────────
    print("\n[2] Heated Zone Geometry (Marx-Langenheim Model)")
    hz = engine.compute_heated_zone()
    for k, v in hz.items():
        print(f"    {k:<20}: {v:.4f}")

    # ── [3] Viscosity check at key temperatures ───────────────────────────────
    print("\n[3] Beggs-Robinson Viscosity Profile")
    print(f"    {'T (deg C)':<14} {'mu (cP)'}")
    print("    " + "-" * 30)
    for T in [47.0, 72.0, 100.0, 150.0, 200.0, 320.0]:
        mu = engine._beggs_robinson_viscosity_cP(T)
        print(f"    {T:<14.1f} {mu:.1f}")

    # ── [4] Production phase ODE integration ──────────────────────────────────
    print("\n[4] Running Boberg-Lantz ODE (60-day production window) ...")
    sim_df, cutoff = engine.generate_cycle_report(
        t_prod_max_days = 60.0,
        n_points        = 400,
    )

    print(f"\n    Simulation shape    : {sim_df.shape}")
    print(f"    T_bar range         : {sim_df['T_bar_c'].min():.1f} -- "
          f"{sim_df['T_bar_c'].max():.1f} deg C")
    print(f"    Viscosity range     : {sim_df['mu_cP'].min():.0f} -- "
          f"{sim_df['mu_cP'].max():.0f} cP")
    print(f"    Cumulative oil      : {sim_df['V_oil_bbl_cumulative'].iloc[-1]:.1f} bbl")
    print(f"    Final SOR           : {sim_df['SOR_cumulative'].iloc[-1]:.2f} bbl/bbl")
    print(f"    Peak viscous penalty: {sim_df['viscous_power_penalty'].max():.1f}x")

    print("\n    First 3 rows:")
    cols = ["T_bar_c", "mu_cP", "q_oil_m3_d", "SOR_cumulative", "viscous_power_penalty"]
    print(sim_df[cols].head(3).to_string())

    # ── [5] Cutoff recommendation ─────────────────────────────────────────────
    print("\n[5] Autonomous CSS Cutoff Recommendation")
    print("-" * 72)
    print(f"  Trigger      : {cutoff['cutoff_trigger']}")
    print(f"  Priority     : {cutoff['cutoff_priority']}")
    print(f"  Cutoff Day   : Day {cutoff['recommended_cutoff_day']:.1f} of production")
    print(f"  T_bar        : {cutoff['T_bar_at_cutoff_c']:.1f} deg C")
    print(f"  Viscosity    : {cutoff['mu_at_cutoff_cP']:.0f} cP")
    print(f"  Visc Penalty : {cutoff['viscous_penalty_at_cutoff']:.2f}x")
    print(f"  Oil produced : {cutoff['V_oil_at_cutoff_bbl']:.1f} bbl")
    print(f"  SOR          : {cutoff['SOR_at_cutoff']:.2f} bbl/bbl")
    print(f"  GHG avoided  : {cutoff['ghg_excess_avoided_frac']*100:.1f}%")
    print("-" * 72)
    print(f"\n  RECOMMENDATION:\n  {cutoff['recommendation']}")

    # ── [6] All trigger days ──────────────────────────────────────────────────
    print("\n[6] All Trigger Days:")
    for k, v in cutoff["all_trigger_days"].items():
        day_str = f"Day {v:.1f}" if v is not None else "Not triggered"
        print(f"    {k:<30}: {day_str}")

    print("\n[PASS] Smoke test passed.\n")
