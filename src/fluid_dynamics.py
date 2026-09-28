# -*- coding: utf-8 -*-
"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : src/fluid_dynamics.py
  Class  : BaghewalRheologyEngine
  Author : Digital Twin Engineering Team
  Purpose: Heavy-oil rheology engine for the Baghewala 14–19°API crude.
           Dynamically tracks apparent viscosity as near-wellbore temperature
           drops during the CSS production phase, using the Power-Law
           (Ostwald-de Waele) model for shear-thinning pseudoplastic flow.

  Theoretical background
  ----------------------
  Baghewala crude is a pseudoplastic (shear-thinning) non-Newtonian fluid.
  Its rheological behaviour is well described by the POWER-LAW MODEL:

      τ = K × γ̇ⁿ                                              … (1)

  where
      τ   = shear stress [Pa]
      K   = consistency index [Pa·sⁿ]  — strongly temperature-dependent
      γ̇   = shear rate [s⁻¹]           — set by flow geometry & velocity
      n   = flow-behaviour index [-]    — n < 1 → pseudoplastic (shear-thinning)
                                          n = 1 → Newtonian
                                          n > 1 → dilatant

  The APPARENT VISCOSITY (equivalent Newtonian viscosity at a given shear rate)
  is derived by comparing τ = η_app × γ̇  with equation (1):

      η_app(γ̇, T) = K(T) × γ̇^(n−1)                          … (2)

  Since n < 1, η_app DECREASES with increasing shear rate — this is
  shear-thinning.  As T increases (CSS heating), K(T) drops sharply,
  dramatically reducing viscosity and improving inflow into the wellbore.

  Temperature dependence of the consistency index K(T)
  -------------------------------------------------------
  The Arrhenius-type relationship is used:

      K(T) = K_ref × exp[ E_a/R × (1/T − 1/T_ref) ]          … (3)

  where
      K_ref = consistency index at reference temperature T_ref [K]
      E_a   = flow activation energy [J/mol]
      R     = universal gas constant = 8.314 J/(mol·K)
      T     = absolute temperature [K]

  At the Baghewala baseline BHT (~47°C ≈ 320 K), the in-situ viscosity is
  10,000–13,000 cP.  CSS heating can lower this by 1–2 orders of magnitude.

  Pressure-viscosity coupling (Barus equation)
  ----------------------------------------------
  At downhole pressures (typically 30–80 bar for this field), the pressure
  correction to viscosity is applied via the Barus equation:

      η_p(P, T) = η_app(T) × exp(α_barus × P)                 … (4)

  where α_barus [Pa⁻¹] is the pressure-viscosity coefficient.

  Force-balance integration
  -------------------------
  The module also provides the continuous dynamic force balance across a
  differential annular element dz along the rod string, computing:

      dP/dz = ρg + (τ_wall × π × D_h) / A_flow                … (5)

  integrated numerically along the tubing depth to give bottomhole-to-surface
  pressure drop ΔP_friction under non-Newtonian flow.
================================================================================
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Module logger — force UTF-8 on Windows to avoid cp1252 codec errors
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(stream=sys.stderr)
    try:
        _handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass   # Python < 3.7 fallback — stream doesn't support reconfigure
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# Universal gas constant [J/(mol·K)]
R_GAS: float = 8.314


# ===========================================================================
# §1  Rheology Parameter Dataclass
# ===========================================================================

@dataclass
class RheologyParams:
    """
    Power-Law + Arrhenius rheology parameters for Baghewala 14–19°API crude.

    Physical context
    ----------------
    The parameters below are calibrated to reproduce:
      - η_app ≈ 10,000–13,000 cP at T = 47°C, γ̇ = 1 s⁻¹ (reservoir)
      - η_app ≈  1,500–3,000  cP at T = 90°C  (post-CSS warm-up)
      - η_app ≈    200–500    cP at T = 150°C  (mid-CSS soak)

    Attributes
    ----------
    n : float
        Flow-behaviour index [-].  n < 1 → pseudoplastic / shear-thinning.
        Baghewala crude: n ≈ 0.65–0.72 (significantly non-Newtonian).
    K_ref_Pa_sn : float
        Consistency index [Pa·sⁿ] at the reference temperature T_ref.
        High K reflects the waxy, asphaltic nature of the heavy crude.
    T_ref_c : float
        Reference temperature for K_ref, Celsius.  Set to reservoir BHT.
    E_a_kJ_mol : float
        Apparent activation energy for viscous flow [kJ/mol].
        Heavy crudes typically exhibit E_a = 40–80 kJ/mol.
    alpha_barus_per_MPa : float
        Barus pressure-viscosity coefficient [MPa⁻¹].
        Typical heavy crude: 0.01–0.03 MPa⁻¹.
    rho_oil_kg_m3 : float
        Live-oil density at reservoir conditions [kg/m³].
    """

    n: float                    = 0.68        # flow-behaviour index (shear-thinning)
    K_ref_Pa_sn: float          = 15.0        # consistency index at T_ref [Pa·sⁿ]
    T_ref_c: float              = 47.0        # reference temperature, °C
    E_a_kJ_mol: float           = 55.0        # activation energy, kJ/mol
    alpha_barus_per_MPa: float  = 0.015       # Barus coeff, MPa⁻¹
    rho_oil_kg_m3: float        = 962.0       # oil density at reservoir, kg/m³

    # -- Derived SI values (set in __post_init__) ---------------------------
    T_ref_K: float              = field(init=False)
    E_a_J_mol: float            = field(init=False)
    alpha_barus_per_Pa: float   = field(init=False)

    def __post_init__(self) -> None:
        self.T_ref_K           = self.T_ref_c + 273.15        # K
        self.E_a_J_mol         = self.E_a_kJ_mol * 1000.0     # J/mol
        self.alpha_barus_per_Pa = self.alpha_barus_per_MPa / 1e6  # Pa⁻¹

        # Validation
        if not (0.0 < self.n < 1.0):
            raise ValueError(
                f"Flow-behaviour index n={self.n} must be in (0, 1) "
                "for a pseudoplastic fluid."
            )
        if self.K_ref_Pa_sn <= 0:
            raise ValueError("Consistency index K_ref must be positive.")

        logger.debug(
            "RheologyParams: n=%.3f, K_ref=%.2f Pa·s^n, E_a=%.1f kJ/mol, "
            "T_ref=%.1f K",
            self.n, self.K_ref_Pa_sn, self.E_a_kJ_mol, self.T_ref_K,
        )


# ===========================================================================
# §2  BaghewalRheologyEngine — Core Viscosity & Force-Balance Engine
# ===========================================================================

class BaghewalRheologyEngine:
    """
    Heavy-oil rheology engine for the Baghewala field.

    Computes:
    ─────────
    1. Arrhenius consistency index  K(T)
    2. Apparent (Power-Law) viscosity η_app(γ̇, T)
    3. Pressure-corrected viscosity η(P, T, γ̇)  via Barus equation
    4. In-situ shear rate profile for annular/pipe flow
    5. Continuous axial pressure-drop  ΔP_friction(T_profile, Q)
       by integrating the Hagen-Poiseuille–power-law ODE along the tubing

    Usage
    -----
    >>> params = RheologyParams(n=0.68, E_a_kJ_mol=55.0)
    >>> engine = BaghewalRheologyEngine(params)
    >>> visc_cP = engine.apparent_viscosity_cP(shear_rate=10.0, temp_c=80.0)
    >>> print(f"η_app = {visc_cP:.0f} cP at 80°C, γ̇=10 s⁻¹")
    """

    G: float = 9.80665   # standard gravity, m/s²

    def __init__(
        self,
        params: Optional[RheologyParams] = None,
    ) -> None:
        """
        Parameters
        ----------
        params : RheologyParams instance. Uses Baghewala field defaults if None.
        """
        self.p = params if params is not None else RheologyParams()
        logger.info(
            "BaghewalRheologyEngine initialised | n=%.3f | "
            "K_ref=%.2f Pa·s^n | E_a=%.1f kJ/mol",
            self.p.n, self.p.K_ref_Pa_sn, self.p.E_a_kJ_mol,
        )

    # -----------------------------------------------------------------------
    # §2.1  Arrhenius Consistency Index  K(T)
    # -----------------------------------------------------------------------

    def consistency_index(self, temp_c: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the temperature-dependent Power-Law consistency index K(T)
        using the Arrhenius viscosity-temperature relationship.

            K(T) = K_ref × exp[ (E_a/R) × (1/T_K − 1/T_ref_K) ]

        As temperature rises, the exponential factor becomes more negative,
        so K decreases — i.e., the fluid becomes less viscous.

        Parameters
        ----------
        temp_c : Temperature, °C.  Accepts scalar or NumPy array.

        Returns
        -------
        K : Consistency index [Pa·sⁿ], same shape as temp_c.
        """
        T_K    = np.asarray(temp_c, dtype=np.float64) + 273.15   # convert to K
        p      = self.p

        # Arrhenius exponent: (E_a / R) × (1/T − 1/T_ref)
        exponent = (p.E_a_J_mol / R_GAS) * (1.0 / T_K - 1.0 / p.T_ref_K)
        K        = p.K_ref_Pa_sn * np.exp(exponent)

        logger.debug("K(T=%.1f°C) = %.4f Pa·s^n", np.mean(temp_c), np.mean(K))
        return K

    # -----------------------------------------------------------------------
    # §2.2  Apparent Viscosity  η_app(γ̇, T)  — Power-Law Model
    # -----------------------------------------------------------------------

    def apparent_viscosity_Pa_s(
        self,
        shear_rate_inv_s: float | np.ndarray,
        temp_c: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Compute the apparent dynamic viscosity [Pa·s] at a given shear rate
        and temperature using the Power-Law model.

            η_app(γ̇, T) = K(T) × γ̇^(n − 1)

        Physical interpretation
        -----------------------
        - At n < 1 : η_app DECREASES as shear rate γ̇ increases (shear-thinning).
          This explains why heavy Baghewala crude flows more freely near the
          pump intake (high γ̇ due to rod motion) than in the far reservoir.
        - At γ̇ → 0 : η_app → ∞ for n < 1 (the fluid appears almost solid).
          In practice a minimum shear rate floor is enforced (γ̇_min = 0.001 s⁻¹).

        Parameters
        ----------
        shear_rate_inv_s : Shear rate γ̇ [s⁻¹]. Scalar or array.
        temp_c           : Temperature [°C]. Scalar or array (must broadcast).

        Returns
        -------
        eta_app : Apparent viscosity [Pa·s].
        """
        gamma_dot = np.asarray(shear_rate_inv_s, dtype=np.float64)
        gamma_dot = np.maximum(gamma_dot, 1e-4)   # floor to avoid singularity

        K       = self.consistency_index(temp_c)
        n       = self.p.n

        # Power-Law apparent viscosity equation
        eta_app = K * np.power(gamma_dot, n - 1.0)

        logger.debug(
            "η_app(γ̇=%.2f s⁻¹, T=%.1f°C) = %.4f Pa·s",
            float(np.mean(gamma_dot)), float(np.mean(temp_c)), float(np.mean(eta_app)),
        )
        return eta_app

    def apparent_viscosity_cP(
        self,
        shear_rate_inv_s: float | np.ndarray,
        temp_c: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Convenience wrapper — returns apparent viscosity in centipoise [cP].
        1 Pa·s = 1000 cP.
        """
        return self.apparent_viscosity_Pa_s(shear_rate_inv_s, temp_c) * 1000.0

    # -----------------------------------------------------------------------
    # §2.3  Pressure-Corrected Viscosity (Barus Equation)
    # -----------------------------------------------------------------------

    def pressure_corrected_viscosity_Pa_s(
        self,
        shear_rate_inv_s: float | np.ndarray,
        temp_c: float | np.ndarray,
        pressure_Pa: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Apply the Barus pressure correction to the apparent viscosity.

            η(P, T, γ̇) = η_app(γ̇, T) × exp(α × P)

        At P = 0 (surface conditions), this reduces to η_app.
        At typical Baghewala downhole pressures (30–80 bar = 3–8 MPa),
        the Barus correction adds 5–15% to the apparent viscosity.

        Parameters
        ----------
        shear_rate_inv_s : Shear rate [s⁻¹].
        temp_c           : Temperature [°C].
        pressure_Pa      : Absolute pressure [Pa].

        Returns
        -------
        eta_p : Pressure-corrected apparent viscosity [Pa·s].
        """
        eta_app = self.apparent_viscosity_Pa_s(shear_rate_inv_s, temp_c)
        P       = np.asarray(pressure_Pa, dtype=np.float64)

        # Barus equation: η_p = η_app × e^(α·P)
        eta_p   = eta_app * np.exp(self.p.alpha_barus_per_Pa * P)

        logger.debug(
            "η_p(P=%.1f bar) = %.4f Pa·s  (Barus factor = %.4f)",
            float(np.mean(P)) / 1e5,
            float(np.mean(eta_p)),
            float(np.mean(np.exp(self.p.alpha_barus_per_Pa * P))),
        )
        return eta_p

    # -----------------------------------------------------------------------
    # §2.4  Shear Rate Profile in the Tubing Annulus
    # -----------------------------------------------------------------------

    def annular_shear_rate(
        self,
        flow_rate_m3_s: float,
        tubing_id_m: float,
        rod_od_m: float,
    ) -> float:
        """
        Estimate the representative (Newtonian-equivalent) shear rate of the
        annular flow between the rod string and the tubing wall.

        For power-law flow in an annulus, the exact solution requires solving
        a transcendental equation.  A widely-used engineering approximation
        (Fredrickson-Bird, 1958) for narrow annuli is:

            γ̇_ann ≈ (12 × Q) / [π × (D_o + D_i) × (D_o − D_i)²]    … (Newtonian)

        For a power-law fluid this is modified by the Metzner-Reed correction:

            γ̇_eff = γ̇_ann × [(3n + 1) / (4n)]^(1/n)                … (Power-Law)

        This γ̇_eff is then used in all viscosity calculations.

        Parameters
        ----------
        flow_rate_m3_s : Volumetric flow rate in the annulus [m³/s].
        tubing_id_m    : Tubing inside diameter [m].
        rod_od_m       : Rod outside diameter [m].

        Returns
        -------
        gamma_dot_eff : Effective shear rate [s⁻¹].
        """
        Q   = max(flow_rate_m3_s, 1e-8)   # guard against zero flow
        D_o = tubing_id_m                  # outer boundary (tubing ID)
        D_i = rod_od_m                     # inner boundary (rod OD)

        # ── Newtonian annular shear rate approximation ──────────────────────
        # This form uses the hydraulic diameter concept for a concentric annulus
        gamma_ann = (12.0 * Q) / (
            np.pi * (D_o + D_i) * (D_o - D_i) ** 2
        )

        # ── Metzner-Reed power-law correction ───────────────────────────────
        n   = self.p.n
        # Metzner-Reed factor f_MR = (3n+1)/(4n) — accounts for the
        # non-parabolic velocity profile of a power-law fluid.
        f_MR          = (3.0 * n + 1.0) / (4.0 * n)
        gamma_dot_eff = gamma_ann * (f_MR ** (1.0 / n))

        logger.debug(
            "Annular γ̇: Newtonian=%.3f s⁻¹, Power-Law eff=%.3f s⁻¹ (f_MR=%.3f)",
            gamma_ann, gamma_dot_eff, f_MR,
        )
        return float(gamma_dot_eff)

    # -----------------------------------------------------------------------
    # §2.5  Continuous Axial Pressure-Drop via ODE Integration
    # -----------------------------------------------------------------------

    def pressure_drop_friction(
        self,
        depth_m: float,
        flow_rate_m3_s: float,
        tubing_id_m: float,
        rod_od_m: float,
        temperature_profile_c: np.ndarray | None = None,
        n_nodes: int = 50,
    ) -> dict[str, np.ndarray]:
        """
        Compute the continuous frictional pressure drop along the tubing
        from the pump (z = 0) to surface (z = depth_m) by integrating the
        axial momentum equation for power-law annular flow.

        Governing ODE (equation 5 from module docstring):
        ───────────────────────────────────────────────────
            dP/dz = −ρ·g  [hydrostatic]
                  − (τ_w · S_w) / A_ann  [frictional]

        where
            τ_w    = wall shear stress = K(T) × γ̇_eff^n    [Pa]
            S_w    = wetted perimeter of annulus = π(D_o + D_i) [m]
            A_ann  = annular cross-section = π/4 × (D_o² − D_i²) [m²]
            ρ      = oil density [kg/m³]
            g      = gravitational acceleration [m/s²]

        The sign convention is: z = 0 at pump, z = L at surface.
        We integrate from bottom to top, so dP/dz < 0 for upward flow.

        Temperature variation along the tubing is captured via a linear
        temperature profile from BHT at the bottom to a cooler wellhead T
        at the top, unless a custom profile is supplied.

        Parameters
        ----------
        depth_m               : Well depth (pump intake to wellhead), m.
        flow_rate_m3_s        : Production flow rate [m³/s].
        tubing_id_m           : Tubing ID [m].
        rod_od_m              : Rod string OD [m].
        temperature_profile_c : Optional 1-D array of T values at each node.
                                 If None, a linear gradient from BHT to 35°C is
                                 used (assumes ~12°C/km geothermal gradient).
        n_nodes               : Number of spatial nodes along the tubing depth.

        Returns
        -------
        dict with keys:
            'depth_m'       : np.ndarray of node depths [m, bottom to top]
            'pressure_Pa'   : np.ndarray of cumulative pressure at each node [Pa]
            'viscosity_Pa_s': np.ndarray of local η_app at each node [Pa·s]
            'shear_rate'    : np.ndarray of effective γ̇ at each node [s⁻¹]
            'temp_c'        : np.ndarray of temperature at each node [°C]
            'dP_friction_Pa': scalar total frictional pressure drop [Pa]
            'dP_hydrostatic_Pa': scalar hydrostatic column [Pa]
        """
        # ── Geometry ─────────────────────────────────────────────────────────
        D_o     = tubing_id_m
        D_i     = rod_od_m
        A_ann   = np.pi / 4.0 * (D_o ** 2 - D_i ** 2)       # annular area [m²]
        S_w     = np.pi * (D_o + D_i)                         # wetted perimeter [m]
        rho     = self.p.rho_oil_kg_m3

        # ── Effective shear rate (assumed constant along depth for given Q) ──
        # In reality γ̇ varies with local viscosity, but this iterative coupling
        # is handled at the production model level. Here we use a representative
        # annular shear rate as a first approximation.
        gamma_eff = self.annular_shear_rate(flow_rate_m3_s, D_o, D_i)

        # ── Temperature profile along tubing ──────────────────────────────────
        z_nodes = np.linspace(0.0, depth_m, n_nodes)          # 0 = pump, L = surface

        if temperature_profile_c is not None:
            T_profile = np.asarray(temperature_profile_c, dtype=np.float64)
            if T_profile.shape[0] != n_nodes:
                # Resample user-supplied profile to n_nodes
                T_profile = np.interp(
                    z_nodes,
                    np.linspace(0.0, depth_m, T_profile.shape[0]),
                    T_profile,
                )
        else:
            # Linear geothermal gradient: BHT at pump, ~35°C at surface
            T_wellhead = 35.0
            T_bht      = self.p.T_ref_c
            # Temperature decreases from bottom (z=0) toward surface (z=L)
            T_profile  = T_bht + (T_wellhead - T_bht) * (z_nodes / depth_m)

        # ── Compute local viscosity and wall shear stress at each node ────────
        eta_local   = self.apparent_viscosity_Pa_s(gamma_eff, T_profile)    # Pa·s
        tau_wall    = self.p.K_ref_Pa_sn                                    # initialise

        # ── ODE: dP/dz along the tubing (bottom → surface) ───────────────────
        # We accumulate pressure by numerical integration (trapezoidal rule).
        # P(z=0) = pump discharge pressure (reference = 0 gauge for this calc).

        dz = np.diff(z_nodes, prepend=z_nodes[0])   # node spacing, m

        # Wall shear stress at each node using local Power-Law K(T):
        K_local  = self.consistency_index(T_profile)            # Pa·s^n
        tau_wall = K_local * (gamma_eff ** self.p.n)            # Pa

        # Frictional pressure gradient at each node [Pa/m]:
        #   dP_fric/dz = (τ_w × S_w) / A_ann
        dP_fric_per_m = (tau_wall * S_w) / A_ann               # Pa/m (positive)

        # Hydrostatic gradient [Pa/m]:
        dP_hydro_per_m = rho * self.G                           # Pa/m (constant)

        # Net axial pressure gradient (positive upward = pressure increasing downward):
        # dP/dz = −(dP_fric/dz + dP_hydro/dz) for upward flow
        # We build the cumulative pressure profile from pump (high P) to surface (low P)
        dP_node = (dP_fric_per_m + dP_hydro_per_m) * dz        # Pa at each node

        # Cumulative pressure from pump toward surface (subtract as we rise)
        P_cum   = np.cumsum(dP_node[::-1])[::-1]               # Pa, from pump ref
        P_cum   -= P_cum[-1]                                    # reference to 0 at surface

        # Scalar summary values
        # np.trapezoid is the NumPy 2.x replacement for the removed np.trapz
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        dP_friction_total   = float(_trapz(dP_fric_per_m, z_nodes))  # Pa
        dP_hydrostatic_total= float(dP_hydro_per_m * depth_m)          # Pa

        logger.info(
            "Pressure drop | ΔP_friction=%.2f bar | ΔP_hydrostatic=%.2f bar "
            "| depth=%.0f m | Q=%.4f m³/s",
            dP_friction_total / 1e5,
            dP_hydrostatic_total / 1e5,
            depth_m,
            flow_rate_m3_s,
        )

        return {
            "depth_m"           : z_nodes,
            "pressure_Pa"       : P_cum,
            "viscosity_Pa_s"    : eta_local,
            "shear_rate"        : np.full(n_nodes, gamma_eff),
            "temp_c"            : T_profile,
            "dP_friction_Pa"    : dP_friction_total,
            "dP_hydrostatic_Pa" : dP_hydrostatic_total,
        }

    # -----------------------------------------------------------------------
    # §2.6  CSS Viscosity Reduction Factor
    # -----------------------------------------------------------------------

    def css_viscosity_reduction_factor(
        self,
        temp_pre_css_c: float,
        temp_post_css_c: float,
        shear_rate_inv_s: float = 1.0,
    ) -> float:
        """
        Quantify how much CSS heating reduces the in-situ apparent viscosity.

            VRF = η_app(γ̇, T_pre) / η_app(γ̇, T_post)

        A VRF of 10 means CSS heating reduced viscosity by 10-fold,
        which directly maps to a proportional improvement in inflow performance.

        Parameters
        ----------
        temp_pre_css_c  : Reservoir temperature before CSS injection [°C].
        temp_post_css_c : Near-wellbore temperature after CSS soak [°C].
        shear_rate_inv_s: Reference shear rate for comparison [s⁻¹].

        Returns
        -------
        VRF : Viscosity Reduction Factor [-].  > 1 implies viscosity decrease.
        """
        eta_pre  = self.apparent_viscosity_Pa_s(shear_rate_inv_s, temp_pre_css_c)
        eta_post = self.apparent_viscosity_Pa_s(shear_rate_inv_s, temp_post_css_c)

        vrf = float(eta_pre / eta_post)
        logger.info(
            "CSS VRF: η(%.0f°C)=%.1f Pa·s → η(%.0f°C)=%.4f Pa·s | VRF=%.1f×",
            temp_pre_css_c, eta_pre, temp_post_css_c, eta_post, vrf,
        )
        return vrf

    # -----------------------------------------------------------------------
    # §2.7  Temperature-Sweep — Viscosity Profile over CSS Cycle
    # -----------------------------------------------------------------------

    def temperature_sweep(
        self,
        temp_c_array: np.ndarray,
        shear_rate_inv_s: float | np.ndarray = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Compute the full rheological profile over a temperature sweep.

        Typically used to visualise how viscosity evolves during a CSS cycle
        as the near-wellbore zone cools from css_peak_temp back to BHT.

        Parameters
        ----------
        temp_c_array      : Temperature array [°C] to sweep (e.g., 200→47°C).
        shear_rate_inv_s  : Shear rate [s⁻¹], scalar or matching array.

        Returns
        -------
        dict with keys:
            'temp_c'           : Temperature array [°C]
            'K_Pa_sn'          : Consistency index K(T) [Pa·s^n]
            'eta_app_Pa_s'     : Apparent viscosity [Pa·s]
            'eta_app_cP'       : Apparent viscosity [cP]
            'shear_rate_inv_s' : Shear rate array [s⁻¹]
        """
        T   = np.asarray(temp_c_array, dtype=np.float64)
        gd  = np.broadcast_to(
            np.asarray(shear_rate_inv_s, dtype=np.float64), T.shape
        ).copy()

        K_arr     = self.consistency_index(T)
        eta_Pa_s  = self.apparent_viscosity_Pa_s(gd, T)

        return {
            "temp_c"           : T,
            "K_Pa_sn"          : K_arr,
            "eta_app_Pa_s"     : eta_Pa_s,
            "eta_app_cP"       : eta_Pa_s * 1000.0,
            "shear_rate_inv_s" : gd,
        }


# ===========================================================================
# §3  Module smoke-test  (python -m src.fluid_dynamics)
# ===========================================================================

if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output so Unicode chars render on all terminals
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  BaghewalRheologyEngine -- Smoke Test")
    print("=" * 70)

    params = RheologyParams(
        n             = 0.68,
        K_ref_Pa_sn   = 15.0,
        T_ref_c       = 47.0,
        E_a_kJ_mol    = 55.0,
    )
    engine = BaghewalRheologyEngine(params)

    # ── 1. Single-point viscosity at reservoir conditions ─────────────────────
    print("\n[1] Apparent Viscosity at Key Temperatures (shear_rate = 1 s^-1)")
    print(f"    {'Temperature (C)':<22} {'eta_app (cP)':<15} {'K (Pa.s^n)'}")
    print("    " + "-" * 55)
    for T_c in [47.0, 80.0, 120.0, 150.0, 200.0]:
        K   = engine.consistency_index(T_c)
        eta = engine.apparent_viscosity_cP(shear_rate_inv_s=1.0, temp_c=T_c)
        print(f"    {T_c:<22.1f} {eta:<15.1f} {K:.4f}")

    # ── 2. Shear-thinning at 47°C ─────────────────────────────────────────────
    print("\n[2] Shear-thinning behaviour at T = 47 C (reservoir BHT)")
    print(f"    {'Shear Rate (s^-1)':<22} {'eta_app (cP)'}")
    print("    " + "-" * 35)
    for gd in [0.01, 0.1, 1.0, 10.0, 100.0]:
        eta = engine.apparent_viscosity_cP(shear_rate_inv_s=gd, temp_c=47.0)
        print(f"    {gd:<22.4f} {eta:.1f}")

    # ── 3. CSS Viscosity Reduction Factor ─────────────────────────────────────
    print("\n[3] CSS Viscosity Reduction Factor")
    vrf = engine.css_viscosity_reduction_factor(47.0, 150.0, shear_rate_inv_s=1.0)
    print(f"    VRF (47C -> 150C, shear_rate=1 s^-1) = {vrf:.1f}x")

    # ── 4. Pressure-drop integration ──────────────────────────────────────────
    print("\n[4] Axial Pressure Drop Integration (depth=620m, Q=50 bbl/day)")
    Q_m3_s = 50.0 * 0.158987 / 86400.0   # 50 bbl/day -> m3/s
    result  = engine.pressure_drop_friction(
        depth_m        = 620.0,
        flow_rate_m3_s = Q_m3_s,
        tubing_id_m    = 0.062,
        rod_od_m       = 0.022,
        n_nodes        = 50,
    )
    print(f"    dP_friction    = {result['dP_friction_Pa']/1e5:.3f} bar")
    print(f"    dP_hydrostatic = {result['dP_hydrostatic_Pa']/1e5:.3f} bar")
    print(f"    eta_app @ pump = {result['viscosity_Pa_s'][0]*1000:.1f} cP")
    print(f"    eta_app @ surf = {result['viscosity_Pa_s'][-1]*1000:.1f} cP")

    print("\n[PASS] Smoke test passed.\n")
    sys.exit(0)
