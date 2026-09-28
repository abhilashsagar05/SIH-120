"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : src/data_ingestion.py
  Class  : BaghewalaSCADA
  Author : Digital Twin Engineering Team
  Purpose: Synthetic high-frequency SCADA telemetry generator for the Baghewala
           field SRP + CSS production system.

  Physics simulated
  -----------------
  1. Surface rod displacement  — harmonic walking-beam kinematics
  2. Polished rod load (PRL)   — dynamic balance of forces on the rod string:
       PRL = W_rod + W_fluid + F_inertia - F_friction ± F_pump_card
  3. Near-wellbore temperature — exponential CSS thermal decay overlaid with
       wellbore heat-conduction transient and formation-fluid thermal mixing

  Key field constants (Baghewala)
  --------------------------------
  API gravity  : 14 – 19°API   → heavy crude, ~935–969 kg/m³
  BHT baseline : 46 – 48°C
  In-situ visc : 10,000 – 13,000 cP at reservoir temperature
  SRP strokes  : 4–6 SPM nominal
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger — importers can inject their own handler
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ===========================================================================
# §1  Field & Equipment Parameter Dataclass
# ===========================================================================

@dataclass
class BaghewalaSRPConfig:
    """
    Immutable hardware and reservoir parameters for one Baghewala SRP well.
    All physical quantities use SI base units internally; helper properties
    expose field-convenient units where noted.

    Parameters
    ----------
    well_id            : Unique well identifier string (e.g. 'BGW-07').
    depth_m            : True vertical depth to pump intake, metres.
    stroke_length_m    : Polished-rod stroke length, metres.
    spm                : Nominal strokes per minute at surface.
    rod_od_m           : Rod outside diameter, metres.
    rod_density_kg_m3  : Rod material density (steel ≈ 7850 kg/m³).
    tubing_id_m        : Tubing inside diameter, metres.
    oil_api            : Stock-tank crude API gravity.
    bht_baseline_c     : Virgin (pre-steam) bottom-hole temperature, °C.
    css_peak_temp_c    : Peak near-wellbore temperature post-steam injection, °C.
    css_thermal_decay_days : Characteristic time (τ) for exponential BHT
                             recovery after CSS soak period, days.
    pump_efficiency    : Volumetric efficiency of downhole pump [0–1].
    noise_std_frac     : Standard deviation of Gaussian noise as a fraction
                         of the signal amplitude (e.g. 0.02 = 2% noise).
    """

    well_id: str                   = "BGW-07"
    depth_m: float                 = 620.0        # Jodhpur Sdst depth
    stroke_length_m: float         = 3.0          # m
    spm: float                     = 4.5          # strokes per minute
    rod_od_m: float                = 0.022        # 7/8" rod (≈22 mm OD)
    rod_density_kg_m3: float       = 7850.0       # carbon steel
    tubing_id_m: float             = 0.062        # 2-3/8" tubing
    oil_api: float                 = 16.5         # midpoint Baghewala range
    bht_baseline_c: float          = 47.0         # °C
    css_peak_temp_c: float         = 200.0        # °C  post-steam near-wellbore
    css_thermal_decay_days: float  = 30.0         # τ for exponential decay
    pump_efficiency: float         = 0.72         # fraction
    noise_std_frac: float          = 0.02         # 2 % Gaussian noise

    # --- derived constants (computed in __post_init__) ---------------------
    oil_density_kg_m3: float       = field(init=False)
    rod_cross_area_m2: float       = field(init=False)
    tubing_cross_area_m2: float    = field(init=False)
    G: float                       = field(default=9.80665, init=False)  # m/s²

    def __post_init__(self) -> None:
        # Oil density from API gravity  (ρ = 141.5/(131.5 + API) × 1000 kg/m³)
        self.oil_density_kg_m3 = (141.5 / (131.5 + self.oil_api)) * 1000.0
        # Cross-sectional areas
        self.rod_cross_area_m2   = np.pi / 4 * self.rod_od_m ** 2
        self.tubing_cross_area_m2 = np.pi / 4 * self.tubing_id_m ** 2

        logger.debug(
            "SRPConfig initialised: well=%s, ρ_oil=%.1f kg/m³, "
            "A_rod=%.6f m², A_tub=%.6f m²",
            self.well_id, self.oil_density_kg_m3,
            self.rod_cross_area_m2, self.tubing_cross_area_m2,
        )


# ===========================================================================
# §2  BaghewalaSCADA — Telemetry Simulator
# ===========================================================================

class BaghewalaSCADA:
    """
    Synthetic SCADA telemetry generator for the Baghewala field SRP system.

    Simulates four primary SCADA channels at configurable sample frequency:

    ┌────────────────────────────┬───────────────────────────────────────────┐
    │ Channel                    │ Physics basis                             │
    ├────────────────────────────┼───────────────────────────────────────────┤
    │ rod_displacement_m         │ Harmonic walking-beam kinematics          │
    │ polished_rod_load_kN       │ Dynamic force balance on rod string       │
    │ near_wellbore_temp_c       │ CSS exponential decay + wellbore transient│
    │ pump_intake_pressure_bar   │ Hydrostatic column ± drawdown fluctuation │
    └────────────────────────────┴───────────────────────────────────────────┘

    Usage
    -----
    >>> cfg   = BaghewalaSRPConfig(well_id="BGW-12", spm=5.0)
    >>> scada = BaghewalaSCADA(cfg, sample_hz=10.0)
    >>> df    = scada.generate(duration_hours=24.0, css_day_offset=5.0)
    >>> df.head()
    """

    def __init__(
        self,
        config: Optional[BaghewalaSRPConfig] = None,
        sample_hz: float = 10.0,
        random_seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        config      : BaghewalaSRPConfig instance (defaults to field midpoint).
        sample_hz   : SCADA sample frequency, Hz (10 Hz → 0.1 s resolution).
        random_seed : NumPy random seed for reproducibility.
        """
        self.cfg        = config if config is not None else BaghewalaSRPConfig()
        self.sample_hz  = sample_hz
        self._dt        = 1.0 / sample_hz          # seconds per sample
        self._rng       = np.random.default_rng(random_seed)

        # Angular frequency of the walking beam (rad/s)
        # ω = 2π × (SPM / 60)
        self._omega: float = 2.0 * np.pi * (self.cfg.spm / 60.0)

        logger.info(
            "BaghewalaSCADA ready | well=%s | %.1f Hz | ω=%.4f rad/s",
            self.cfg.well_id, sample_hz, self._omega,
        )

    # -----------------------------------------------------------------------
    # §2.1  Walking-Beam Kinematics — Surface Rod Displacement
    # -----------------------------------------------------------------------

    def _rod_displacement(self, t: np.ndarray) -> np.ndarray:
        """
        Compute surface polished-rod displacement using harmonic kinematics.

        The walking-beam converts the crank's circular motion into an
        approximately sinusoidal surface displacement.  A small third-harmonic
        term (≈3 % amplitude) accounts for the non-ideal four-bar linkage
        geometry found on conventional beam pumping units.

        Model
        -----
            x(t) = (L/2) × [sin(ωt) + 0.03 × sin(3ωt)]

        where L is the full stroke length (peak-to-peak = L).

        Parameters
        ----------
        t : time array, seconds

        Returns
        -------
        displacement : metres, measured from mid-stroke (positive = upstroke)
        """
        half_stroke = self.cfg.stroke_length_m / 2.0

        # Primary harmonic (fundamental walking-beam motion)
        primary      = half_stroke * np.sin(self._omega * t)
        # Third-harmonic correction for 4-bar linkage non-linearity
        third_harm   = half_stroke * 0.03 * np.sin(3.0 * self._omega * t)

        displacement = primary + third_harm
        return displacement

    # -----------------------------------------------------------------------
    # §2.2  Dynamic Polished Rod Load (PRL)
    # -----------------------------------------------------------------------

    def _polished_rod_load(
        self,
        t: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the dynamic Polished Rod Load (PRL) in Newtons.

        The PRL is the instantaneous tension at the top of the rod string.
        It varies throughout the stroke because several forces are phase-shifted
        with respect to each other.  The complete dynamic balance is:

            PRL(t) = W_rod_buoyed
                   + W_fluid (upstroke only)
                   - F_friction
                   + F_inertia
                   + ε_noise

        Component breakdown
        -------------------
        W_rod_buoyed :
            Buoyed weight of the rod string in the produced fluid.
            W_rod  = ρ_steel × g × A_rod × L_depth
            W_buoy = W_rod × (1 − ρ_oil/ρ_steel)

        W_fluid :
            Weight of the fluid column lifted by the pump plunger.
            Active on upstroke (sin(ωt) > 0), off on downstroke.
            W_fluid = ρ_oil × g × (A_tubing − A_rod) × L_depth × η_pump

        F_inertia :
            Inertial reaction of the rod string during acceleration.
            Uses the velocity obtained by differentiating the displacement:
                v(t)  = dx/dt = (L/2) × ω × cos(ωt)
                a(t)  = dv/dt = −(L/2) × ω² × sin(ωt)
            F_inertia = m_rod × a(t)

        F_friction :
            Viscous drag of the heavy crude against the rod string OD.
            Modelled as proportional to velocity (Stokes-drag approximation).
            A tunable coefficient C_fric absorbs wellbore contact friction.

        Parameters
        ----------
        t            : time array, seconds
        displacement : rod displacement array, metres (from _rod_displacement)

        Returns
        -------
        prl_kN : Polished rod load array, kilo-Newtons
        """
        cfg = self.cfg
        g   = cfg.G

        # ── Rod string mass ─────────────────────────────────────────────────
        m_rod = cfg.rod_density_kg_m3 * cfg.rod_cross_area_m2 * cfg.depth_m  # kg

        # ── Buoyed rod weight ────────────────────────────────────────────────
        # Buoyancy factor: γ = 1 − (ρ_fluid / ρ_rod)
        buoyancy_factor = 1.0 - (cfg.oil_density_kg_m3 / cfg.rod_density_kg_m3)
        W_rod_buoyed    = m_rod * g * buoyancy_factor                          # N

        # ── Fluid column weight above pump plunger ───────────────────────────
        # Effective annular area that the plunger acts on:
        #   A_eff = A_tubing_inside − A_rod (net lift area)
        A_eff    = cfg.tubing_cross_area_m2 - cfg.rod_cross_area_m2           # m²
        W_fluid  = cfg.oil_density_kg_m3 * g * A_eff * cfg.depth_m * cfg.pump_efficiency  # N

        # ── Velocity and acceleration of polished rod ─────────────────────────
        # Derivative of x(t) = (L/2)[sin(ωt) + 0.03 sin(3ωt)]:
        #   v(t) = (L/2)ω[cos(ωt) + 0.09 cos(3ωt)]
        #   a(t) = −(L/2)ω²[sin(ωt) + 0.27 sin(3ωt)]
        half_L = cfg.stroke_length_m / 2.0
        v_rod  = half_L * self._omega * (
            np.cos(self._omega * t) + 0.09 * np.cos(3.0 * self._omega * t)
        )
        a_rod  = -half_L * (self._omega ** 2) * (
            np.sin(self._omega * t) + 0.27 * np.sin(3.0 * self._omega * t)
        )

        # ── Inertial force of rod string ─────────────────────────────────────
        F_inertia = m_rod * a_rod                                              # N

        # ── Viscous & contact friction ────────────────────────────────────────
        # C_fric is an empirical drag coefficient (N·s/m).
        # For a 10,000–13,000 cP fluid at slow SRP speeds, friction is
        # predominantly viscous; its sign opposes velocity direction.
        # Baseline value: 2 × 10³ N·s/m (calibrated to Baghewala field logs)
        C_fric    = 2000.0                                                     # N·s/m
        F_friction = C_fric * v_rod                                            # N  (opposes v)

        # ── Upstroke / downstroke pump-card asymmetry ─────────────────────────
        # The fluid load is only borne by the rods on the UPSTROKE.
        # upstroke_mask = 1 when v > 0 (displacement increasing), 0 otherwise.
        upstroke_mask = (v_rod > 0).astype(float)

        # ── Dynamic PRL assembly ──────────────────────────────────────────────
        # PRL = static buoyed rod weight
        #     + fluid weight (upstroke only)
        #     + inertia (alternates sign through the stroke)
        #     − friction (always opposing motion → subtract from net PRL)
        prl_N = (
            W_rod_buoyed
            + W_fluid * upstroke_mask
            + F_inertia
            - F_friction
        )

        # ── Gaussian sensor noise ─────────────────────────────────────────────
        prl_amplitude = W_rod_buoyed + W_fluid           # peak-ish reference
        noise = self._rng.normal(
            scale=cfg.noise_std_frac * prl_amplitude,
            size=t.shape,
        )
        prl_N += noise

        return prl_N / 1000.0                                                  # kN

    # -----------------------------------------------------------------------
    # §2.3  Near-Wellbore Temperature Transient (CSS thermal model)
    # -----------------------------------------------------------------------

    def _near_wellbore_temperature(
        self,
        t: np.ndarray,
        css_day_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate the near-wellbore temperature trajectory during and after
        a Cyclic Steam Stimulation (CSS) event.

        CSS thermal sequence
        --------------------
        1. Injection phase : Temperature rises rapidly to css_peak_temp_c.
        2. Soak phase      : Temperature held near peak for soak duration.
        3. Production phase: Exponential decay back toward BHT baseline,
                             modulated by formation-fluid mixing and wellbore
                             heat-conduction transients (superimposed log-term).

        Mathematical model
        ------------------
        Let τ = css_thermal_decay_days (characteristic cooling time).
        Let t_day = elapsed days since start of production (after soak).

            T_css(t_day) = T_baseline + ΔT × exp(−t_day / τ)
                         + ΔT_conduct × ln(1 + t_day / τ_cond) × exp(−t_day / τ)

        where
            ΔT       = T_peak − T_baseline          (heating delta)
            ΔT_cond  = 0.08 × ΔT                   (wellbore conduction term)
            τ_cond   = 5 days                       (conduction lag time)

        A small high-frequency thermal noise term is added to mimic wellbore
        fluid-mixing heterogeneity (σ = 0.5°C).

        Parameters
        ----------
        t              : time array, seconds (referenced from recording start)
        css_day_offset : days elapsed since CSS soak end when recording begins
                         (0 = recording starts immediately post-soak)

        Returns
        -------
        temp_c : near-wellbore temperature array, °C
        """
        cfg         = self.cfg
        delta_T     = cfg.css_peak_temp_c - cfg.bht_baseline_c   # °C delta
        tau_sec     = cfg.css_thermal_decay_days * 86400.0        # τ in seconds
        tau_cond    = 5.0 * 86400.0                               # 5-day conduction lag

        # Elapsed time from CSS soak end (seconds)
        t_elapsed = t + css_day_offset * 86400.0

        # ── Primary exponential decay ─────────────────────────────────────────
        T_primary = cfg.bht_baseline_c + delta_T * np.exp(-t_elapsed / tau_sec)

        # ── Wellbore heat-conduction modulation ───────────────────────────────
        # The log-term captures the transient radial heat conduction from the
        # steam-heated near-wellbore zone into the formation.  As time increases,
        # the conduction term rises initially (heat reaches further radii) but
        # the overall envelope still decays exponentially.
        delta_T_cond = 0.08 * delta_T
        T_conduction = (
            delta_T_cond
            * np.log1p(t_elapsed / tau_cond)
            * np.exp(-t_elapsed / tau_sec)
        )

        # ── High-frequency fluid-mixing heterogeneity noise ───────────────────
        T_noise = self._rng.normal(scale=0.5, size=t.shape)

        # ── Clamp to physical range ───────────────────────────────────────────
        T_total = T_primary + T_conduction + T_noise
        T_total = np.clip(T_total, cfg.bht_baseline_c, cfg.css_peak_temp_c)

        return T_total

    # -----------------------------------------------------------------------
    # §2.4  Pump Intake Pressure
    # -----------------------------------------------------------------------

    def _pump_intake_pressure(self, t: np.ndarray) -> np.ndarray:
        """
        Estimate pump intake pressure (PIP) at downhole pump depth.

        PIP ≈ hydrostatic head of fluid column minus the dynamic drawdown.

            PIP(t) = ρ_oil × g × H_fluid
                   − ΔP_drawdown × sin(ω_dd × t)
                   + ε_noise

        The sinusoidal drawdown oscillation (ω_dd ≈ 0.1 × ω_rod) captures
        the low-frequency pressure pulse caused by each pump stroke in the
        near-wellbore region.

        Returns
        -------
        pip_bar : pump intake pressure, bar
        """
        cfg = self.cfg
        # Static hydrostatic pressure (assuming partial fluid column fill)
        fill_fraction = 0.85                         # 85% liquid level (typical)
        H_fluid       = fill_fraction * cfg.depth_m  # effective fluid head, m

        P_hydrostatic = cfg.oil_density_kg_m3 * cfg.G * H_fluid  # Pa

        # Drawdown oscillation — low frequency (once per ~10 strokes)
        omega_dd     = 0.1 * self._omega
        dP_drawdown  = 0.05 * P_hydrostatic           # 5% amplitude fluctuation
        P_drawdown   = dP_drawdown * np.sin(omega_dd * t)

        # Sensor noise (1% of hydrostatic)
        P_noise = self._rng.normal(
            scale=cfg.noise_std_frac * P_hydrostatic, size=t.shape
        )

        pip_Pa  = P_hydrostatic - P_drawdown + P_noise
        pip_bar = np.clip(pip_Pa / 1e5, 0.0, None)   # convert Pa → bar, clamp ≥0

        return pip_bar

    # -----------------------------------------------------------------------
    # §2.5  Public API — generate()
    # -----------------------------------------------------------------------

    def generate(
        self,
        duration_hours: float = 24.0,
        start_timestamp: str  = "2026-01-01 00:00:00",
        css_day_offset: float = 3.0,
    ) -> pd.DataFrame:
        """
        Generate a synthetic SCADA telemetry DataFrame for the configured well.

        Parameters
        ----------
        duration_hours  : Length of the telemetry window to simulate, hours.
        start_timestamp : ISO-format string for the first timestamp.
        css_day_offset  : Days elapsed since CSS soak end when this recording
                          begins (controls how far along the thermal decay curve
                          the simulation starts).  Use 0.0 for fresh post-steam.

        Returns
        -------
        pd.DataFrame with columns:
            timestamp               : DatetimeTZNaive index (freq = 1/sample_hz s)
            well_id                 : str — well identifier
            rod_displacement_m      : float — polished rod displacement, m
            rod_velocity_m_s        : float — polished rod velocity, m/s
            polished_rod_load_kN    : float — dynamic PRL, kN
            near_wellbore_temp_c    : float — near-wellbore temperature, °C
            pump_intake_pressure_bar: float — PIP, bar
            stroke_phase_rad        : float — instantaneous crank angle, rad ∈ [0, 2π)
        """
        n_samples = int(duration_hours * 3600.0 * self.sample_hz)
        logger.info(
            "Generating %d samples (%.1f h × %.0f Hz) for well %s, "
            "CSS offset = %.1f days",
            n_samples, duration_hours, self.sample_hz,
            self.cfg.well_id, css_day_offset,
        )

        # ── Time axis ─────────────────────────────────────────────────────────
        t = np.arange(n_samples, dtype=np.float64) * self._dt   # seconds

        # ── Channel computations ──────────────────────────────────────────────
        displacement  = self._rod_displacement(t)
        prl_kN        = self._polished_rod_load(t, displacement)
        temp_c        = self._near_wellbore_temperature(t, css_day_offset)
        pip_bar       = self._pump_intake_pressure(t)

        # ── Rod velocity (finite-difference for any non-analytic displacement) ─
        rod_vel = np.gradient(displacement, self._dt)

        # ── Stroke phase angle ────────────────────────────────────────────────
        phase_rad = (self._omega * t) % (2.0 * np.pi)

        # ── Timestamp index ───────────────────────────────────────────────────
        timestamps = pd.date_range(
            start=start_timestamp,
            periods=n_samples,
            freq=pd.tseries.frequencies.to_offset(f"{self._dt}s"),
        )

        # ── Assemble DataFrame ────────────────────────────────────────────────
        df = pd.DataFrame(
            {
                "well_id"                  : self.cfg.well_id,
                "rod_displacement_m"       : displacement,
                "rod_velocity_m_s"         : rod_vel,
                "polished_rod_load_kN"     : prl_kN,
                "near_wellbore_temp_c"     : temp_c,
                "pump_intake_pressure_bar" : pip_bar,
                "stroke_phase_rad"         : phase_rad,
            },
            index=timestamps,
        )
        df.index.name = "timestamp"

        logger.info(
            "Telemetry generated: shape=%s | PRL range=[%.2f, %.2f] kN | "
            "T range=[%.1f, %.1f] °C",
            df.shape,
            df["polished_rod_load_kN"].min(),
            df["polished_rod_load_kN"].max(),
            df["near_wellbore_temp_c"].min(),
            df["near_wellbore_temp_c"].max(),
        )
        return df


# ===========================================================================
# §3  Convenience factory — multi-well fleet simulation
# ===========================================================================

def simulate_well_fleet(
    well_configs: list[BaghewalaSRPConfig],
    duration_hours: float = 8.0,
    sample_hz: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate a fleet of Baghewala SRP wells and return a combined DataFrame.

    Parameters
    ----------
    well_configs    : List of BaghewalaSRPConfig instances, one per well.
    duration_hours  : Simulation window length for each well.
    sample_hz       : Uniform sample frequency for all wells.

    Returns
    -------
    pd.DataFrame concatenated across all wells, with 'well_id' as a column
    for easy group-by operations.
    """
    frames: list[pd.DataFrame] = []
    for i, cfg in enumerate(well_configs):
        scada  = BaghewalaSCADA(cfg, sample_hz=sample_hz, random_seed=42 + i)
        df     = scada.generate(
            duration_hours=duration_hours,
            css_day_offset=float(i * 2),   # stagger CSS offsets per well
        )
        frames.append(df)
        logger.info("Well %s telemetry complete.", cfg.well_id)

    combined = pd.concat(frames, axis=0).sort_index()
    return combined


# ===========================================================================
# §4  Module smoke-test  (python -m src.data_ingestion)
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  BaghewalaSCADA — Smoke Test")
    print("=" * 70)

    # Single-well test
    cfg   = BaghewalaSRPConfig(well_id="BGW-07", spm=4.5, oil_api=16.5)
    scada = BaghewalaSCADA(cfg, sample_hz=10.0)
    df    = scada.generate(duration_hours=1.0, css_day_offset=3.0)

    print("\n[1] DataFrame info:")
    print(df.info())

    print("\n[2] First 5 rows:")
    print(df.head().to_string())

    print("\n[3] Descriptive statistics:")
    print(df.describe().to_string())

    # Save to synthetic data directory
    out_path = "data/synthetic/BGW-07_telemetry_1h.parquet"
    try:
        df.to_parquet(out_path)
        print(f"\n[4] Saved to {out_path}")
    except Exception as exc:
        print(f"\n[4] Could not save parquet: {exc} (run from project root)")

    print("\n Smoke test passed.\n")
    sys.exit(0)
