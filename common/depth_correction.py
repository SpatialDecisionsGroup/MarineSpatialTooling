"""Water column depth correction and sun glint removal for optically shallow coastal water.

Two complementary approaches:

1. Lyzenga (1978) depth-invariant bottom index
   Uses log-transformed band ratios to cancel depth-dependent attenuation without
   requiring measured depth.  Works on any reflectance/Rrs source.

2. Explicit Beer-Lambert correction
   Requires measured water depth (available in the Tampa Bay JSON Depth field) and
   approximate diffuse attenuation coefficients (Kd) per band.  Produces per-band
   depth-corrected bottom reflectance.

Both can be applied to bands sampled from GEE SR products or ACOLITE rhos output.

Sun glint correction (Hedley et al. 2005) is also provided for use when ACOLITE's
own glint removal has not been applied.

References
----------
Lyzenga, D.R. (1978) Passive remote sensing techniques for mapping water depth and
    bottom features. Applied Optics, 17(3), 379-383.
Beer-Lambert model: Maritorena et al. (1994).
Hedley, J.D. et al. (2005) Simple and robust removal of sun glint for mapping
    shallow-water benthos. International Journal of Remote Sensing, 26(10), 2107-2112.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Lyzenga depth-invariant bottom index
# ---------------------------------------------------------------------------

# (output_column, band_i_column, band_j_column, ki_kj_ratio)
# ki_kj ratios for moderately turbid coastal / estuarine water (Class II).
# Sources: Lyzenga 1981, Mumby et al. 1998.  Refine from scene-level regression
# when sufficient depth variation is present in the imagery.

S2_LYZENGA_PAIRS: dict[str, tuple[str, str, float]] = {
    "s2_lyzenga_b1b2": ("s2_b1", "s2_b2", 1.5),   # 443 / 492 nm  coastal / blue
    "s2_lyzenga_b2b3": ("s2_b2", "s2_b3", 1.8),   # 492 / 560 nm  blue / green
    "s2_lyzenga_b3b4": ("s2_b3", "s2_b4", 0.6),   # 560 / 665 nm  green / red
}

LS_LYZENGA_PAIRS: dict[str, tuple[str, str, float]] = {
    "ls_lyzenga_b1b2": ("ls_b1", "ls_b2", 1.4),   # 443 / 482 nm  coastal / blue
    "ls_lyzenga_b2b3": ("ls_b2", "ls_b3", 1.7),   # 482 / 561 nm  blue / green
    "ls_lyzenga_b3b4": ("ls_b3", "ls_b4", 0.6),   # 561 / 655 nm  green / red
}


def lyzenga_index(ri: float, rj: float, ki_kj: float) -> float:
    """Lyzenga depth-invariant bottom index for one band pair.

    DII_ij = ln(Ri) – (ki/kj) × ln(Rj)

    Both inputs must be strictly positive; returns NaN otherwise.
    """
    if np.isnan(ri) or np.isnan(rj) or ri <= 0 or rj <= 0:
        return np.nan
    return float(np.log(ri) - ki_kj * np.log(rj))


def add_lyzenga_columns(
    features: dict[str, float],
    pairs: dict[str, tuple[str, str, float]],
) -> dict[str, float]:
    """Add Lyzenga indices to a feature dict in place and return it."""
    for col_name, (bi_col, bj_col, ki_kj) in pairs.items():
        features[col_name] = lyzenga_index(
            features.get(bi_col, np.nan),
            features.get(bj_col, np.nan),
            ki_kj,
        )
    return features


# ---------------------------------------------------------------------------
# Explicit Beer-Lambert depth correction
# ---------------------------------------------------------------------------

# Approximate Kd (m⁻¹) for moderately turbid coastal/estuarine water.
# Only visible bands are corrected; SWIR is almost fully absorbed by the water
# column and does not carry bottom signal.
# Adjust these values if scene-level Kd estimates are available from ACOLITE.
S2_KD: dict[str, float] = {
    "s2_b1": 0.50,   # 443 nm
    "s2_b2": 0.30,   # 492 nm
    "s2_b3": 0.15,   # 560 nm
    "s2_b4": 0.25,   # 665 nm
    "s2_b5": 0.30,   # 704 nm
    "s2_b6": 0.35,   # 740 nm
    "s2_b7": 0.40,   # 783 nm
    "s2_b8": 0.45,   # 842 nm
    "s2_b8a": 0.45,  # 865 nm
}

LS_KD: dict[str, float] = {
    "ls_b1": 0.50,   # 443 nm
    "ls_b2": 0.30,   # 482 nm
    "ls_b3": 0.15,   # 561 nm
    "ls_b4": 0.25,   # 655 nm
    "ls_b5": 0.45,   # 865 nm
}


def beer_lambert_correction(rrs: float, depth_m: float, kd: float) -> float:
    """Two-way Beer-Lambert depth correction.

    Rrs_bottom = Rrs_measured × exp(2 × Kd × depth_m)

    Assumes Rrs_optically_deep ≈ 0, which holds for short visible wavelengths
    over seagrass beds.  depth_m must be a positive number (metres below surface).
    Result is capped at 1.0 to keep values physically plausible.
    """
    if np.isnan(rrs) or np.isnan(depth_m) or depth_m <= 0 or kd <= 0:
        return np.nan
    return float(min(rrs * np.exp(2.0 * kd * depth_m), 1.0))


def add_depth_corrected_columns(
    features: dict[str, float],
    depth_m: float,
    kd_map: dict[str, float],
    prefix: str = "drc_",
) -> dict[str, float]:
    """Add Beer-Lambert depth-corrected columns to a feature dict in place."""
    for band_col, kd in kd_map.items():
        features[f"{prefix}{band_col}"] = beer_lambert_correction(
            features.get(band_col, np.nan), depth_m, kd
        )
    return features


# Flat lists of column names for initialising DataFrame columns.
S2_DRC_COLUMNS: list[str] = [f"drc_{c}" for c in S2_KD]
LS_DRC_COLUMNS: list[str] = [f"drc_{c}" for c in LS_KD]
S2_LYZENGA_COLUMNS: list[str] = list(S2_LYZENGA_PAIRS.keys())
LS_LYZENGA_COLUMNS: list[str] = list(LS_LYZENGA_PAIRS.keys())


# ---------------------------------------------------------------------------
# Hedley et al. (2005) sun glint correction
# ---------------------------------------------------------------------------

def hedley_sunglint_correction(
    band_values: dict[str, float],
    nir_col: str,
    nir_min: float,
    slopes: dict[str, float],
) -> dict[str, float]:
    """Apply Hedley et al. sun glint correction to visible bands.

    For each visible band i:
        Rrs_corrected_i = Rrs_i – b_i × (NIR – NIR_min)

    Parameters
    ----------
    band_values : mapping of column name → reflectance value
    nir_col     : column name for the NIR band used as glint proxy
    nir_min     : minimum NIR value from optically deep (glint-free) water
    slopes      : mapping of visible band column → regression slope b_i,
                  pre-computed from sun-glint pixels in the target scene

    Returns a new dict with corrected values for the keys in `slopes`.
    """
    nir = band_values.get(nir_col, np.nan)
    corrected = dict(band_values)
    if np.isnan(nir):
        return corrected
    glint_signal = float(nir) - nir_min
    for band_col, slope in slopes.items():
        orig = band_values.get(band_col, np.nan)
        if not np.isnan(orig):
            corrected[band_col] = float(orig) - slope * glint_signal
    return corrected
