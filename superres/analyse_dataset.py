"""
Dataset analysis for the super-resolution training set.

Each panel is saved as its own file AND as part of a combined multi-panel
figure for the group.  Naming convention: <group><panel>_<description>.png

  01a  spatial_environment          world map, environment class
  01b  spatial_depth                world map, depth class
  01c  spatial_turbidity            world map, turbidity class
  01   spatial_distribution         combined 1×3
  02   depth_turbidity_density      2-D histogram, estuary/offshore overlaid
  03a  reflectance_landsat          per-band violin, Landsat only
  03b  reflectance_sentinel2        per-band violin, Sentinel-2 only
  03_panels  reflectance_panels     Landsat + S2 side-by-side subplots (1×2)
  03   reflectance_combined         both sensors on one axis, sorted by wavelength
  04a  spectral_lr_hr_scatter       LR vs HR reflectance scatter
  04b  spectral_bias_by_band        S2−Landsat bias boxplot per band
  04c  spectral_bias_vs_gap         bias vs LR–HR acquisition time gap
  04   spectral_difficulty          combined 1×3
  05a  hist_depth_by_env            depth histogram by environment
  05b  hist_turbidity_by_env        turbidity histogram by environment
  05c  hist_depth_by_turbidity      depth histogram by turbidity class
  05d  hist_turbidity_by_depth      turbidity histogram by depth class
  05e  hist_depth_all               depth histogram, all samples
  05f  hist_turbidity_log_all       log turbidity histogram, all samples
  05   class_histograms             combined 2×3
  06a  temporal_year                sample count per year
  06b  temporal_season              season × environment count
  06   temporal                     combined 1×2
  07a  coverage_hemisphere          hemisphere pie chart
  07b  coverage_latitude            latitude histogram
  07   patch_coverage               combined 1×2
  08   psd_curves                   radial PSD: LR, HR, bicubic LR  (single panel)
  09a–g lr_hr_hist_<band>           paired LR/HR intensity histogram per band
  09   lr_hr_histograms             combined 2×4 (7 panels)
  10a  alignment_scatter            phase-correlation 2-D shift scatter
  10b  alignment_magnitude          shift magnitude histogram
  10c  alignment_cdf                shift magnitude CDF
  10   alignment_quality            combined 1×3
  11a  thumbnails_random            random LR/HR RGB pairs
  11b  thumbnails_bad_alignment     worst phase-correlation shift pairs
  11c  thumbnails_clipping          worst Landsat saturation pairs

Usage
-----
  uv run python -m superres.analyse_dataset \\
      data/landsat2sentinel/metadata/dataset_manifest.csv \\
      data/landsat2sentinel/data \\
      --processed-dir data/landsat2sentinel/processed \\
      --output figures/dataset_analysis \\
      --n-band-samples 150
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom
from tqdm import tqdm

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ─── spectral metadata ───────────────────────────────────────────────────────

LANDSAT_SPECTRAL_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
LANDSAT_WAVELENGTH = {
    "SR_B1": 443, "SR_B2": 482, "SR_B3": 562, "SR_B4": 655,
    "SR_B5": 865, "SR_B6": 1610, "SR_B7": 2200,
}
LANDSAT_LABEL = {
    "SR_B1": "B1\n(Ultra-blue)", "SR_B2": "B2\n(Blue)", "SR_B3": "B3\n(Green)",
    "SR_B4": "B4\n(Red)", "SR_B5": "B5\n(NIR)", "SR_B6": "B6\n(SWIR-1)",
    "SR_B7": "B7\n(SWIR-2)",
}

S2_SPECTRAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_WAVELENGTH = {
    "B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705,
    "B6": 740, "B7": 783, "B8": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}
S2_LABEL = {
    "B1": "B1\n(443)", "B2": "B2\n(490)", "B3": "B3\n(560)", "B4": "B4\n(665)",
    "B5": "B5\n(705)", "B6": "B6\n(740)", "B7": "B7\n(783)", "B8": "B8\n(842)",
    "B8A": "B8A\n(865)", "B11": "B11\n(1610)", "B12": "B12\n(2190)",
}

# Spectrally matched LR→HR band pairs
COMPARABLE_PAIRS = [
    ("SR_B1", "B1", "~443 nm"),
    ("SR_B2", "B2", "~490 nm"),
    ("SR_B3", "B3", "~560 nm"),
    ("SR_B4", "B4", "~665 nm"),
    ("SR_B5", "B8A", "~865 nm"),
    ("SR_B6", "B11", "~1610 nm"),
    ("SR_B7", "B12", "~2200 nm"),
]

# The first matched-band pair used for spatial analysis (alignment, PSD)
LR_SPATIAL_BAND_IDX = 3   # SR_B4 (red, band index 1-based in rasterio)
HR_SPATIAL_BAND_IDX = 4   # B4   (red)

ENV_PALETTE  = {"estuary": "#2196F3", "offshore": "#FF7043"}
DEPTH_PALETTE = {"shallow_1": "#B3E5FC", "shallow_2": "#039BE5", "shallow_3": "#01579B"}
TURB_PALETTE  = {"clear": "#A5D6A7", "moderate": "#FFA726", "turbid": "#E53935"}
DEPTH_BIN_LABELS = {"shallow_1": "0–10 m", "shallow_2": "10–30 m", "shallow_3": "30–60 m"}

LR_COLOR = "#1565C0"   # dark blue for Landsat
HR_COLOR = "#AD1457"   # dark pink for Sentinel-2

FIG_DPI = 150


# ─── helpers ─────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def _s2_acquisition_date(item_id_str: str) -> datetime | None:
    try:
        ids = json.loads(item_id_str)
        if ids:
            m = re.search(r"/(\d{8})T", ids[0])
            if m:
                return datetime.strptime(m.group(1), "%Y%m%d")
    except Exception:
        pass
    return None


def _lr_dates(processed_dir: Path, location_id: int) -> list[datetime]:
    lr_dir = processed_dir / f"sample_{location_id:06d}" / "landsat"
    dates = []
    for f in sorted(lr_dir.glob("*.tif")):
        m = re.search(r"_(\d{4}-\d{2}-\d{2})\.tif", f.name)
        if m:
            try:
                dates.append(datetime.strptime(m.group(1), "%Y-%m-%d"))
            except ValueError:
                pass
    return dates


def _compute_time_gaps(df: pd.DataFrame, processed_dir: Path, location_ids: list[int]) -> dict[int, float]:
    """Minimum |S2_date – any LR_date| in days for each location_id."""
    s2_dates = {
        int(row["location_id"]): _s2_acquisition_date(row["highres_item_ids"])
        for _, row in df.iterrows()
        if _s2_acquisition_date(row["highres_item_ids"]) is not None
    }
    gaps: dict[int, float] = {}
    for loc_id in location_ids:
        s2_dt = s2_dates.get(loc_id)
        if s2_dt is None:
            continue
        lr_dts = _lr_dates(processed_dir, loc_id)
        if lr_dts:
            gaps[loc_id] = min(abs((s2_dt - d).days) for d in lr_dts)
    return gaps


def _sample_reflectance(
    processed_dir: Path,
    df: pd.DataFrame,
    satellite: str,
    band_names: list[str],
    band_indices: list[int],
    n_samples: int,
) -> pd.DataFrame:
    """
    Open N random processed rasters for *satellite*, compute patch-mean
    reflectance per band (clipped to [0, 1]; NaN nodata already masked),
    and return a long-format DataFrame.

    The long tail in unprocessed data comes from two sources:
      - Landsat: DN=0 nodata pixels → reflectance = -0.2 after scale/offset.
        The processed data sets these to NaN, so they are excluded here.
      - Sentinel-2: cloud pixels (SCL 8/9) can have reflectance > 1.0.
        These are genuine data; we clip to [0, 1] for display.
    """
    import rasterio

    rows = df.sample(n=min(n_samples, len(df)), random_state=42)
    records = []

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {satellite}", leave=False):
        loc_id = int(row["location_id"])
        sat_dir = processed_dir / f"sample_{loc_id:06d}" / satellite
        if not sat_dir.exists():
            continue
        files = sorted(sat_dir.glob("*.tif"))
        if not files:
            continue
        tif = files[0]
        try:
            with rasterio.open(tif) as src:
                for bname, bidx in zip(band_names, band_indices):
                    if bidx > src.count:
                        continue
                    data = src.read(bidx).astype(np.float32)
                    nodata = src.nodata
                    if nodata is not None:
                        data[data == nodata] = np.nan
                    valid = data[np.isfinite(data)]
                    if valid.size == 0:
                        continue
                    clip_frac = float((valid >= 1.0).mean())
                    valid = np.clip(valid, 0.0, 1.0)
                    records.append({
                        "location_id": loc_id,
                        "band": bname,
                        "reflectance": float(np.mean(valid)),
                        "clip_frac": clip_frac,
                        "environment_class": row["environment_class"],
                        "depth_class": row["depth_class"],
                        "turbidity_class": row["turbidity_class"],
                        "depth_m": row["depth_m"],
                        "turbidity_index": row["turbidity_index"],
                    })
        except Exception:
            continue

    return pd.DataFrame(records)


def _radial_psd(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    2D FFT → radial-averaged power spectral density, normalised by pixel count.

    Dividing by h*w gives power-per-pixel units so that a bicubically
    upsampled image (which has scale_factor^2 more pixels but no new
    high-frequency content) starts near the native LR curve at low frequencies.

    Returns (frequencies, psd) where frequencies are in units of
    cycles per pixel (0 to 0.5 Nyquist).
    """
    win = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
    f = np.fft.fftshift(np.fft.fft2(patch * win))
    h, w = patch.shape
    power = np.abs(f) ** 2 / (h * w)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy).astype(int)
    r_max = min(cx, cy)
    freqs = np.arange(r_max) / (2 * r_max)   # cycles per pixel, 0..0.5
    psd = np.array([power[r == i].mean() if (r == i).any() else 0.0 for i in range(r_max)])
    return freqs, psd


# ─── figure functions ─────────────────────────────────────────────────────────
# Pattern: each group saves individual panel files (01a, 01b, …) AND a
# combined multi-panel file (01_spatial_distribution.png).
# Panel rendering is factored into _draw_* helpers that accept an ax so the
# same code runs for both the individual and combined figures.

def _world_boundary():
    if not HAS_GEOPANDAS:
        return None
    try:
        import geodatasets
        return gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except Exception:
        return None


# ── 01 Spatial distribution ───────────────────────────────────────────────────

def _draw_spatial(ax, df, col, palette, title, world):
    if world is not None:
        world.boundary.plot(ax=ax, linewidth=0.4, color="#888888")
    for label, color in palette.items():
        sub = df[df[col] == label]
        ax.scatter(sub["longitude"], sub["latitude"],
                   c=color, s=6, alpha=0.55, linewidths=0, label=label)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.7)
    ax.grid(True, linewidth=0.3, alpha=0.5)


def fig_spatial_distribution(df: pd.DataFrame, out_dir: Path):
    world = _world_boundary()
    specs = [
        ("environment_class", ENV_PALETTE,   "Environment class",
         "01a_spatial_environment.png"),
        ("depth_class",       DEPTH_PALETTE, "Depth class",
         "01b_spatial_depth.png"),
        ("turbidity_class",   TURB_PALETTE,  "Turbidity class",
         "01c_spatial_turbidity.png"),
    ]
    for col, palette, title, fname in specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_spatial(ax, df, col, palette, title, world)
        _savefig(fig, out_dir / fname)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    for ax, (col, palette, title, _) in zip(axes, specs):
        _draw_spatial(ax, df, col, palette, title, world)
    axes[1].set_ylabel(""); axes[2].set_ylabel("")
    fig.suptitle(f"Spatial distribution of {len(df):,} dataset patches", fontsize=12)
    _savefig(fig, out_dir / "01_spatial_distribution.png")


# ── 02 Depth–turbidity density ────────────────────────────────────────────────

_LOG_TICKS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])


def _set_log_yaxis(ax):
    ax.set_yticks(_LOG_TICKS)
    ax.set_yticklabels([f"{10**t:.3g}" for t in _LOG_TICKS])
    ax.set_ylabel("Turbidity index (log scale)")
    for thresh, color in [(0.03, "#FFA726"), (0.08, "#E53935")]:
        ax.axhline(np.log10(thresh), color=color, lw=1.0, ls="--", alpha=0.8)



def _draw_depth_turb_hist(ax, valid):
    turb_vals = valid["turbidity_index"]
    depth_bins = np.linspace(valid["depth_m"].min(), valid["depth_m"].max(), 31)
    log_bins   = np.linspace(np.log10(turb_vals.min()), np.log10(turb_vals.max()), 31)

    for env, hex_color in ENV_PALETTE.items():
        sub = valid[valid["environment_class"] == env]
        if sub.empty:
            continue
        counts, xedges, yedges = np.histogram2d(
            sub["depth_m"], np.log10(sub["turbidity_index"]),
            bins=[depth_bins, log_bins],
        )
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"env_{env}", ["#ffffff00", hex_color], N=256,
        )
        norm = mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 1))
        masked = np.ma.masked_where(counts == 0, counts)
        ax.pcolormesh(xedges, yedges, masked.T, cmap=cmap, norm=norm, alpha=0.75)

    _set_log_yaxis(ax)
    ax.set_xlabel("Depth (m)")
    ax.set_title("Depth–turbidity density (estuary / offshore)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=e) for e, c in ENV_PALETTE.items()]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def fig_depth_turbidity_density(df: pd.DataFrame, out_dir: Path):
    valid = df[df["turbidity_index"] > 0].copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_depth_turb_hist(ax, valid)
    _savefig(fig, out_dir / "02_depth_turbidity_density.png")


def fig_band_reflectance(ls_df: pd.DataFrame, s2_df: pd.DataFrame, out_dir: Path):
    """
    All bands (Landsat + Sentinel-2) on a single axis, sorted by central
    wavelength. Matched-wavelength bands appear side-by-side. A dashed
    separator marks the ~700 nm gap between NIR and SWIR.

    LR nodata already masked (NaN) in processed data; S2 cloud pixels
    (reflectance > 1.0) clipped to [0, 1].
    """
    if ls_df.empty and s2_df.empty:
        print("  (skipping band reflectance – no raster data)")
        return

    from matplotlib.patches import Patch

    # Build entries: (wavelength, band_name, satellite_label, data_array)
    entries = []
    for band in LANDSAT_SPECTRAL_BANDS:
        data = ls_df[ls_df["band"] == band]["reflectance"].dropna().values
        if data.size >= 2:
            entries.append((LANDSAT_WAVELENGTH[band], band, "Landsat", data))
    for band in S2_SPECTRAL_BANDS:
        data = s2_df[s2_df["band"] == band]["reflectance"].dropna().values
        if data.size >= 2:
            entries.append((S2_WAVELENGTH[band], band, "Sentinel-2", data))

    # Sort by wavelength, then Landsat before S2 so matched pairs are adjacent
    entries.sort(key=lambda e: (e[0], 0 if e[2] == "Landsat" else 1))

    positions = list(range(len(entries)))

    # Find where to draw the VNIR/SWIR separator
    vnir_last = max((i for i, e in enumerate(entries) if e[0] <= 900), default=None)
    swir_first = min((i for i, e in enumerate(entries) if e[0] >= 1600), default=None)
    gap_x = (vnir_last + swir_first) / 2 if (vnir_last is not None and swir_first is not None) else None

    fig, ax = plt.subplots(figsize=(14, 5))

    parts = ax.violinplot(
        [e[3] for e in entries],
        positions=positions,
        widths=0.72,
        showmedians=True,
        showextrema=False,
    )

    for body, (_, _, satellite, _) in zip(parts["bodies"], entries):
        color = LR_COLOR if satellite == "Landsat" else HR_COLOR
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.72)
        body.set_linewidth(0.8)

    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.8)
    parts["cmedians"].set_zorder(3)

    # x-axis: band name on top line, wavelength below
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{e[1]}\n{e[0]} nm" for e in entries],
        fontsize=7,
    )

    # VNIR / SWIR separator
    if gap_x is not None:
        ax.axvline(gap_x, color="#90A4AE", lw=1.5, ls="--", alpha=0.8)
        ylim = ax.get_ylim()
        ax.text(vnir_last + 0.3, 0.97, "VNIR ⟶", ha="right", va="top",
                fontsize=8, color="#546E7A", transform=ax.get_xaxis_transform())
        ax.text(swir_first - 0.3, 0.97, "⟵ SWIR", ha="left", va="top",
                fontsize=8, color="#546E7A", transform=ax.get_xaxis_transform())

    ax.set_ylabel("Surface reflectance [0–1]")
    ax.set_ylim(-0.01, 1.01)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_title(
        "Per-band surface reflectance – Landsat 8/9 and Sentinel-2\n"
        "(patch-mean, sorted by central wavelength; white line = median)",
        fontsize=10,
    )
    ax.legend(
        handles=[
            Patch(facecolor=LR_COLOR, alpha=0.72, label="Landsat 8/9  (30 m LR)"),
            Patch(facecolor=HR_COLOR, alpha=0.72, label="Sentinel-2   (10 m HR)"),
        ],
        fontsize=9,
    )
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    _savefig(fig, out_dir / "03_reflectance_combined.png")

    # Individual per-sensor subplots + side-by-side panels figure
    def _draw_violin(ax, df_in, order, labels, color, title):
        if df_in.empty:
            return
        plot_df = df_in[df_in["band"].isin(order)].copy()
        plot_df["band_label"] = plot_df["band"].map(labels)
        label_order = [labels[b] for b in order if b in labels]
        sns.violinplot(data=plot_df, x="band_label", y="reflectance",
                       order=label_order, color=color,
                       inner="quartile", cut=0, ax=ax, linewidth=0.7)
        ax.set_xlabel("Band (central wavelength nm)", fontsize=9)
        ax.set_ylabel("Surface reflectance [0–1]")
        ax.set_ylim(-0.01, 1.01)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    ls_title = "Landsat 8/9 – per-band surface reflectance"
    s2_title = "Sentinel-2 – per-band surface reflectance"

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_violin(ax, ls_df, LANDSAT_SPECTRAL_BANDS, LANDSAT_LABEL, LR_COLOR, ls_title)
    _savefig(fig, out_dir / "03a_reflectance_landsat.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_violin(ax, s2_df, S2_SPECTRAL_BANDS, S2_LABEL, HR_COLOR, s2_title)
    _savefig(fig, out_dir / "03b_reflectance_sentinel2.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    _draw_violin(axes[0], ls_df, LANDSAT_SPECTRAL_BANDS, LANDSAT_LABEL, LR_COLOR, ls_title)
    _draw_violin(axes[1], s2_df, S2_SPECTRAL_BANDS, S2_LABEL, HR_COLOR, s2_title)
    axes[1].set_ylabel("")
    fig.suptitle("Per-band surface reflectance (patch-mean; LR nodata masked, "
                 "S2 cloud pixels clipped to [0,1])", fontsize=11)
    _savefig(fig, out_dir / "03_reflectance_panels.png")

    # 03c – Landsat clipping rate per band
    if not ls_df.empty and "clip_frac" in ls_df.columns:
        clip_by_band = (
            ls_df.groupby("band")["clip_frac"]
            .agg(["mean", "max"])
            .reindex(LANDSAT_SPECTRAL_BANDS)
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = np.arange(len(clip_by_band))
        labels = [LANDSAT_LABEL.get(b, b) for b in clip_by_band.index]
        ax.bar(xs, clip_by_band["mean"] * 100, color=LR_COLOR, alpha=0.85,
               label="mean clip %")
        ax.bar(xs, clip_by_band["max"] * 100, color=LR_COLOR, alpha=0.3,
               label="max clip %")
        ax.axhline(5, color="#E53935", ls="--", lw=1.0, label="5% flag threshold")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Pixels with reflectance ≥ 1.0 (%)")
        ax.set_title("Landsat saturation (clipping) rate per band\n"
                     "(dark = mean across samples; light = worst sample)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")
        _savefig(fig, out_dir / "03c_clipping_rate.png")


# ── 04 Spectral difficulty ────────────────────────────────────────────────────

def _draw_spec_scatter(ax, pairs_df):
    cmap = plt.colormaps.get_cmap("tab10")
    for i, (_, grp) in enumerate(pairs_df.groupby("band_label", sort=False)):
        ax.scatter(grp["landsat"], grp["sentinel2"], s=8, alpha=0.4,
                   color=cmap(i % 10), label=grp["band_label"].iloc[0], linewidths=0)
    lim_max = pairs_df[["landsat", "sentinel2"]].max().max() * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, label="1:1")
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.set_xlabel("Landsat reflectance"); ax.set_ylabel("Sentinel-2 reflectance")
    ax.set_title("Reflectance: LR vs HR (spectrally matched bands)", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def _draw_spec_bias_band(ax, pairs_df):
    sns.boxplot(data=pairs_df, x="band_label", y="diff",
                hue="turbidity_class", hue_order=["clear", "moderate", "turbid"],
                palette=TURB_PALETTE,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                ax=ax, linewidth=0.7)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Band pair"); ax.set_ylabel("S2 − Landsat reflectance")
    ax.set_title("Cross-spectral bias by band (coloured by turbidity class)", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="turbidity", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def _draw_spec_bias_gap(ax, pairs_df):
    gap_df = pairs_df.dropna(subset=["gap_days"])
    if not gap_df.empty:
        for turb, color in TURB_PALETTE.items():
            sub = gap_df[gap_df["turbidity_class"] == turb]
            ax.scatter(sub["gap_days"], sub["diff"], c=color, s=6, alpha=0.3,
                       linewidths=0, label=turb)
        trend = gap_df.sort_values("gap_days")
        win = max(30, len(trend) // 20)
        rolled = trend["diff"].rolling(win, min_periods=5, center=True).mean()
        ax.plot(trend["gap_days"], rolled, color="k", lw=1.5, label="rolling mean")
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.6)
        ax.set_xlabel("Min |LR – HR| acquisition gap (days)")
        ax.set_ylabel("S2 − Landsat reflectance")
        ax.set_title("Cross-spectral bias vs acquisition time gap\n"
                     "(all matched bands; coloured by turbidity)", fontsize=10)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, linewidth=0.3, alpha=0.4)
    else:
        ax.text(0.5, 0.5, "No gap data available", ha="center", va="center",
                transform=ax.transAxes)


def fig_spectral_difficulty(
    ls_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    df_manifest: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path,
):
    if ls_df.empty or s2_df.empty:
        print("  (skipping spectral difficulty – missing raster data)")
        return

    ls_pivot = ls_df.pivot_table(
        index=["location_id", "turbidity_class", "environment_class", "depth_class"],
        columns="band", values="reflectance",
    ).reset_index()
    s2_pivot = s2_df.pivot_table(
        index=["location_id", "turbidity_class", "environment_class", "depth_class"],
        columns="band", values="reflectance",
    ).reset_index()

    combined = pd.merge(ls_pivot, s2_pivot, on="location_id", suffixes=("_ls", "_s2"))
    turb_col = "turbidity_class_ls" if "turbidity_class_ls" in combined else "turbidity_class"

    pair_dfs = []
    for ls_b, s2_b, label in COMPARABLE_PAIRS:
        if ls_b not in combined.columns or s2_b not in combined.columns:
            continue
        tmp = combined[["location_id", turb_col, ls_b, s2_b]].copy().dropna()
        tmp.columns = ["location_id", "turbidity_class", "landsat", "sentinel2"]
        tmp["band_label"] = label
        tmp["diff"] = tmp["sentinel2"] - tmp["landsat"]
        pair_dfs.append(tmp)

    if not pair_dfs:
        print("  (skipping spectral difficulty – no comparable band pairs found)")
        return

    pairs_df = pd.concat(pair_dfs, ignore_index=True)
    sampled_ids = list(pairs_df["location_id"].unique())
    gaps = _compute_time_gaps(df_manifest, processed_dir, sampled_ids)
    pairs_df["gap_days"] = pairs_df["location_id"].map(gaps)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_spec_scatter(ax, pairs_df)
    ax.set_aspect("equal")
    _savefig(fig, out_dir / "04a_spectral_lr_hr_scatter.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_spec_bias_band(ax, pairs_df)
    _savefig(fig, out_dir / "04b_spectral_bias_by_band.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_spec_bias_gap(ax, pairs_df)
    _savefig(fig, out_dir / "04c_spectral_bias_vs_gap.png")

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    _draw_spec_scatter(axes[0], pairs_df)
    _draw_spec_bias_band(axes[1], pairs_df)
    _draw_spec_bias_gap(axes[2], pairs_df)
    axes[1].set_ylabel(""); axes[2].set_ylabel("")
    fig.suptitle("Spectral difficulty: LR–HR reflectance comparison", fontsize=12)
    _savefig(fig, out_dir / "04_spectral_difficulty.png")


# ── 05 Class histograms ───────────────────────────────────────────────────────

def _draw_class_hist(ax, series_by_label, xlabel, title):
    for label, (data, color) in series_by_label.items():
        ax.hist(data, bins=40, color=color, alpha=0.6, label=label, edgecolor="none")
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def _draw_depth_all(ax, df):
    ax.hist(df["depth_m"], bins=50, color="#5C6BC0", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Depth (m)"); ax.set_ylabel("Count")
    ax.set_title("Depth distribution – all samples", fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def _draw_turbidity_log_all(ax, df):
    valid_turb = df["turbidity_index"].replace(0, np.nan).dropna()
    ax.hist(np.log10(valid_turb), bins=50, color="#26A69A", alpha=0.85, edgecolor="none")
    ax.set_xlabel("log₁₀(turbidity index)"); ax.set_ylabel("Count")
    ax.set_title("Turbidity index (log scale) – all samples", fontsize=10)
    for thresh, color, label in [(np.log10(0.03), "#FFA726", "clear|moderate"),
                                  (np.log10(0.08), "#E53935", "moderate|turbid")]:
        ax.axvline(thresh, color=color, lw=1.2, ls="--", label=label)
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def fig_class_histograms(df: pd.DataFrame, out_dir: Path):
    hist_specs = [
        ({e: (df[df["environment_class"] == e]["depth_m"], c)
          for e, c in ENV_PALETTE.items()},
         "Depth (m)", "Depth by environment class", "05a_hist_depth_by_env.png"),
        ({e: (df[df["environment_class"] == e]["turbidity_index"], c)
          for e, c in ENV_PALETTE.items()},
         "Turbidity index", "Turbidity by environment class", "05b_hist_turbidity_by_env.png"),
        ({t: (df[df["turbidity_class"] == t]["depth_m"], c)
          for t, c in TURB_PALETTE.items()},
         "Depth (m)", "Depth by turbidity class", "05c_hist_depth_by_turbidity.png"),
        ({DEPTH_BIN_LABELS[d]: (df[df["depth_class"] == d]["turbidity_index"], c)
          for d, c in DEPTH_PALETTE.items()},
         "Turbidity index", "Turbidity by depth class", "05d_hist_turbidity_by_depth.png"),
    ]
    for series_by_label, xlabel, title, fname in hist_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        _draw_class_hist(ax, series_by_label, xlabel, title)
        _savefig(fig, out_dir / fname)

    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_depth_all(ax, df)
    _savefig(fig, out_dir / "05e_hist_depth_all.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_turbidity_log_all(ax, df)
    _savefig(fig, out_dir / "05f_hist_turbidity_log_all.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, (series_by_label, xlabel, title, _) in zip(axes.flat, hist_specs):
        _draw_class_hist(ax, series_by_label, xlabel, title)
    _draw_depth_all(axes[1, 1], df)
    _draw_turbidity_log_all(axes[1, 2], df)
    for row_axes in axes:
        for ax in row_axes[1:]:
            ax.set_ylabel("")
    fig.suptitle("Class distribution histograms", fontsize=12)
    _savefig(fig, out_dir / "05_class_histograms.png")


# ── 06 Temporal distribution ──────────────────────────────────────────────────

def _draw_temporal_year(ax, df):
    year_counts = df["s2_year"].value_counts().sort_index()
    ax.bar(year_counts.index.astype(str), year_counts.values, color="#5C6BC0", alpha=0.85)
    ax.set_xlabel("Sentinel-2 acquisition year"); ax.set_ylabel("Count")
    ax.set_title("Temporal coverage (S2 acquisition year)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def _draw_temporal_season(ax, df):
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    sns.countplot(data=df, x="season_label", order=season_order,
                  hue="environment_class", palette=ENV_PALETTE, ax=ax, alpha=0.85)
    ax.set_xlabel("Season"); ax.set_ylabel("Count")
    ax.set_title("Season distribution by environment class", fontsize=10)
    ax.legend(title="environment", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def fig_temporal(df: pd.DataFrame, out_dir: Path):
    import re as _re
    df = df.copy()

    def _s2_year(ids_str):
        try:
            ids = json.loads(ids_str)
            if ids:
                m = _re.search(r"/(\d{8})T", ids[0])
                if m:
                    return int(m.group(1)[:4])
        except Exception:
            pass
        return None

    df["s2_year"] = df["highres_item_ids"].apply(_s2_year)
    season_labels = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    df["season_label"] = df["season_id"].map(season_labels)

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_temporal_year(ax, df)
    _savefig(fig, out_dir / "06a_temporal_year.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_temporal_season(ax, df)
    _savefig(fig, out_dir / "06b_temporal_season.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _draw_temporal_year(axes[0], df)
    _draw_temporal_season(axes[1], df)
    axes[1].set_ylabel("")
    fig.suptitle("Temporal coverage", fontsize=12)
    _savefig(fig, out_dir / "06_temporal.png")


# ── 07 Patch coverage ─────────────────────────────────────────────────────────

def _draw_coverage_hemisphere(ax, df):
    counts = df["hemisphere"].value_counts()
    ax.pie(counts.values, labels=counts.index,
           colors=["#42A5F5", "#EF5350"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Hemisphere split", fontsize=10)


def _draw_coverage_latitude(ax, df):
    ax.hist(df["latitude"], bins=40, color="#66BB6A", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Latitude (°)"); ax.set_ylabel("Count")
    ax.set_title("Latitude distribution", fontsize=10)
    ax.axvline(0, color="k", lw=1.0, ls="--", label="equator")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def fig_patch_coverage(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["hemisphere"] = np.where(df["latitude"] >= 0, "Northern", "Southern")

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_coverage_hemisphere(ax, df)
    _savefig(fig, out_dir / "07a_coverage_hemisphere.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_coverage_latitude(ax, df)
    _savefig(fig, out_dir / "07b_coverage_latitude.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _draw_coverage_hemisphere(axes[0], df)
    _draw_coverage_latitude(axes[1], df)
    fig.suptitle("Geographic coverage", fontsize=12)
    _savefig(fig, out_dir / "07_patch_coverage.png")


def fig_psd_curves(processed_dir: Path, df: pd.DataFrame, n_samples: int, out_dir: Path):
    """
    Radial-averaged PSD for HR (10 m), LR (30 m), and bicubically upsampled LR.
    LR Nyquist = 16.7 cyc/km; HR Nyquist = 50 cyc/km.
    """
    import rasterio

    lr_psds, hr_psds, bic_psds = [], [], []
    hr_pixel_m, lr_pixel_m = 10.0, 30.0

    rows = df.sample(n=min(n_samples, len(df)), random_state=7)

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="  PSD", leave=False):
        loc_id = int(row["location_id"])
        sample = f"sample_{loc_id:06d}"
        lr_files = sorted((processed_dir / sample / "landsat").glob("*.tif"))
        hr_files = sorted((processed_dir / sample / "sentinel2").glob("*.tif"))
        if not lr_files or not hr_files:
            continue
        try:
            with rasterio.open(lr_files[0]) as src:
                lr_patch = src.read(LR_SPATIAL_BAND_IDX).astype(np.float32)
            with rasterio.open(hr_files[0]) as src:
                hr_patch = src.read(HR_SPATIAL_BAND_IDX).astype(np.float32)
        except Exception:
            continue
        if lr_patch.size == 0 or hr_patch.size == 0:
            continue
        if np.all(lr_patch == 0) or np.all(hr_patch == 0):
            continue
        scale_factor = hr_patch.shape[0] / lr_patch.shape[0]
        bic_patch = zoom(lr_patch, scale_factor, order=3)
        h, w = hr_patch.shape
        bic_patch = bic_patch[:h, :w]
        _, hr_psd  = _radial_psd(hr_patch)
        _, lr_psd  = _radial_psd(lr_patch)
        _, bic_psd = _radial_psd(bic_patch)
        hr_psds.append(hr_psd); lr_psds.append(lr_psd); bic_psds.append(bic_psd)

    if not hr_psds:
        print("  (skipping PSD – no valid patches)")
        return

    def _freq_to_cykm(n, pixel_m):
        return np.arange(n) / (2 * n) / pixel_m * 1000.0

    hr_len  = min(len(p) for p in hr_psds)
    lr_len  = min(len(p) for p in lr_psds)
    bic_len = min(len(p) for p in bic_psds)
    hr_arr  = np.stack([p[:hr_len]  for p in hr_psds])
    lr_arr  = np.stack([p[:lr_len]  for p in lr_psds])
    bic_arr = np.stack([p[:bic_len] for p in bic_psds])

    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot(arr, pixel_m, color, label):
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        freqs = _freq_to_cykm(arr.shape[1], pixel_m)
        ax.loglog(freqs[1:], mean[1:], color=color, lw=1.8, label=label)
        ax.fill_between(freqs[1:], np.maximum(mean[1:] - std[1:], 1e-12),
                        mean[1:] + std[1:], alpha=0.2, color=color)

    _plot(hr_arr,  hr_pixel_m, HR_COLOR,  f"Sentinel-2 HR (10 m)  n={len(hr_psds)}")
    _plot(bic_arr, hr_pixel_m, "#78909C", "Bicubic↑ LR (10 m equiv.)")
    _plot(lr_arr,  lr_pixel_m, LR_COLOR,  f"Landsat LR (30 m)  n={len(lr_psds)}")

    lr_nyquist = 1000.0 / (2 * lr_pixel_m)
    ax.axvline(lr_nyquist, color=LR_COLOR, ls=":", lw=1.2, alpha=0.8,
               label=f"LR Nyquist ({lr_nyquist:.1f} cyc/km)")
    ax.set_xlabel("Spatial frequency (cycles / km)", fontsize=11)
    ax.set_ylabel("Power spectral density (per pixel, log)", fontsize=11)
    ax.set_title("Radial-averaged PSD: LR, HR, and bicubically upsampled LR\n"
                 "(band 4 / red; mean ± 1 SD across samples)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    _savefig(fig, out_dir / "08_psd_curves.png")


# ── 09 LR/HR paired histograms ────────────────────────────────────────────────

def _draw_lr_hr_hist(ax, lr_arr, hr_arr, ls_b, s2_b, label, lr_clip_frac=0.0):
    bins = np.linspace(0, 1, 60)
    if lr_arr.size:
        ax.hist(lr_arr, bins=bins, color=LR_COLOR, alpha=0.6, density=True,
                label=f"LR  {ls_b} (Landsat)", edgecolor="none")
    if hr_arr.size:
        ax.hist(hr_arr, bins=bins, color=HR_COLOR, alpha=0.6, density=True,
                label=f"HR  {s2_b} (Sentinel-2)", edgecolor="none")
    if lr_clip_frac > 0:
        ax.axvline(1.0, color="#E53935", ls="--", lw=1.0)
        ax.text(0.99, 0.97, f"LR clipped ≥1.0:\n{lr_clip_frac:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#E53935",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E53935", alpha=0.8))
    ax.set_xlabel("Surface reflectance"); ax.set_ylabel("Density")
    ax.set_title(f"LR vs HR pixel intensity – {label}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def fig_lr_hr_histograms(processed_dir: Path, df: pd.DataFrame, n_samples: int, out_dir: Path):
    """One file per matched band pair plus a combined 2×4 panel figure."""
    import rasterio

    lr_band_idx = {ls_b: LANDSAT_SPECTRAL_BANDS.index(ls_b) + 1
                   for ls_b, _, _ in COMPARABLE_PAIRS if ls_b in LANDSAT_SPECTRAL_BANDS}
    hr_band_idx = {s2_b: S2_SPECTRAL_BANDS.index(s2_b) + 1
                   for _, s2_b, _ in COMPARABLE_PAIRS if s2_b in S2_SPECTRAL_BANDS}

    ls_pixels: dict[str, list]      = {b: [] for b, _, _ in COMPARABLE_PAIRS}
    ls_clip_counts: dict[str, list] = {b: [] for b, _, _ in COMPARABLE_PAIRS}
    s2_pixels: dict[str, list]      = {s: [] for _, s, _ in COMPARABLE_PAIRS}

    rows = df.sample(n=min(n_samples, len(df)), random_state=13)
    cap = max(1, 50_000 // n_samples)

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="  histograms", leave=False):
        loc_id = int(row["location_id"])
        sample = f"sample_{loc_id:06d}"
        lr_files = sorted((processed_dir / sample / "landsat").glob("*.tif"))
        hr_files = sorted((processed_dir / sample / "sentinel2").glob("*.tif"))
        if not lr_files or not hr_files:
            continue
        try:
            with rasterio.open(lr_files[0]) as src:
                for ls_b, _, _ in COMPARABLE_PAIRS:
                    bidx = lr_band_idx.get(ls_b)
                    if bidx and bidx <= src.count:
                        d = src.read(bidx).astype(np.float32).ravel()
                        if src.nodata is not None:
                            d = d[d != src.nodata]
                        d = d[np.isfinite(d)]
                        ls_clip_counts[ls_b].append(float((d >= 1.0).mean()) if d.size else 0.0)
                        d = np.clip(d, 0, 1)
                        ls_pixels[ls_b].extend(d[:cap].tolist())
            with rasterio.open(hr_files[0]) as src:
                for _, s2_b, _ in COMPARABLE_PAIRS:
                    bidx = hr_band_idx.get(s2_b)
                    if bidx and bidx <= src.count:
                        d = src.read(bidx).astype(np.float32).ravel()
                        if src.nodata is not None:
                            d = d[d != src.nodata]
                        d = np.clip(d[np.isfinite(d)], 0, 1)
                        s2_pixels[s2_b].extend(d[:cap].tolist())
        except Exception:
            continue

    letters = "abcdefg"
    clip_fracs = {ls_b: float(np.mean(ls_clip_counts[ls_b])) if ls_clip_counts[ls_b] else 0.0
                  for ls_b, _, _ in COMPARABLE_PAIRS}
    pair_arrays = [(np.array(ls_pixels[ls_b]), np.array(s2_pixels[s2_b]), ls_b, s2_b, label)
                   for ls_b, s2_b, label in COMPARABLE_PAIRS]

    for i, (lr_arr, hr_arr, ls_b, s2_b, label) in enumerate(pair_arrays):
        fig, ax = plt.subplots(figsize=(6, 4))
        _draw_lr_hr_hist(ax, lr_arr, hr_arr, ls_b, s2_b, label, lr_clip_frac=clip_fracs[ls_b])
        fname = f"09{letters[i]}_lr_hr_hist_{label.replace(' ', '').replace('~', '')}.png"
        _savefig(fig, out_dir / fname)

    fig, axes = plt.subplots(2, 4, figsize=(24, 8))
    for i, (lr_arr, hr_arr, ls_b, s2_b, label) in enumerate(pair_arrays):
        row, col = divmod(i, 4)
        _draw_lr_hr_hist(axes[row, col], lr_arr, hr_arr, ls_b, s2_b, label,
                         lr_clip_frac=clip_fracs[ls_b])
    axes[1, 3].set_visible(False)
    for row_axes in axes:
        for ax in row_axes[1:]:
            ax.set_ylabel("")
    fig.suptitle("Paired LR / HR pixel intensity histograms (density)", fontsize=12)
    _savefig(fig, out_dir / "09_lr_hr_histograms.png")


# ── 10 Alignment quality ──────────────────────────────────────────────────────

def _draw_align_scatter(ax, sx, sy, stats):
    ax.scatter(sx, sy, s=10, alpha=0.4, color="#546E7A", linewidths=0)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Shift x (HR pixels)"); ax.set_ylabel("Shift y (HR pixels)")
    ax.set_title(f"Registration shift – 2-D distribution\n{stats}", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.4)


_ALIGN_THRESHOLD_PX = 3   # flag threshold used in check_dataset


def _draw_align_magnitude(ax, mag):
    ax.hist(mag, bins=40, color="#546E7A", alpha=0.85, edgecolor="none")
    med = np.median(mag)
    ax.axvline(med, color="k", ls="--", lw=1.2, label=f"median {med:.2f} px")
    bad_frac = (mag > _ALIGN_THRESHOLD_PX).mean() * 100
    ax.axvline(_ALIGN_THRESHOLD_PX, color="#E53935", ls="--", lw=1.2,
               label=f">{_ALIGN_THRESHOLD_PX} px: {bad_frac:.1f}% of samples")
    ax.set_xlabel("Shift magnitude (HR pixels)"); ax.set_ylabel("Count")
    ax.set_title("Registration shift magnitude", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")


def _draw_align_cdf(ax, mag):
    sorted_mag = np.sort(mag)
    cdf = np.arange(1, len(sorted_mag) + 1) / len(sorted_mag)
    ax.plot(sorted_mag, cdf * 100, color="#546E7A", lw=2)
    for threshold, color in [(1.0, "gray"), (_ALIGN_THRESHOLD_PX, "#E53935"), (10.0, "#B71C1C")]:
        frac = (mag <= threshold).mean() * 100
        ax.axvline(threshold, color=color, ls=":", lw=0.9)
        ax.text(threshold + 0.03, frac - 3, f"{frac:.0f}%\n≤{threshold}px",
                fontsize=7, va="top", color=color)
    ax.set_xlabel("Shift magnitude (HR pixels)"); ax.set_ylabel("Cumulative %")
    ax.set_title("Registration shift – CDF", fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4); ax.set_ylim(0, 105)


def fig_alignment_quality(processed_dir: Path, df: pd.DataFrame, n_samples: int, out_dir: Path):
    """
    Phase-correlation shift between bicubically upsampled LR and HR.
    (0, 0) = perfect pixel alignment.
    """
    import rasterio

    shifts_x, shifts_y, shift_ids = [], [], []
    rows = df.sample(n=min(n_samples, len(df)), random_state=99)

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="  alignment", leave=False):
        loc_id = int(row["location_id"])
        sample = f"sample_{loc_id:06d}"
        lr_files = sorted((processed_dir / sample / "landsat").glob("*.tif"))
        hr_files = sorted((processed_dir / sample / "sentinel2").glob("*.tif"))
        if not lr_files or not hr_files:
            continue
        try:
            with rasterio.open(lr_files[0]) as src:
                lr_raw = src.read(LR_SPATIAL_BAND_IDX).astype(np.float32)
            with rasterio.open(hr_files[0]) as src:
                hr_raw = src.read(HR_SPATIAL_BAND_IDX).astype(np.float32)
        except Exception:
            continue
        if hr_raw.size == 0 or lr_raw.size == 0:
            continue
        hr = hr_raw.copy()
        bic = zoom(lr_raw, hr.shape[0] / lr_raw.shape[0], order=3)
        bic = bic[:hr.shape[0], :hr.shape[1]]
        if bic.shape != hr.shape:
            continue
        for arr in (hr, bic):
            arr -= arr.mean()
            if arr.std() > 0:
                arr /= arr.std()
        cross = np.fft.fft2(hr) * np.conj(np.fft.fft2(bic))
        corr = np.abs(np.fft.ifft2(cross / (np.abs(cross) + 1e-12)))
        corr = np.fft.fftshift(corr)
        h, w = corr.shape
        peak = np.unravel_index(corr.argmax(), corr.shape)
        dy, dx = peak[0] - h // 2, peak[1] - w // 2
        shifts_x.append(float(dx)); shifts_y.append(float(dy)); shift_ids.append(loc_id)

    if not shifts_x:
        print("  (skipping alignment – no valid pairs)")
        return

    sx, sy = np.array(shifts_x), np.array(shifts_y)
    mag = np.hypot(sx, sy)
    bad_frac = (mag > _ALIGN_THRESHOLD_PX).mean() * 100
    stats = (f"n={len(sx)}  median={np.median(mag):.2f} px  "
             f"p95={np.percentile(mag, 95):.2f} px  "
             f">{_ALIGN_THRESHOLD_PX} px: {bad_frac:.1f}%  (1 HR px = 10 m)")

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_align_scatter(ax, sx, sy, stats)
    ax.set_aspect("equal")
    _savefig(fig, out_dir / "10a_alignment_scatter.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_align_magnitude(ax, mag)
    _savefig(fig, out_dir / "10b_alignment_magnitude.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_align_cdf(ax, mag)
    _savefig(fig, out_dir / "10c_alignment_cdf.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_align_scatter(axes[0], sx, sy, stats)
    _draw_align_magnitude(axes[1], mag)
    _draw_align_cdf(axes[2], mag)
    axes[1].set_ylabel(""); axes[2].set_ylabel("")
    fig.suptitle("Registration alignment quality", fontsize=12)
    _savefig(fig, out_dir / "10_alignment_quality.png")

    shifts_json = out_dir / "10_alignment_shifts.json"
    import json as _json
    shifts_json.write_text(_json.dumps(
        [{"location_id": lid, "shift_x": dx, "shift_y": dy,
          "shift_magnitude_px": round(float(np.hypot(dx, dy)), 3)}
         for lid, dx, dy in zip(shift_ids, sx.tolist(), sy.tolist())],
        indent=2,
    ))
    print(f"  shifts written → {shifts_json.name}")


# ── 11 Sample thumbnails ─────────────────────────────────────────────────────

# 1-based rasterio band indices for R / G / B
_LR_RGB = (4, 3, 2)   # Landsat: SR_B4 / SR_B3 / SR_B2
_HR_RGB = (4, 3, 2)   # Sentinel-2: B4 / B3 / B2


def _load_rgb(sat_dir: Path, r: int, g: int, b: int) -> "np.ndarray | None":
    """Read three bands, apply 2–98th-percentile stretch, return (H,W,3) uint8 or None."""
    import rasterio as _rio
    files = sorted(sat_dir.glob("*.tif"))
    if not files:
        return None
    try:
        with _rio.open(files[0]) as src:
            if src.count < max(r, g, b):
                return None
            bands = np.stack([src.read(i).astype(np.float32) for i in (r, g, b)], axis=-1)
    except Exception:
        return None
    finite = np.isfinite(bands)
    for c in range(3):
        ch = bands[:, :, c]
        valid = ch[finite[:, :, c]]
        if valid.size >= 2:
            lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
            if hi > lo:
                bands[:, :, c] = (ch - lo) / (hi - lo)
    bands[~finite] = 0
    return (np.clip(bands, 0, 1) * 255).astype(np.uint8)


def _render_sample_row(axes, processed_dir: Path, loc_id: int, tag: str, meta: dict):
    """Fill a 2-element axes row with LR and HR RGB for one sample."""
    sample = f"sample_{loc_id:06d}"
    lr_rgb = _load_rgb(processed_dir / sample / "landsat",   *_LR_RGB)
    hr_rgb = _load_rgb(processed_dir / sample / "sentinel2", *_HR_RGB)
    env   = meta.get("environment_class", "?")
    depth = meta.get("depth_class", "?")
    turb  = meta.get("turbidity_class", "?")

    for ax, rgb, sensor in zip(axes, [lr_rgb, hr_rgb], ["Landsat LR 30 m", "Sentinel-2 HR 10 m"]):
        if rgb is not None:
            ax.imshow(rgb)
        else:
            ax.set_facecolor("#222222")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="white")
        ax.axis("off")

    # Overlay info text on the LR panel
    info = f"#{loc_id}  {env} / {depth} / {turb}"
    if tag:
        info += f"\n{tag}"
    axes[0].text(0.02, 0.98, info,
                 transform=axes[0].transAxes, fontsize=6.5,
                 va="top", ha="left", color="white",
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55, ec="none"))
    # Sensor labels as small overlaid text on each panel
    for ax, label in zip(axes, ["Landsat LR 30 m", "Sentinel-2 HR 10 m"]):
        ax.text(0.98, 0.02, label,
                transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, ec="none"))


def _save_thumbnail_grid(
    samples: list[dict],
    processed_dir: Path,
    df: pd.DataFrame,
    path: Path,
    suptitle: str,
):
    """Render a grid (one row per sample, LR | HR columns) and save to path."""
    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.0 * n),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    if n == 1:
        axes = axes[None, :]
    meta_lookup = df.set_index("location_id").to_dict("index")
    for i, s in enumerate(samples):
        meta = meta_lookup.get(s["location_id"], {})
        _render_sample_row(axes[i], processed_dir, s["location_id"], s.get("tag", ""), meta)
    suptitle_height = 0.28 / (3.0 * n)   # fraction of figure height for title row
    fig.subplots_adjust(top=1.0 - suptitle_height, bottom=0.0, left=0.0, right=1.0)
    fig.suptitle(suptitle, fontsize=9, y=1.0 - suptitle_height * 0.15)
    fig.patch.set_facecolor("black")
    path.parent.mkdir(parents=True, exist_ok=True)
    _savefig(fig, path)


def fig_sample_thumbnails(
    processed_dir: Path,
    df: pd.DataFrame,
    out_dir: Path,
    n_random: int = 9,
    n_interesting: int = 6,
    ls_df: "pd.DataFrame | None" = None,
    alignment_json: "Path | None" = None,
):
    """Save RGB thumbnail grids (LR | HR) to out_dir/thumbnails/.

    Three grids are produced:
      11a  thumbnails_random.png       – random sample
      11b  thumbnails_bad_alignment.png – worst phase-correlation shifts
      11c  thumbnails_clipping.png      – worst Landsat saturation
    """
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)

    # ── random ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=77)
    rand_ids = rng.choice(df["location_id"].values, size=min(n_random, len(df)),
                          replace=False).tolist()
    _save_thumbnail_grid(
        [{"location_id": int(i), "tag": "random"} for i in rand_ids],
        processed_dir, df,
        thumb_dir / "11a_thumbnails_random.png",
        "Random sample pairs (Landsat LR  |  Sentinel-2 HR)",
    )

    # ── bad alignment ─────────────────────────────────────────────────────────
    if alignment_json and alignment_json.exists():
        import json as _json
        shifts = _json.loads(alignment_json.read_text())
        shifts.sort(key=lambda x: x["shift_magnitude_px"], reverse=True)
        bad_align = [
            {"location_id": s["location_id"],
             "tag": f"shift {s['shift_magnitude_px']:.0f} px = {s['shift_magnitude_px']*10:.0f} m"}
            for s in shifts[:n_interesting]
            if (processed_dir / f"sample_{s['location_id']:06d}").exists()
        ]
        if bad_align:
            _save_thumbnail_grid(
                bad_align, processed_dir, df,
                thumb_dir / "11b_thumbnails_bad_alignment.png",
                f"Worst alignment shifts (largest phase-correlation offset)",
            )

    # ── high clipping ─────────────────────────────────────────────────────────
    if ls_df is not None and not ls_df.empty and "clip_frac" in ls_df.columns:
        worst_clip = (
            ls_df.groupby("location_id")["clip_frac"]
            .max()
            .nlargest(n_interesting)
            .reset_index()
        )
        clip_samples = [
            {"location_id": int(row["location_id"]),
             "tag": f"max clip {row['clip_frac']:.1%}"}
            for _, row in worst_clip.iterrows()
            if (processed_dir / f"sample_{int(row['location_id']):06d}").exists()
        ]
        if clip_samples:
            _save_thumbnail_grid(
                clip_samples, processed_dir, df,
                thumb_dir / "11c_thumbnails_clipping.png",
                "Worst Landsat saturation (highest clip fraction across bands)",
            )


# ─── driver ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", help="Path to dataset_manifest.csv")
    parser.add_argument("data_dir", help="Path to dataset data directory (sample_* folders)")
    parser.add_argument("--processed-dir", default=None,
                        help="Path to postprocessed data (default: <data_dir>/../processed)")
    parser.add_argument("--output", "-o", default="figures/dataset_analysis",
                        help="Output directory for figures")
    parser.add_argument("--n-band-samples", type=int, default=150,
                        help="Random samples for band / PSD / histogram analysis (default: 150)")
    parser.add_argument("--skip-rasters", action="store_true",
                        help="Skip all raster-based figures (figs 3–4, 8–10)")
    parser.add_argument("--images", action="store_true",
                        help="Only generate sample thumbnails (fig 11); skip all other figures")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir) if args.processed_dir else Path(args.data_dir).parent / "processed"

    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    print(f"  {len(df):,} samples\n")

    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    warnings.filterwarnings("ignore", category=UserWarning)

    if args.images:
        print("[11] Sample thumbnails")
        ls_df = pd.DataFrame()
        if processed_dir.exists():
            ls_df = _sample_reflectance(
                processed_dir, df, "landsat",
                LANDSAT_SPECTRAL_BANDS, list(range(1, len(LANDSAT_SPECTRAL_BANDS) + 1)),
                args.n_band_samples,
            )
        fig_sample_thumbnails(
            processed_dir, df, out_dir,
            ls_df=ls_df if not ls_df.empty else None,
            alignment_json=out_dir / "10_alignment_shifts.json",
        )
        print(f"\nDone. Thumbnails written to {out_dir / 'thumbnails'}/")
        return

    print("[01] Spatial distribution map")
    fig_spatial_distribution(df, out_dir)

    print("[02] Depth–turbidity density")
    fig_depth_turbidity_density(df, out_dir)

    ls_df = pd.DataFrame()
    s2_df = pd.DataFrame()
    if not args.skip_rasters and processed_dir.exists():
        n = args.n_band_samples
        print(f"[03] Per-band reflectance  ({n} samples each satellite)")
        ls_df = _sample_reflectance(
            processed_dir, df, "landsat",
            LANDSAT_SPECTRAL_BANDS, list(range(1, len(LANDSAT_SPECTRAL_BANDS) + 1)), n,
        )
        s2_df = _sample_reflectance(
            processed_dir, df, "sentinel2",
            S2_SPECTRAL_BANDS, list(range(1, len(S2_SPECTRAL_BANDS) + 1)), n,
        )
        fig_band_reflectance(ls_df, s2_df, out_dir)

        print("[04] Spectral difficulty + acquisition time gap")
        fig_spectral_difficulty(ls_df, s2_df, df, processed_dir, out_dir)
    else:
        reason = "skip_rasters flag" if args.skip_rasters else f"processed dir not found: {processed_dir}"
        print(f"[03] Per-band reflectance  (skipped: {reason})")
        print(f"[04] Spectral difficulty   (skipped)")

    print("[05] Class histograms")
    fig_class_histograms(df, out_dir)

    print("[06] Temporal distribution")
    fig_temporal(df, out_dir)

    print("[07] Patch coverage")
    fig_patch_coverage(df, out_dir)

    if not args.skip_rasters and processed_dir.exists():
        print(f"[08] PSD curves  ({args.n_band_samples} samples)")
        fig_psd_curves(processed_dir, df, args.n_band_samples, out_dir)

        print(f"[09] LR vs HR paired histograms  ({args.n_band_samples} samples)")
        fig_lr_hr_histograms(processed_dir, df, args.n_band_samples, out_dir)

        print(f"[10] Alignment quality  ({args.n_band_samples} samples)")
        fig_alignment_quality(processed_dir, df, args.n_band_samples, out_dir)

        print("[11] Sample thumbnails")
        fig_sample_thumbnails(
            processed_dir, df, out_dir,
            ls_df=ls_df if not ls_df.empty else None,
            alignment_json=out_dir / "10_alignment_shifts.json",
        )
    else:
        print("[08–11] Raster figures skipped")

    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
