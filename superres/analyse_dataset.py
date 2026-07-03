"""
Dataset analysis for the super-resolution training set.

Produces a suite of figures saved to <output_dir>/:
  01_spatial_distribution.png      – world map coloured by class
  02_depth_turbidity_density.png   – joint density (log-turbidity y-axis)
  03_band_reflectance.png          – per-band reflectance, Landsat & S2
  04_spectral_difficulty.png       – LR vs HR bias, + bias vs acquisition gap
  05_class_histograms.png          – histograms per class
  06_temporal.png                  – year and season coverage
  07_patch_coverage.png            – hemisphere split and latitude distribution
  08_psd_curves.png                – power spectral density: LR, HR, bicubic LR
  09_lr_hr_histograms.png          – paired LR vs HR intensity histograms
  10_alignment_quality.png         – phase-correlation registration shift

Usage
-----
  uv run python -m superres.analyse_dataset \\
      data/landsat2sentinel/metadata/dataset_manifest.csv \\
      data/landsat2sentinel/data \\
      --processed-dir superres/processed \\
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
import matplotlib.gridspec as gridspec
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
                    valid = np.clip(valid, 0.0, 1.0)
                    if valid.size == 0:
                        continue
                    records.append({
                        "location_id": loc_id,
                        "band": bname,
                        "reflectance": float(np.mean(valid)),
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
    2D FFT → radial-averaged power spectral density.

    Returns (frequencies, psd) where frequencies are in units of
    cycles per pixel (0 to 0.5 Nyquist).
    """
    win = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
    f = np.fft.fftshift(np.fft.fft2(patch * win))
    power = np.abs(f) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy).astype(int)
    r_max = min(cx, cy)
    freqs = np.arange(r_max) / (2 * r_max)   # cycles per pixel, 0..0.5
    psd = np.array([power[r == i].mean() if (r == i).any() else 0.0 for i in range(r_max)])
    return freqs, psd


# ─── figure functions ─────────────────────────────────────────────────────────

def fig_spatial_distribution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    columns = [
        ("environment_class", ENV_PALETTE,   "Environment class"),
        ("depth_class",       DEPTH_PALETTE, "Depth class"),
        ("turbidity_class",   TURB_PALETTE,  "Turbidity class"),
    ]

    for ax, (col, palette, title) in zip(axes, columns):
        if HAS_GEOPANDAS:
            try:
                import geodatasets
                world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
                world.boundary.plot(ax=ax, linewidth=0.4, color="#888888")
            except Exception:
                pass
        for label, color in palette.items():
            sub = df[df[col] == label]
            ax.scatter(sub["longitude"], sub["latitude"],
                       c=color, s=6, alpha=0.55, linewidths=0, label=label)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if ax is axes[0] else "")
        ax.legend(fontsize=8, markerscale=2, framealpha=0.7)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle(f"Spatial distribution of {len(df):,} dataset patches", fontsize=13)
    _savefig(fig, out_dir / "01_spatial_distribution.png")


def fig_depth_turbidity_density(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    valid = df[df["turbidity_index"] > 0].copy()
    log_turb = np.log10(valid["turbidity_index"])
    log_ticks = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])

    ax = axes[0]
    for env, color in ENV_PALETTE.items():
        sub = valid[valid["environment_class"] == env]
        ax.scatter(sub["depth_m"], np.log10(sub["turbidity_index"]),
                   c=color, s=5, alpha=0.35, linewidths=0, label=env)
    try:
        tmp = valid.copy()
        tmp["log_turbidity"] = log_turb
        sns.kdeplot(data=tmp, x="depth_m", y="log_turbidity",
                    hue="environment_class", palette=ENV_PALETTE,
                    levels=5, linewidths=1.0, ax=ax, legend=False)
    except Exception:
        pass
    for thresh, color, label in [(0.03, "#FFA726", "clear|moderate"),
                                  (0.08, "#E53935", "moderate|turbid")]:
        ax.axhline(np.log10(thresh), color=color, lw=1.0, ls="--", alpha=0.8, label=label)
    ax.set_yticks(log_ticks)
    ax.set_yticklabels([f"{10**t:.3g}" for t in log_ticks])
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Turbidity index (log scale)")
    ax.set_title("Depth–turbidity joint density\n(coloured by environment)", fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    ax2 = axes[1]
    turb_vals = valid["turbidity_index"]
    log_bins = np.linspace(np.log10(turb_vals.min()), np.log10(turb_vals.max()), 31)
    h = ax2.hist2d(valid["depth_m"], np.log10(turb_vals),
                   bins=[30, log_bins], cmap="YlOrRd", norm=mcolors.LogNorm(vmin=1))
    plt.colorbar(h[3], ax=ax2, label="count (log scale)")
    for thresh, color in [(0.03, "#FFA726"), (0.08, "#E53935")]:
        ax2.axhline(np.log10(thresh), color=color, lw=1.0, ls="--", alpha=0.8)
    ax2.set_yticks(log_ticks)
    ax2.set_yticklabels([f"{10**t:.3g}" for t in log_ticks])
    ax2.set_xlabel("Depth (m)")
    ax2.set_ylabel("Turbidity index (log scale)")
    ax2.set_title("Depth–turbidity 2-D histogram", fontsize=10)
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle("Depth and turbidity coverage", fontsize=12)
    _savefig(fig, out_dir / "02_depth_turbidity_density.png")


def fig_band_reflectance(ls_df: pd.DataFrame, s2_df: pd.DataFrame, out_dir: Path):
    """
    Two-panel combined reflectance figure: Landsat (left) and Sentinel-2 (right).

    Values are from postprocessed data (Landsat nodata already NaN-masked).
    Sentinel-2 cloud pixels (SCL 8/9) can have reflectance > 1.0; these are
    clipped to [0, 1] for display — they are not corrupt, just unmasked cloud.
    """
    if ls_df.empty and s2_df.empty:
        print("  (skipping band reflectance – no raster data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    def _violin(df_in, order, labels, color, ax, xlabel):
        plot_df = df_in[df_in["band"].isin(order)].copy()
        plot_df["band_label"] = plot_df["band"].map(labels)
        label_order = [labels[b] for b in order if b in labels]
        sns.violinplot(data=plot_df, x="band_label", y="reflectance",
                       order=label_order, color=color,
                       inner="quartile", cut=0, ax=ax, linewidth=0.7)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Surface reflectance [0–1]")
        ax.set_ylim(-0.01, 1.01)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    if not ls_df.empty:
        _violin(ls_df, LANDSAT_SPECTRAL_BANDS, LANDSAT_LABEL, LR_COLOR,
                axes[0], "Landsat 8/9 band  (central wavelength nm)")
        axes[0].set_title("Landsat 8/9 – per-band surface reflectance", fontsize=10)

    if not s2_df.empty:
        _violin(s2_df, S2_SPECTRAL_BANDS, S2_LABEL, HR_COLOR,
                axes[1], "Sentinel-2 band  (central wavelength nm)")
        axes[1].set_title("Sentinel-2 – per-band surface reflectance", fontsize=10)
        axes[1].set_ylabel("")

    fig.suptitle(
        "Per-band patch-mean reflectance  "
        "(LR nodata=NaN masked; S2 cloud pixels clipped to [0,1])",
        fontsize=11,
    )
    _savefig(fig, out_dir / "03_band_reflectance.png")


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
    if turb_col not in combined:
        turb_col = "turbidity_class"

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

    # Compute acquisition time gaps for sampled location IDs
    sampled_ids = list(pairs_df["location_id"].unique())
    gaps = _compute_time_gaps(df_manifest, processed_dir, sampled_ids)
    pairs_df["gap_days"] = pairs_df["location_id"].map(gaps)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # Panel 1: Landsat vs S2 scatter per matched band
    ax = axes[0]
    cmap = plt.colormaps.get_cmap("tab10")
    for i, (_, grp) in enumerate(pairs_df.groupby("band_label", sort=False)):
        ax.scatter(grp["landsat"], grp["sentinel2"], s=8, alpha=0.4,
                   color=cmap(i % 10), label=grp["band_label"].iloc[0], linewidths=0)
    lim_max = pairs_df[["landsat", "sentinel2"]].max().max() * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, label="1:1")
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.set_xlabel("Landsat reflectance"); ax.set_ylabel("Sentinel-2 reflectance")
    ax.set_title("Reflectance: LR vs HR\n(spectrally matched bands)", fontsize=10)
    ax.legend(fontsize=7, markerscale=2); ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Panel 2: S2 – Landsat bias boxplot by band, hued by turbidity
    ax2 = axes[1]
    sns.boxplot(data=pairs_df, x="band_label", y="diff",
                hue="turbidity_class", hue_order=["clear", "moderate", "turbid"],
                palette=TURB_PALETTE,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                ax=ax2, linewidth=0.7)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Band pair"); ax2.set_ylabel("S2 − Landsat reflectance")
    ax2.set_title("Cross-spectral bias by band\n(coloured by turbidity)", fontsize=10)
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(title="turbidity", fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    # Panel 3: bias vs LR–HR acquisition time gap, coloured by turbidity
    ax3 = axes[2]
    gap_df = pairs_df.dropna(subset=["gap_days"])
    if not gap_df.empty:
        for turb, color in TURB_PALETTE.items():
            sub = gap_df[gap_df["turbidity_class"] == turb]
            ax3.scatter(sub["gap_days"], sub["diff"], c=color, s=6, alpha=0.3,
                        linewidths=0, label=turb)
        # smoothed trend (rolling mean over sorted gap)
        trend = gap_df.sort_values("gap_days")
        win = max(30, len(trend) // 20)
        rolled = trend["diff"].rolling(win, min_periods=5, center=True).mean()
        ax3.plot(trend["gap_days"], rolled, color="k", lw=1.5, label="rolling mean")
        ax3.axhline(0, color="k", lw=0.6, ls="--", alpha=0.6)
        ax3.set_xlabel("Min |LR – HR| acquisition gap (days)")
        ax3.set_ylabel("S2 − Landsat reflectance")
        ax3.set_title("Bias vs acquisition time gap\n(all matched bands; coloured by turbidity)", fontsize=10)
        ax3.legend(fontsize=8, markerscale=2)
        ax3.grid(True, linewidth=0.3, alpha=0.4)
    else:
        ax3.text(0.5, 0.5, "No gap data available", ha="center", va="center",
                 transform=ax3.transAxes)

    fig.suptitle("Cross-spectral difficulty proxy", fontsize=12)
    _savefig(fig, out_dir / "04_spectral_difficulty.png")


def fig_class_histograms(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for env, color in ENV_PALETTE.items():
        ax.hist(df[df["environment_class"] == env]["depth_m"], bins=40,
                color=color, alpha=0.6, label=env, edgecolor="none")
    ax.set_xlabel("Depth (m)"); ax.set_ylabel("Count")
    ax.set_title("Depth by environment", fontsize=10); ax.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for env, color in ENV_PALETTE.items():
        ax2.hist(df[df["environment_class"] == env]["turbidity_index"], bins=40,
                 color=color, alpha=0.6, label=env, edgecolor="none")
    ax2.set_xlabel("Turbidity index"); ax2.set_title("Turbidity by environment", fontsize=10)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    for tc, color in TURB_PALETTE.items():
        ax3.hist(df[df["turbidity_class"] == tc]["depth_m"], bins=40,
                 color=color, alpha=0.6, label=tc, edgecolor="none")
    ax3.set_xlabel("Depth (m)"); ax3.set_title("Depth by turbidity class", fontsize=10)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    for dc, color in DEPTH_PALETTE.items():
        ax4.hist(df[df["depth_class"] == dc]["turbidity_index"], bins=40,
                 color=color, alpha=0.6, label=DEPTH_BIN_LABELS[dc], edgecolor="none")
    ax4.set_xlabel("Turbidity index"); ax4.set_ylabel("Count")
    ax4.set_title("Turbidity by depth class", fontsize=10); ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df["depth_m"], bins=50, color="#5C6BC0", alpha=0.85, edgecolor="none")
    ax5.set_xlabel("Depth (m)"); ax5.set_title("Depth – all samples", fontsize=10)

    ax6 = fig.add_subplot(gs[1, 2])
    valid_turb = df["turbidity_index"].replace(0, np.nan).dropna()
    ax6.hist(np.log10(valid_turb), bins=50, color="#26A69A", alpha=0.85, edgecolor="none")
    ax6.set_xlabel("log₁₀(turbidity index)")
    ax6.set_title("Turbidity index (log scale) – all samples", fontsize=10)
    for thresh, color, label in [(np.log10(0.03), "#FFA726", "clear|moderate"),
                                  (np.log10(0.08), "#E53935", "moderate|turbid")]:
        ax6.axvline(thresh, color=color, lw=1.2, ls="--", label=label)
    ax6.legend(fontsize=7)

    fig.suptitle("Per-class histograms", fontsize=13)
    _savefig(fig, out_dir / "05_class_histograms.png", tight=False)


def fig_temporal(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["year_start"] = pd.to_datetime(df["date_range_start"], errors="coerce").dt.year
    season_labels = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    df["season_label"] = df["season_id"].map(season_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    year_counts = df["year_start"].value_counts().sort_index()
    ax.bar(year_counts.index.astype(str), year_counts.values, color="#5C6BC0", alpha=0.85)
    ax.set_xlabel("Year (date_range_start)"); ax.set_ylabel("Count")
    ax.set_title("Temporal coverage (year of first image)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    ax2 = axes[1]
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    sns.countplot(data=df, x="season_label", order=season_order,
                  hue="environment_class", palette=ENV_PALETTE, ax=ax2, alpha=0.85)
    ax2.set_xlabel("Season"); ax2.set_ylabel("Count")
    ax2.set_title("Season distribution by environment", fontsize=10)
    ax2.legend(title="environment", fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    fig.suptitle("Temporal distribution", fontsize=12)
    _savefig(fig, out_dir / "06_temporal.png")


def fig_patch_coverage(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["hemisphere"] = np.where(df["latitude"] >= 0, "Northern", "Southern")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    counts = df["hemisphere"].value_counts()
    ax.pie(counts.values, labels=counts.index,
           colors=["#42A5F5", "#EF5350"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Hemisphere split", fontsize=10)

    ax2 = axes[1]
    ax2.hist(df["latitude"], bins=40, color="#66BB6A", alpha=0.85, edgecolor="none")
    ax2.set_xlabel("Latitude (°)"); ax2.set_ylabel("Count")
    ax2.set_title("Latitude distribution", fontsize=10)
    ax2.axvline(0, color="k", lw=1.0, ls="--", label="equator")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    fig.suptitle("Spatial coverage", fontsize=12)
    _savefig(fig, out_dir / "07_patch_coverage.png")


def fig_psd_curves(processed_dir: Path, df: pd.DataFrame, n_samples: int, out_dir: Path):
    """
    Radial-averaged power spectral density for HR (10 m), LR (30 m), and
    bicubically upsampled LR (at HR resolution). Demonstrates that HR contains
    recoverable high-frequency spatial content.

    x-axis: cycles per km.  LR Nyquist = 1/(2×30m) × 1000 = 16.7 cyc/km.
    HR Nyquist = 1/(2×10m) × 1000 = 50 cyc/km.
    """
    import rasterio

    lr_psds, hr_psds, bic_psds = [], [], []
    hr_pixel_m, lr_pixel_m = 10.0, 30.0

    rows = df.sample(n=min(n_samples, len(df)), random_state=7)

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="  PSD", leave=False):
        loc_id = int(row["location_id"])
        sample = f"sample_{loc_id:06d}"

        lr_dir = processed_dir / sample / "landsat"
        hr_dir = processed_dir / sample / "sentinel2"
        lr_files = sorted(lr_dir.glob("*.tif"))
        hr_files = sorted(hr_dir.glob("*.tif"))
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

        # Bicubic upsample LR to HR dimensions
        scale_factor = hr_patch.shape[0] / lr_patch.shape[0]
        bic_patch = zoom(lr_patch, scale_factor, order=3)
        # Crop or pad to exactly match HR size if needed
        h, w = hr_patch.shape
        bic_patch = bic_patch[:h, :w]

        _, hr_psd  = _radial_psd(hr_patch)
        _, lr_psd  = _radial_psd(lr_patch)
        _, bic_psd = _radial_psd(bic_patch)

        hr_psds.append(hr_psd)
        lr_psds.append(lr_psd)
        bic_psds.append(bic_psd)

    if not hr_psds:
        print("  (skipping PSD – no valid patches)")
        return

    def _freq_to_cykm(freqs_cpp, pixel_m):
        return freqs_cpp / pixel_m * 1000.0

    hr_len  = min(len(p) for p in hr_psds)
    lr_len  = min(len(p) for p in lr_psds)
    bic_len = min(len(p) for p in bic_psds)

    hr_arr  = np.stack([p[:hr_len]  for p in hr_psds])
    lr_arr  = np.stack([p[:lr_len]  for p in lr_psds])
    bic_arr = np.stack([p[:bic_len] for p in bic_psds])

    hr_freqs  = _freq_to_cykm(np.arange(hr_len)  / (2 * hr_len),  hr_pixel_m)
    lr_freqs  = _freq_to_cykm(np.arange(lr_len)  / (2 * lr_len),  lr_pixel_m)
    bic_freqs = _freq_to_cykm(np.arange(bic_len) / (2 * bic_len), hr_pixel_m)

    fig, ax = plt.subplots(figsize=(9, 6))

    def _plot_psd(freqs, arr, color, label):
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        # Skip freq=0 (DC component)
        ax.loglog(freqs[1:], mean[1:], color=color, lw=1.8, label=label)
        ax.fill_between(freqs[1:], np.maximum(mean[1:] - std[1:], 1e-12),
                         mean[1:] + std[1:], alpha=0.2, color=color)

    _plot_psd(hr_freqs,  hr_arr,  HR_COLOR,        f"Sentinel-2 HR (10 m)  n={len(hr_psds)}")
    _plot_psd(bic_freqs, bic_arr, "#78909C",        f"Bicubic↑ LR (10 m equiv.)")
    _plot_psd(lr_freqs,  lr_arr,  LR_COLOR,         f"Landsat LR (30 m)  n={len(lr_psds)}")

    # Mark LR Nyquist
    lr_nyquist = 1000.0 / (2 * lr_pixel_m)
    ax.axvline(lr_nyquist, color=LR_COLOR, ls=":", lw=1.2, alpha=0.8,
               label=f"LR Nyquist ({lr_nyquist:.1f} cyc/km)")

    ax.set_xlabel("Spatial frequency (cycles / km)", fontsize=11)
    ax.set_ylabel("Power spectral density (log)", fontsize=11)
    ax.set_title(
        "Radial-averaged PSD: LR, HR, and bicubically upsampled LR\n"
        "(band 4 / red, mean ± std across samples; shaded = ±1 SD)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    _savefig(fig, out_dir / "08_psd_curves.png", tight=False)


def fig_lr_hr_histograms(
    processed_dir: Path,
    df: pd.DataFrame,
    n_samples: int,
    out_dir: Path,
):
    """
    Paired pixel-intensity histograms: LR (Landsat) vs HR (Sentinel-2) for
    each spectrally matched band pair.
    """
    import rasterio

    # Collect per-band pixel samples from both satellites
    ls_pixels: dict[str, list[float]] = {b: [] for b, _, _ in COMPARABLE_PAIRS}
    s2_pixels: dict[str, list[float]] = {s: [] for _, s, _ in COMPARABLE_PAIRS}
    band_to_lr_idx = {ls_b: i + 1 for i, (ls_b, _, _) in enumerate(COMPARABLE_PAIRS)}
    band_to_hr_idx = {s2_b: i + 1 for i, (_, s2_b, _) in enumerate(COMPARABLE_PAIRS)}

    # Pre-compute actual 1-based rasterio indices for each band
    lr_band_idx = {ls_b: LANDSAT_SPECTRAL_BANDS.index(ls_b) + 1
                   for ls_b, _, _ in COMPARABLE_PAIRS if ls_b in LANDSAT_SPECTRAL_BANDS}
    hr_band_idx = {s2_b: S2_SPECTRAL_BANDS.index(s2_b) + 1
                   for _, s2_b, _ in COMPARABLE_PAIRS if s2_b in S2_SPECTRAL_BANDS}

    rows = df.sample(n=min(n_samples, len(df)), random_state=13)
    MAX_PIX = 50_000

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
                        nodata = src.nodata
                        if nodata is not None:
                            d = d[d != nodata]
                        d = np.clip(d[np.isfinite(d)], 0, 1)
                        if d.size:
                            ls_pixels[ls_b].extend(d[:MAX_PIX // n_samples].tolist())
            with rasterio.open(hr_files[0]) as src:
                for _, s2_b, _ in COMPARABLE_PAIRS:
                    bidx = hr_band_idx.get(s2_b)
                    if bidx and bidx <= src.count:
                        d = src.read(bidx).astype(np.float32).ravel()
                        nodata = src.nodata
                        if nodata is not None:
                            d = d[d != nodata]
                        d = np.clip(d[np.isfinite(d)], 0, 1)
                        if d.size:
                            s2_pixels[s2_b].extend(d[:MAX_PIX // n_samples].tolist())
        except Exception:
            continue

    n_pairs = len(COMPARABLE_PAIRS)
    ncols = 4
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = axes.ravel() if nrows > 1 else [axes] if ncols == 1 else axes.ravel()

    bins = np.linspace(0, 1, 60)
    for i, (ls_b, s2_b, label) in enumerate(COMPARABLE_PAIRS):
        ax = axes_flat[i]
        lr_arr = np.array(ls_pixels[ls_b])
        hr_arr = np.array(s2_pixels[s2_b])
        if lr_arr.size:
            ax.hist(lr_arr, bins=bins, color=LR_COLOR, alpha=0.6, density=True,
                    label=f"LR {ls_b}", edgecolor="none")
        if hr_arr.size:
            ax.hist(hr_arr, bins=bins, color=HR_COLOR, alpha=0.6, density=True,
                    label=f"HR {s2_b}", edgecolor="none")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Reflectance", fontsize=8)
        ax.set_ylabel("Density" if i % ncols == 0 else "", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Paired LR vs HR pixel-intensity histograms (matched spectral bands)", fontsize=12)
    _savefig(fig, out_dir / "09_lr_hr_histograms.png")


def fig_alignment_quality(
    processed_dir: Path,
    df: pd.DataFrame,
    n_samples: int,
    out_dir: Path,
):
    """
    Phase-correlation registration shift between bicubically upsampled LR and HR.

    A shift of (0, 0) means perfect pixel alignment. The distribution of
    sub-pixel shifts characterises the quality of the grid-based registration.
    """
    import rasterio
    from scipy.signal import fftconvolve

    shifts_x, shifts_y, peak_vals = [], [], []

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
        hr = hr_raw
        bic = zoom(lr_raw, hr.shape[0] / lr_raw.shape[0], order=3)
        bic = bic[:hr.shape[0], :hr.shape[1]]
        if bic.shape != hr.shape:
            continue

        # Normalise
        for arr in (hr, bic):
            arr -= arr.mean()
            std = arr.std()
            if std > 0:
                arr /= std

        # Phase correlation
        F_hr  = np.fft.fft2(hr)
        F_bic = np.fft.fft2(bic)
        cross = F_hr * np.conj(F_bic)
        denom = np.abs(cross) + 1e-12
        corr = np.abs(np.fft.ifft2(cross / denom))
        corr = np.fft.fftshift(corr)

        h, w = corr.shape
        peak_idx = np.unravel_index(corr.argmax(), corr.shape)
        dy = peak_idx[0] - h // 2
        dx = peak_idx[1] - w // 2
        # Wrap shifts > half-image to the other sign
        if abs(dy) > h // 4 or abs(dx) > w // 4:
            continue  # likely noise, skip
        shifts_x.append(float(dx))
        shifts_y.append(float(dy))
        peak_vals.append(float(corr.max()))

    if not shifts_x:
        print("  (skipping alignment – no valid pairs)")
        return

    sx, sy = np.array(shifts_x), np.array(shifts_y)
    mag = np.hypot(sx, sy)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 2D scatter of shifts
    ax = axes[0]
    ax.scatter(sx, sy, s=10, alpha=0.4, color="#546E7A", linewidths=0)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Shift x (HR pixels)"); ax.set_ylabel("Shift y (HR pixels)")
    ax.set_title("Phase-correlation shift: 2-D distribution", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Shift magnitude histogram
    ax2 = axes[1]
    ax2.hist(mag, bins=40, color="#546E7A", alpha=0.85, edgecolor="none")
    ax2.axvline(np.median(mag), color="k", ls="--", lw=1.2,
                label=f"median {np.median(mag):.2f} px")
    ax2.set_xlabel("Shift magnitude (HR pixels)"); ax2.set_ylabel("Count")
    ax2.set_title("Alignment shift magnitude", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    # Cumulative distribution
    ax3 = axes[2]
    sorted_mag = np.sort(mag)
    cdf = np.arange(1, len(sorted_mag) + 1) / len(sorted_mag)
    ax3.plot(sorted_mag, cdf * 100, color="#546E7A", lw=2)
    for threshold in [0.5, 1.0, 2.0]:
        frac = (mag <= threshold).mean() * 100
        ax3.axvline(threshold, color="gray", ls=":", lw=0.8)
        ax3.text(threshold + 0.03, frac - 3, f"{frac:.0f}%\n≤{threshold}px",
                 fontsize=7, va="top")
    ax3.set_xlabel("Shift magnitude (HR pixels)"); ax3.set_ylabel("Cumulative %")
    ax3.set_title("CDF of alignment shift magnitude", fontsize=10)
    ax3.grid(True, linewidth=0.3, alpha=0.4)
    ax3.set_ylim(0, 105)

    fig.suptitle(
        f"Registration alignment quality  (n={len(sx)} samples, band 4/red)\n"
        f"mean={mag.mean():.2f} px  median={np.median(mag):.2f} px  "
        f"p95={np.percentile(mag, 95):.2f} px  (HR pixel = 10 m)",
        fontsize=10,
    )
    _savefig(fig, out_dir / "10_alignment_quality.png")


# ─── driver ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", help="Path to dataset_manifest.csv")
    parser.add_argument("data_dir", help="Path to dataset data directory (sample_* folders)")
    parser.add_argument("--processed-dir", default="superres/processed",
                        help="Path to postprocessed data (default: superres/processed)")
    parser.add_argument("--output", "-o", default="figures/dataset_analysis",
                        help="Output directory for figures")
    parser.add_argument("--n-band-samples", type=int, default=150,
                        help="Random samples for band / PSD / histogram analysis (default: 150)")
    parser.add_argument("--skip-rasters", action="store_true",
                        help="Skip all raster-based figures (figs 3–4, 8–10)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir)

    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    print(f"  {len(df):,} samples\n")

    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    warnings.filterwarnings("ignore", category=UserWarning)

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
    else:
        print("[08–10] Raster figures skipped")

    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
