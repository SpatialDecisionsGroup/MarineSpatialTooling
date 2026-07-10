"""
Interactive dataset reviewer for the super-resolution training set.

Displays all Landsat images and the Sentinel-2 image for each sample,
lets you flag samples (or specific Landsat images) for replacement or
deletion, and saves decisions to a JSON file for later batch-processing.

Decision format in review_decisions.json
-----------------------------------------
Simple decisions are stored as strings:
    "1234": "keep" | "replace_hr" | "replace_both" | "delete"

Partial LR replacement is stored as a dict:
    "1234": {"action": "replace_lr", "files": ["landsat_02_2022-04-17.tif", ...]}

When "files" is absent or null, the whole LR stack is replaced.

Usage
-----
    uv run streamlit run superres/review_dataset.py -- \\
        --data-dir data/landsat2sentinel/data

Decisions are written to <data-dir>/../review_decisions.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import streamlit as st

# ─── constants ───────────────────────────────────────────────────────────────

_LR_SCALE,  _LR_OFFSET  = 0.0000275, -0.2   # Landsat C2 L2
_HR_SCALE,  _HR_OFFSET  = 1e-4,       0.0   # Sentinel-2 L2A
_LR_RGB = (4, 3, 2)   # SR_B4 / SR_B3 / SR_B2
_HR_RGB = (4, 3, 2)   # B4   / B3   / B2

# action key → (display label, hex colour)
DECISIONS: dict[str, tuple[str, str]] = {
    "keep":         ("✓ Keep",         "#2e7d32"),
    "replace_lr":   ("Replace LR",     "#e65100"),
    "replace_hr":   ("Replace HR",     "#1565c0"),
    "replace_both": ("Replace Both",   "#6a1b9a"),
    "delete":       ("🗑 Delete",       "#b71c1c"),
}
_SELECT_COLOUR = (255, 140, 0)   # orange border for selected LR images
_SELECT_BORDER = 6               # border width in pixels


# ─── image helpers ────────────────────────────────────────────────────────────

def _load_rgb(
    path: Path,
    bands: tuple[int, int, int],
    scale: float,
    offset: float,
) -> "np.ndarray | None":
    """Return (H, W, 3) uint8 with a 2–98 percentile stretch, or None on failure."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            r, g, b = bands
            if src.count < max(r, g, b):
                return None
            raw = np.stack([src.read(i).astype(np.float32) for i in (r, g, b)], axis=-1)
            nodata = src.nodata
        nd_mask = (raw == nodata) if nodata is not None else (raw == 0)
        refl = raw * scale + offset
        valid = refl[~nd_mask]
        if valid.size < 10:
            return None
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi <= lo:
            hi = lo + 1e-6
        stretched = np.clip((refl - lo) / (hi - lo), 0.0, 1.0)
        stretched[nd_mask] = 0.0
        return (stretched * 255).astype(np.uint8)
    except Exception:
        return None


def _load_lr_batch(
    paths: list[Path],
    bands: tuple[int, int, int],
    scale: float,
    offset: float,
) -> list["np.ndarray | None"]:
    """Load all LR images with a shared 2–98 percentile so brightness is comparable."""
    try:
        import rasterio
        raws, nd_masks = [], []
        for path in paths:
            try:
                with rasterio.open(path) as src:
                    r, g, b = bands
                    if src.count < max(r, g, b):
                        raws.append(None); nd_masks.append(None); continue
                    raw = np.stack(
                        [src.read(i).astype(np.float32) for i in (r, g, b)], axis=-1
                    )
                    nodata = src.nodata
                nd = (raw == nodata) if nodata is not None else (raw == 0)
                raws.append(raw); nd_masks.append(nd)
            except Exception:
                raws.append(None); nd_masks.append(None)

        valid_all = [
            (raw * scale + offset)[~nd]
            for raw, nd in zip(raws, nd_masks)
            if raw is not None
        ]
        if not valid_all:
            return [None] * len(paths)
        combined = np.concatenate(valid_all)
        lo, hi = np.percentile(combined, 2), np.percentile(combined, 98)
        if hi <= lo:
            hi = lo + 1e-6

        results = []
        for raw, nd in zip(raws, nd_masks):
            if raw is None:
                results.append(None)
                continue
            refl = raw * scale + offset
            stretched = np.clip((refl - lo) / (hi - lo), 0.0, 1.0)
            stretched[nd] = 0.0
            results.append((stretched * 255).astype(np.uint8))
        return results
    except Exception:
        return [None] * len(paths)


def _add_border(
    rgb: np.ndarray,
    colour: tuple[int, int, int] = _SELECT_COLOUR,
    width: int = _SELECT_BORDER,
) -> np.ndarray:
    """Draw a solid coloured border on a uint8 (H, W, 3) image."""
    result = rgb.copy()
    result[:width, :] = colour
    result[-width:, :] = colour
    result[:, :width] = colour
    result[:, -width:] = colour
    return result


# ─── data helpers ─────────────────────────────────────────────────────────────

@st.cache_data
def _all_sample_ids(data_dir: str) -> list[int]:
    return sorted(
        int(d.name.split("_")[1])
        for d in Path(data_dir).glob("sample_*")
        if d.is_dir()
        and (Path(data_dir) / d.name / "sample_metadata.json").exists()
    )


def _meta(data_dir: Path, loc_id: int) -> dict:
    f = data_dir / f"sample_{loc_id:06d}" / "sample_metadata.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _load_decisions(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_decisions(decisions: dict, path: Path) -> None:
    path.write_text(json.dumps(decisions, indent=2, sort_keys=True))


def _get_action(dec) -> str:
    """Return the action string from a decision (string or dict)."""
    if isinstance(dec, dict):
        return dec.get("action", "")
    return dec or ""


def _get_selected_files(dec) -> list[str]:
    """Return the list of specifically-selected LR files from a decision, or []."""
    if isinstance(dec, dict) and dec.get("action") == "replace_lr":
        return dec.get("files") or []
    return []


# ─── UI helpers ──────────────────────────────────────────────────────────────

def _decision_badge(dec) -> str:
    action = _get_action(dec)
    label, color = DECISIONS.get(action, (action, "#888"))
    files = _get_selected_files(dec)
    detail = f"<br/><small>{len(files)} image{'s' if len(files) != 1 else ''}</small>" if files else ""
    return (
        f"<div style='background:{color};color:white;padding:6px 14px;"
        f"border-radius:6px;font-weight:bold;text-align:center'>"
        f"{label}{detail}</div>"
    )


def _render_lr_grid(
    lr_files: list[Path],
    lr_rgbs: list,
    loc_id: int,
    selected: set[str],
) -> None:
    """Render Landsat images in rows of 4 with per-image select toggles."""
    if not lr_files:
        st.warning("No Landsat images found.")
        return

    n = len(lr_files)
    per_row = 4

    for row_start in range(0, n, per_row):
        row_files = lr_files[row_start : row_start + per_row]
        row_rgbs  = lr_rgbs [row_start : row_start + per_row]
        cols = st.columns(per_row)

        for col, fpath, rgb in zip(cols, row_files, row_rgbs):
            fname   = fpath.name
            parts   = fpath.stem.split("_")
            date    = parts[-1] if len(parts) >= 3 else fpath.stem
            is_sel  = fname in selected

            with col:
                if rgb is not None:
                    display = _add_border(rgb) if is_sel else rgb
                    st.image(display, use_container_width=True)
                else:
                    bg = "#3a2000" if is_sel else "#1a1a1a"
                    st.markdown(
                        f"<div style='background:{bg};height:100px;display:flex;"
                        f"align-items:center;justify-content:center;color:#999;"
                        f"border-radius:4px;font-size:12px'>{date}<br/>(no data)</div>",
                        unsafe_allow_html=True,
                    )

                def _toggle(fn=fname):
                    sel = st.session_state.lr_selected.get(str(loc_id), set())
                    if fn in sel:
                        sel.discard(fn)
                    else:
                        sel.add(fn)
                    st.session_state.lr_selected[str(loc_id)] = sel

                st.button(
                    f"{'✓ ' if is_sel else ''}{date}",
                    key=f"sel_{loc_id}_{fname}",
                    on_click=_toggle,
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                    help="Click to select/deselect this image for replacement",
                )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default="data/landsat2sentinel/data")
    args, _ = parser.parse_known_args()
    data_dir       = Path(args.data_dir).resolve()
    decisions_path = data_dir.parent / "review_decisions.json"

    st.set_page_config(page_title="Dataset Review", page_icon="🌊", layout="wide")

    # ── session state ─────────────────────────────────────────────────────────
    if "idx"          not in st.session_state: st.session_state.idx          = 0
    if "decisions"    not in st.session_state: st.session_state.decisions    = _load_decisions(decisions_path)
    if "filter_mode"  not in st.session_state: st.session_state.filter_mode  = "all"
    if "stretch_mode" not in st.session_state: st.session_state.stretch_mode = "shared"
    if "lr_selected"  not in st.session_state: st.session_state.lr_selected  = {}

    decisions    = st.session_state.decisions
    all_ids      = _all_sample_ids(str(data_dir))
    filter_mode  = st.session_state.filter_mode
    stretch_mode = st.session_state.stretch_mode

    if not all_ids:
        st.error(f"No samples found in {data_dir}")
        return

    # Apply filter
    if filter_mode == "unflagged":
        visible = [i for i in all_ids if str(i) not in decisions]
    elif filter_mode == "flagged":
        visible = [i for i in all_ids
                   if _get_action(decisions.get(str(i))) not in ("", "keep")]
    elif filter_mode == "kept":
        visible = [i for i in all_ids if _get_action(decisions.get(str(i))) == "keep"]
    else:
        visible = all_ids

    if not visible:
        st.info("No samples match the current filter.")
        return

    idx    = min(st.session_state.idx, len(visible) - 1)
    loc_id = visible[idx]
    selected: set[str] = st.session_state.lr_selected.get(str(loc_id), set())

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🌊 Dataset Review")
        st.caption(str(data_dir))

        n_total    = len(all_ids)
        n_reviewed = len(decisions)
        n_flagged  = sum(1 for v in decisions.values()
                         if _get_action(v) not in ("", "keep"))
        c1, c2 = st.columns(2)
        c1.metric("Total",    n_total)
        c1.metric("Reviewed", n_reviewed)
        c2.metric("Kept",     n_reviewed - n_flagged)
        c2.metric("Flagged",  n_flagged)
        st.progress(
            n_reviewed / n_total if n_total else 0,
            text=f"{n_reviewed}/{n_total} reviewed",
        )

        st.divider()

        st.subheader("Navigate")
        jump = st.number_input(
            "Jump to location ID",
            min_value=int(min(all_ids)), max_value=int(max(all_ids)),
            value=int(loc_id), step=1, label_visibility="collapsed",
        )
        if int(jump) != loc_id:
            if int(jump) in visible:
                st.session_state.idx = visible.index(int(jump))
                st.rerun()
            else:
                st.caption(f"#{jump} not in current filter.")

        cp, cn = st.columns(2)
        if cp.button("◀ Prev", use_container_width=True, disabled=(idx == 0)):
            st.session_state.idx = idx - 1
            st.rerun()
        if cn.button("Next ▶", use_container_width=True, disabled=(idx == len(visible) - 1)):
            st.session_state.idx = idx + 1
            st.rerun()
        st.caption(f"{idx + 1} / {len(visible)} visible")

        st.divider()

        st.subheader("Filter")
        new_filter = st.radio(
            "Show", ["all", "unflagged", "flagged", "kept"],
            index=["all", "unflagged", "flagged", "kept"].index(filter_mode),
            label_visibility="collapsed",
        )
        if new_filter != filter_mode:
            st.session_state.filter_mode = new_filter
            st.session_state.idx = 0
            st.rerun()

        st.divider()

        st.subheader("LR stretch")
        new_stretch = st.radio(
            "Stretch", ["shared", "per-image"],
            index=0 if stretch_mode == "shared" else 1,
            label_visibility="collapsed",
            help="Shared: same brightness scale across all Landsat images so you can "
                 "compare relative brightness. Per-image: maximises contrast individually.",
        )
        if new_stretch != stretch_mode:
            st.session_state.stretch_mode = new_stretch
            st.rerun()

        st.divider()

        if decisions:
            st.subheader("Decision summary")
            counts: dict[str, int] = {}
            for v in decisions.values():
                a = _get_action(v)
                counts[a] = counts.get(a, 0) + 1
            for key in list(DECISIONS) + ["replace_lr_partial"]:
                if key in counts:
                    label = DECISIONS.get(key, (key, ""))[0]
                    st.write(f"{label}: **{counts[key]}**")

    # ── main panel ────────────────────────────────────────────────────────────
    meta        = _meta(data_dir, loc_id)
    current_dec = decisions.get(str(loc_id))
    cur_action  = _get_action(current_dec)
    env         = meta.get("environment_class",  "?")
    depth_cls   = meta.get("depth_class",         "?")
    turb_cls    = meta.get("turbidity_class",     "?")
    lat         = meta.get("latitude",            "?")
    lon         = meta.get("longitude",           "?")
    depth_m     = meta.get("depth_m")
    n_lr        = meta.get("lowres_count",        "?")
    dr_start    = (meta.get("date_range_start") or "")[:10]
    dr_end      = (meta.get("date_range_end")   or "")[:10]

    hcol, bcol = st.columns([5, 1])
    with hcol:
        st.title(f"Sample #{loc_id}")
        st.caption(
            f"{env} / {depth_cls} / {turb_cls}  ·  ({lat}, {lon})"
            + (f"  ·  {depth_m:.0f} m" if isinstance(depth_m, (int, float)) else "")
            + (f"  ·  LR window {dr_start} → {dr_end}" if dr_start else "")
        )
    with bcol:
        if current_dec:
            st.markdown(_decision_badge(current_dec), unsafe_allow_html=True)

    # ── Landsat images ────────────────────────────────────────────────────────
    ls_dir   = data_dir / f"sample_{loc_id:06d}" / "landsat"
    lr_files = sorted(ls_dir.glob("*.tif")) if ls_dir.exists() else []

    n_sel = len(selected)
    hdr_col, sel_col = st.columns([3, 1])
    with hdr_col:
        st.subheader(
            f"Landsat LR  ({len(lr_files)} images"
            + (f", {n_sel} selected" if n_sel else "")
            + ")"
        )
    with sel_col:
        if n_sel:
            if st.button("Clear selection", use_container_width=True):
                st.session_state.lr_selected[str(loc_id)] = set()
                st.rerun()
        else:
            if st.button("Select all", use_container_width=True):
                st.session_state.lr_selected[str(loc_id)] = {f.name for f in lr_files}
                st.rerun()

    if lr_files:
        if stretch_mode == "shared":
            lr_rgbs = _load_lr_batch(lr_files, _LR_RGB, _LR_SCALE, _LR_OFFSET)
        else:
            lr_rgbs = [_load_rgb(f, _LR_RGB, _LR_SCALE, _LR_OFFSET) for f in lr_files]
        _render_lr_grid(lr_files, lr_rgbs, loc_id, selected)
    else:
        st.warning("No Landsat images found.")

    st.divider()

    # ── Sentinel-2 + metadata ─────────────────────────────────────────────────
    s2_dir   = data_dir / f"sample_{loc_id:06d}" / "sentinel2"
    s2_files = sorted(s2_dir.glob("*.tif")) if s2_dir.exists() else []

    img_col, meta_col = st.columns([1, 2])

    with img_col:
        st.subheader("Sentinel-2 HR")
        if s2_files:
            s2_rgb = _load_rgb(s2_files[0], _HR_RGB, _HR_SCALE, _HR_OFFSET)
            if s2_rgb is not None:
                st.image(s2_rgb, caption=s2_files[0].name, use_container_width=True)
            else:
                st.warning("Could not render HR image.")
        else:
            st.warning("No Sentinel-2 image found.")

    with meta_col:
        st.subheader("Metadata")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Environment",  env)
        mc1.metric("# LR images", n_lr)
        mc2.metric("Depth class",  depth_cls)
        mc2.metric("Depth",        f"{depth_m:.0f} m" if isinstance(depth_m, (int, float)) else "?")
        mc3.metric("Turbidity",    turb_cls)
        turb_idx = meta.get("turbidity_index")
        mc3.metric("Turb. index",  f"{turb_idx:.4f}" if isinstance(turb_idx, float) else "?")

    st.divider()

    # ── action buttons ────────────────────────────────────────────────────────
    st.subheader("Decision")

    def _record(action: str, files: list[str] | None = None) -> None:
        if action == "replace_lr" and files:
            st.session_state.decisions[str(loc_id)] = {"action": "replace_lr", "files": files}
        else:
            st.session_state.decisions[str(loc_id)] = action
        _save_decisions(st.session_state.decisions, decisions_path)
        if idx < len(visible) - 1:
            st.session_state.idx = idx + 1

    def _action_cb(key: str):
        def _cb():
            _record(key)
        return _cb

    # Standard action buttons
    btn_cols = st.columns(len(DECISIONS) + (1 if n_sel else 0))
    for col, (key, (label, _)) in zip(btn_cols, DECISIONS.items()):
        is_cur = (cur_action == key and not _get_selected_files(current_dec))
        col.button(
            label,
            key=f"action_{key}",
            on_click=_action_cb(key),
            use_container_width=True,
            type="primary" if is_cur else "secondary",
        )

    # "Replace Selected" button only shown when images are selected
    if n_sel:
        sel_files = sorted(selected)
        is_cur_partial = (
            isinstance(current_dec, dict)
            and current_dec.get("action") == "replace_lr"
            and set(current_dec.get("files", [])) == selected
        )

        def _replace_selected():
            _record("replace_lr", sel_files)

        btn_cols[-1].button(
            f"Replace {n_sel} selected",
            key="action_replace_selected",
            on_click=_replace_selected,
            use_container_width=True,
            type="primary" if is_cur_partial else "secondary",
            help=f"Mark only the {n_sel} highlighted Landsat image(s) for replacement",
        )

    st.caption(
        "Click image date buttons to select/deselect individual Landsat images. "
        "Decisions auto-advance to the next sample."
    )


if __name__ == "__main__":
    main()
