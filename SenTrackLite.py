import numpy as np
import scipy.ndimage as ndi
from skimage import morphology, segmentation
from dataclasses import dataclass

# --- Additional imports for Figure 3 (A,C) ---
import csv
import re

# --- Additional imports for GUI and TIFF/image handling ---
from pathlib import Path
from typing import Optional, Tuple

import tifffile as tiff
from skimage import filters, measure, util
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max

APP_NAME = "SenTrackLite"
APP_VERSION = "0.3.0"

@dataclass
class Config:
    p_low: float = 0.01
    p_high: float = 99.9
    dapi_sigma: float = 1.0
    nuc_min_area: int = 100
    nuc_min_dist: int = 10
    cell_expand: float = 10.0
    cell_mask_method: str = "expand"  # 'expand' (default) or 'watershed'
    lyso_sigma: float = 1.0
    lyso_thresh: float = 0.5
    lyso_pct: float = 85.0
    lyso_min_area: int = 50
    beta_overlap: float = 0.5
    beta_invert: bool = False
    # --- SA-β-Gal (brightfield) tuning ---
    beta_p_low: float = 0.5
    beta_p_high: float = 99.5
    beta_thr_pct: float = 90.0
    beta_min_area: int = 200
    beta_close_radius: int = 2
    fg_from_lyso: bool = False
    fg_lyso_pct: float = 85.0  # percentile for foreground mask when using Lyso to constrain cell masks

def norm01(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Normalize image to [0,1] using percentile clipping."""
    low = np.percentile(img, p_low)
    high = np.percentile(img, p_high)
    img_clipped = np.clip(img, low, high)
    return (img_clipped - low) / (high - low) if high > low else np.zeros_like(img)


# --- Additional helper functions for GUI ---

def read_tiff_2d(path: Path, plane_index: int = 0) -> np.ndarray:
    """Read a TIFF and return a 2D image.

    Supports common microscope exports:
    - (H, W)
    - (planes, H, W)
    - (H, W, planes)
    - (H, W, 3/4) RGB/RGBA (converted to grayscale)
    """
    try:
        arr = tiff.imread(str(path))
    except ValueError as e:
        msg = str(e)
        # Many LZW (and some other) TIFF compressions require optional codecs.
        # If `imagecodecs` is not installed, try a Pillow fallback (often works via libtiff).
        if "requires the 'imagecodecs' package" in msg or "requires the imagecodecs package" in msg:
            try:
                from PIL import Image

                with Image.open(str(path)) as im:
                    # For multi-page TIFFs, use plane_index as the frame/page index.
                    n_frames = getattr(im, "n_frames", 1)
                    if n_frames and n_frames > 1:
                        idx = int(np.clip(int(plane_index), 0, int(n_frames) - 1))
                        im.seek(idx)
                    arr = np.array(im)
            except Exception as e2:
                raise RuntimeError(
                    "This TIFF appears to be compressed (often LZW). To read it you need ONE of these options:\n"
                    "1) Install `imagecodecs` (recommended):\n"
                    "   python -m pip install imagecodecs\n"
                    "2) Or install Pillow and try again:\n"
                    "   python -m pip install pillow\n"
                    "3) Or convert the TIFF to uncompressed (e.g., in Fiji/ImageJ: File > Save As > Tiff..., choose Compression: None).\n"
                    "Then restart the app."
                ) from e
        else:
            raise

    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF, got {arr.shape} for: {path}")

    # (planes, H, W)
    if arr.shape[0] <= 32 and arr.shape[1] > 32 and arr.shape[2] > 32:
        idx = int(np.clip(plane_index, 0, arr.shape[0] - 1))
        out = np.squeeze(arr[idx])
        if out.ndim != 2:
            raise ValueError(f"Could not extract 2D plane from {path} (shape {arr.shape})")
        return out

    # (H, W, planes) or RGB/RGBA
    if arr.shape[2] <= 32 and arr.shape[0] > 32 and arr.shape[1] > 32:
        d = arr.shape[2]
        if d in (3, 4):
            rgb = arr[:, :, :3].astype(np.float32)
            out = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
            return out
        idx = int(np.clip(plane_index, 0, d - 1))
        out = np.squeeze(arr[:, :, idx])
        if out.ndim != 2:
            raise ValueError(f"Could not extract 2D plane from {path} (shape {arr.shape})")
        return out

    # Ambiguous: fall back to first slice
    out = np.squeeze(arr[0])
    if out.ndim != 2:
        raise ValueError(f"Could not extract 2D plane from {path} (shape {arr.shape})")
    return out


def harmonize_shapes(
    dapi: np.ndarray,
    lyso: np.ndarray,
    beta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Make channels compatible by transposing if needed, then cropping to common overlap."""
    d = np.squeeze(np.asarray(dapi))
    l = np.squeeze(np.asarray(lyso))
    b = np.squeeze(np.asarray(beta)) if beta is not None else None

    if d.ndim != 2 or l.ndim != 2 or (b is not None and b.ndim != 2):
        raise ValueError(f"All channels must be 2D after reading. DAPI {d.shape}, Lyso {l.shape}, Beta {None if b is None else b.shape}")

    if l.shape != d.shape and l.T.shape == d.shape:
        l = l.T
    if b is not None and b.shape != d.shape and b.T.shape == d.shape:
        b = b.T

    shapes = [d.shape, l.shape] + ([b.shape] if b is not None else [])
    h = min(s[0] for s in shapes)
    w = min(s[1] for s in shapes)

    d = d[:h, :w]
    l = l[:h, :w]
    if b is not None:
        b = b[:h, :w]

    return d, l, b


def segment_nuclei(d01: np.ndarray, cfg: Config) -> np.ndarray:
    """Segment nuclei from normalized DAPI [0,1] using Otsu + watershed."""
    img = filters.gaussian(d01, sigma=float(cfg.dapi_sigma), preserve_range=True)
    thr = filters.threshold_otsu(img)
    bw = img > thr
    bw = morphology.remove_small_objects(bw, min_size=int(cfg.nuc_min_area))
    bw = morphology.binary_closing(bw, morphology.disk(2))

    dist = ndi.distance_transform_edt(bw)

    md = int(max(1, getattr(cfg, "nuc_min_dist", 10)))
    peaks = peak_local_max(
        dist,
        labels=bw.astype(np.uint8),
        min_distance=md,
        exclude_border=False,
    )

    markers = np.zeros_like(dist, dtype=np.int32)
    if peaks.size > 0:
        for i, (rr, cc) in enumerate(peaks, start=1):
            markers[int(rr), int(cc)] = i
    else:
        # Fallback: single marker at the global max of dist
        rr, cc = np.unravel_index(int(np.argmax(dist)), dist.shape)
        markers[int(rr), int(cc)] = 1

    lab = segmentation.watershed(-dist, markers=markers, mask=bw)
    lab = morphology.remove_small_objects(lab, min_size=int(cfg.nuc_min_area))
    # Relabel to 1..N
    lab = measure.label(lab > 0)
    return lab.astype(np.int32)


def segment_lyso(l01: np.ndarray, cells: np.ndarray, cfg: Config) -> np.ndarray:
    """Segment lysosome-rich pixels from normalized Lyso [0,1]."""
    img = filters.gaussian(l01, sigma=float(cfg.lyso_sigma), preserve_range=True)

    # Threshold policy: if lyso_thresh in (0,1) treat as fixed, else use percentile.
    if 0.0 < float(cfg.lyso_thresh) < 1.0:
        thr = float(cfg.lyso_thresh)
    else:
        thr = np.percentile(img, float(cfg.lyso_pct))

    bw = img > thr
    bw = morphology.remove_small_objects(bw, min_size=int(cfg.lyso_min_area))
    bw = morphology.binary_opening(bw, morphology.disk(1))

    # Constrain to cell pixels
    bw &= (cells > 0)
    return bw


def segment_beta(beta_img: np.ndarray, cfg: Config) -> np.ndarray:
    """Create a rough binary SA-β-Gal mask from a grayscale brightfield image.

    Tunable via cfg fields (with safe defaults if missing):
      - beta_p_low, beta_p_high: percentile clip for normalization
      - beta_invert: invert after normalization
      - beta_thr_pct: percentile threshold on normalized image
      - beta_min_area: remove small objects
      - beta_close_radius: binary closing radius
    """
    b01 = norm01(
        beta_img.astype(np.float32),
        float(getattr(cfg, "beta_p_low", 0.5)),
        float(getattr(cfg, "beta_p_high", 99.5)),
    )

    if bool(getattr(cfg, "beta_invert", False)):
        b01 = 1.0 - b01

    thr_pct = float(getattr(cfg, "beta_thr_pct", 90.0))
    thr = np.percentile(b01, thr_pct)
    bw = b01 > thr

    min_area = int(getattr(cfg, "beta_min_area", 200))
    bw = morphology.remove_small_objects(bw, min_size=min_area)

    rad = int(getattr(cfg, "beta_close_radius", 2))
    rad = max(0, rad)
    if rad > 0:
        bw = morphology.binary_closing(bw, morphology.disk(rad))

    bw = ndi.binary_fill_holes(bw)
    return bw


def compute_cell_table(cells: np.ndarray, l01: np.ndarray, lyso_bw: np.ndarray, beta_bw: Optional[np.ndarray], cfg: Config):
    """Return a per-cell dataframe-like dict and label map."""
    labels = np.unique(cells)
    labels = labels[labels != 0]

    rows = []
    for lab in labels:
        cm = cells == lab
        cell_area = int(np.count_nonzero(cm))

        lm = lyso_bw & cm
        lyso_area = int(np.count_nonzero(lm))
        lyso_mean = float(np.mean(l01[lm])) if lyso_area > 0 else 0.0
        lyso_int = float(np.sum(l01[lm])) if lyso_area > 0 else 0.0

        is_sen = ""
        beta_overlap = 0.0
        if beta_bw is not None:
            overlap = int(np.count_nonzero(beta_bw & cm))
            beta_overlap = overlap / float(cell_area) if cell_area > 0 else 0.0
            is_sen = int(beta_overlap >= float(cfg.beta_overlap))

        rows.append(
            {
                "cell_id": int(lab),
                "cell_area_px": cell_area,
                "lyso_area_px": lyso_area,
                "lyso_mean": lyso_mean,
                "lyso_integrated": lyso_int,
                "beta_overlap_frac": float(beta_overlap),
                "label_sen": is_sen,
            }
        )

    return rows

def make_foreground_mask(d01: np.ndarray, l01: np.ndarray, cfg: Config) -> np.ndarray:
    """Return a boolean mask of where cells are allowed to exist.

    If cfg.fg_from_lyso is True, build the mask from Lyso intensity.
    Otherwise, allow the full image.
    """
    h, w = d01.shape
    if not getattr(cfg, "fg_from_lyso", False):
        return np.ones((h, w), dtype=bool)

    # Use a robust percentile threshold on the normalized Lyso image.
    pct = float(getattr(cfg, "fg_lyso_pct", 85.0))
    thr = np.percentile(l01, pct)
    fg = l01 > thr

    # Clean up the foreground mask.
    fg = morphology.remove_small_objects(fg, min_size=200)
    fg = morphology.binary_closing(fg, morphology.disk(3))
    fg = ndi.binary_fill_holes(fg)
    return fg

def make_cell_masks(nuclei: np.ndarray, l01: np.ndarray, cfg: Config, d01: np.ndarray = None) -> np.ndarray:
    # Build a foreground mask that constrains where cell pixels can be assigned.
    fg = make_foreground_mask(norm01(d01, cfg.p_low, cfg.p_high), l01, cfg) if d01 is not None else make_foreground_mask(l01, l01, cfg)

    method = getattr(cfg, "cell_mask_method", "expand").strip().lower()

    # Method 1 (existing): nucleus expansion
    if method == "expand":
        dist = int(max(0, round(float(cfg.cell_expand))))
        if dist == 0:
            out = nuclei.astype(np.int32)
        else:
            out = segmentation.expand_labels(nuclei.astype(np.int32), distance=dist).astype(np.int32)
        out[~fg] = 0
        return out

    # Method 2 (new): nuclei-seeded watershed inside foreground (Voronoi-like)
    if method == "watershed":
        markers = nuclei.astype(np.int32)
        if markers.max() == 0:
            return np.zeros_like(markers)

        # Use a distance map so regions grow outward from nuclei within the foreground.
        dist = ndi.distance_transform_edt(fg)
        # Watershed grows basins from markers on the negative distance
        cells = segmentation.watershed(-dist, markers=markers, mask=fg)
        return cells.astype(np.int32)

    raise ValueError(f"Unknown cell_mask_method: {cfg.cell_mask_method}. Use 'expand' or 'watershed'.")


# --- Optional assist-plot helper functions (hidden in UI) ---

def _read_per_cell_csv(csv_path: Path):
    """Read a per-cell CSV into a list of dicts (avoid pandas dependency)."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def _infer_well_id_from_stem(stem: str) -> str:
    """Infer a well like A01 from a filename stem."""
    m = re.search(r"\b([A-H][0-9]{2})\b", stem)
    return m.group(1) if m else "UNKNOWN"


def _infer_group_from_stem(stem: str) -> str:
    """Infer a condition/group name by removing well token from the stem."""
    well = _infer_well_id_from_stem(stem)
    if well == "UNKNOWN":
        return stem
    # Remove the well token and any leftover separators around it
    s = re.sub(rf"([_\-\. ]?){well}([_\-\. ]?)", "_", stem)
    s = re.sub(r"__+", "_", s).strip("_ ")
    return s if s else "Group"


def load_folder_feature_values(folder: Path, feature_col: str):
    """Load feature values and optional classification columns from a folder of per-cell CSVs.

    Returns:
      all_vals: 1D np.ndarray of feature values across all cells
      per_file: list of dict with keys {path, stem, well, group, vals, labels(optional), probs(optional)}
    """
    folder = Path(folder)
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise ValueError(f"No CSV files found in: {folder}")

    per_file = []
    all_vals = []

    for p in csvs:
        rows = _read_per_cell_csv(p)
        if not rows:
            continue

        stem = p.stem
        well = _infer_well_id_from_stem(stem)
        group = _infer_group_from_stem(stem)

        vals = []
        labels = []
        probs = []

        # Try to find a probability column if present
        prob_keys = [
            "sen_prob",
            "senescence_prob",
            "prob_sen",
            "p_sen",
            "pred_prob",
            "pred_proba",
            "probability",
        ]
        has_label = "label_sen" in rows[0]
        prob_key = None
        for k in prob_keys:
            if k in rows[0]:
                prob_key = k
                break

        for r in rows:
            if feature_col in r:
                try:
                    v = float(r[feature_col])
                    if np.isfinite(v):
                        vals.append(v)
                        all_vals.append(v)
                except Exception:
                    pass

            if has_label and "label_sen" in r:
                try:
                    labels.append(int(float(r["label_sen"])))
                except Exception:
                    pass

            if prob_key is not None and prob_key in r:
                try:
                    probs.append(float(r[prob_key]))
                except Exception:
                    pass

        per_file.append(
            {
                "path": p,
                "stem": stem,
                "well": well,
                "group": group,
                "vals": np.asarray(vals, dtype=float),
                "labels": np.asarray(labels, dtype=int) if labels else None,
                "probs": np.asarray(probs, dtype=float) if probs else None,
            }
        )

    return np.asarray(all_vals, dtype=float), per_file


def _frac_senescent(entry: dict, mode: str, threshold: float, feature_for_threshold: str):
    """Compute fraction senescent for one file entry."""
    # mode can be: "label", "prob", "feature"
    if mode == "label" and entry.get("labels") is not None and entry["labels"].size:
        y = entry["labels"].astype(int)
        return float(np.mean(y > 0))

    if mode == "prob" and entry.get("probs") is not None and entry["probs"].size:
        p = entry["probs"].astype(float)
        return float(np.mean(p >= float(threshold)))

    # fallback: feature threshold on the chosen feature column
    v = entry.get("vals")
    if v is None or v.size == 0:
        return 0.0
    return float(np.mean(v >= float(threshold)))


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def make_assist_plots(
    out_png: Path,
    folder_a: Path,
    folder_b: Path,
    plate_folder: Path,
    feature_col: str = "lyso_mean",
    burden_mode: str = "auto",
    prob_threshold: float = 0.5,
):
    """Create optional QC/summary plots as a single PNG.

    Left: distribution of per-cell feature values for two folders of per-cell CSVs.
    Right: per-group senescence burden from a folder of per-well per-cell CSVs.

    This tool is intentionally generic and hidden behind a small UI affordance.
    """
    # --- Distribution plot data
    low_vals, _ = load_folder_feature_values(Path(folder_a), feature_col)
    high_vals, _ = load_folder_feature_values(Path(folder_b), feature_col)

    # --- Burden summary data
    _, per_file = load_folder_feature_values(Path(plate_folder), feature_col)

    # Decide burden mode
    # auto: label if present, else prob if present, else feature threshold
    if burden_mode == "auto":
        has_any_label = any(e.get("labels") is not None and e["labels"].size for e in per_file)
        has_any_prob = any(e.get("probs") is not None and e["probs"].size for e in per_file)
        if has_any_label:
            mode = "label"
        elif has_any_prob:
            mode = "prob"
        else:
            mode = "feature"
    else:
        mode = burden_mode

    # Compute per-well fractions
    for e in per_file:
        e["frac_sen"] = _frac_senescent(e, mode=mode, threshold=prob_threshold, feature_for_threshold=feature_col)

    # Group by inferred group name
    groups = {}
    for e in per_file:
        g = e.get("group", "Group")
        groups.setdefault(g, []).append(e["frac_sen"])

    group_names = sorted(groups.keys())
    group_means = [float(np.mean(groups[g])) if groups[g] else 0.0 for g in group_names]
    group_sems = [_sem(np.asarray(groups[g], dtype=float)) for g in group_names]

    # --- Render
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10.5, 4.2), dpi=300)
    axA = fig.add_subplot(1, 2, 1)
    axC = fig.add_subplot(1, 2, 2)

    # Panel A: violin + medians
    dataA = [low_vals, high_vals]
    parts = axA.violinplot(dataA, showmeans=False, showmedians=True, showextrema=False)
    axA.set_xticks([1, 2])
    axA.set_xticklabels(["Set A", "Set B"], rotation=0)
    axA.set_ylabel(feature_col)
    axA.set_title("Feature distribution")
    axA.grid(True, axis="y", alpha=0.25)

    # Add simple n text
    axA.text(1, axA.get_ylim()[1] * 0.98, f"n={len(low_vals)}", ha="center", va="top", fontsize=8)
    axA.text(2, axA.get_ylim()[1] * 0.98, f"n={len(high_vals)}", ha="center", va="top", fontsize=8)

    # Panel C: per-group burden bars with SEM
    x = np.arange(len(group_names))
    axC.bar(x, group_means, yerr=group_sems, capsize=4)
    axC.set_xticks(x)
    axC.set_xticklabels(group_names, rotation=45, ha="right")
    axC.set_ylim(0, 1)
    axC.set_ylabel("Fraction senescent")
    axC.set_title("Senescence burden")
    axC.grid(True, axis="y", alpha=0.25)

    # Small subtitle about burden mode
    if mode == "label":
        note = "Burden from label_sen"
    elif mode == "prob":
        note = f"Burden from prob >= {prob_threshold}"
    else:
        note = f"Burden from {feature_col} >= {prob_threshold}"
    axC.text(0.01, 0.98, note, transform=axC.transAxes, ha="left", va="top", fontsize=8)

    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(str(out_png), dpi=300)


# --- Output shipping helpers ---
def _safe_stem(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_ ")
    return s or "run"


def _cfg_to_dict(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}


def _summarize_well(rows: list[dict]) -> dict:
    n = int(len(rows))
    out = {
        "n_cells": n,
        "frac_senescent": "",
        "mean_cell_area_px": "",
        "mean_lyso_area_px": "",
        "mean_lyso_mean": "",
        "mean_lyso_integrated": "",
    }
    if n == 0:
        return out

    def _mean(key: str):
        vals = []
        for r in rows:
            try:
                v = float(r.get(key, 0.0))
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        return float(np.mean(vals)) if vals else ""

    out["mean_cell_area_px"] = _mean("cell_area_px")
    out["mean_lyso_area_px"] = _mean("lyso_area_px")
    out["mean_lyso_mean"] = _mean("lyso_mean")
    out["mean_lyso_integrated"] = _mean("lyso_integrated")

    labels = []
    for r in rows:
        v = r.get("label_sen", "")
        if v == "" or v is None:
            continue
        try:
            labels.append(int(float(v)))
        except Exception:
            pass
    if labels:
        out["frac_senescent"] = float(np.mean(np.asarray(labels) > 0))

    return out


def _write_run_manifest(out_dir: Path, base: str, cfg: Config, inputs: dict, preview_scale: int, files_written: list[Path]):
    from datetime import datetime
    import json

    manifest = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs,
        "preview_downsample_factor": int(preview_scale),
        "config": _cfg_to_dict(cfg),
        "outputs": [str(p.name) for p in files_written],
    }

    path = out_dir / f"{base}_run.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return path


def _maybe_zip_outputs(out_dir: Path, base: str, files_written: list[Path]) -> Path:
    import zipfile

    zpath = out_dir / f"{base}_outputs.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files_written:
            if p.exists():
                z.write(p, arcname=p.name)
    return zpath


def gui_main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("SenTrackLite Segmentation GUI")

    cfg = Config()

    # ----------------------------
    # State
    # ----------------------------
    state = {
        "dapi_path": None,
        "lyso_path": None,
        "beta_path": None,
        "dapi": None,
        "lyso": None,
        "beta": None,
        "last_error": None,
        "preview_scale": 1,
    }

    # ----------------------------
    # Tk variables
    # ----------------------------
    v_dapi = tk.StringVar(value="")
    v_lyso = tk.StringVar(value="")
    v_beta = tk.StringVar(value="")

    v_plane = tk.StringVar(value="0")

    v_p_low = tk.StringVar(value=str(cfg.p_low))
    v_p_high = tk.StringVar(value=str(cfg.p_high))
    v_dapi_sigma = tk.StringVar(value=str(cfg.dapi_sigma))
    v_nuc_min_area = tk.StringVar(value=str(cfg.nuc_min_area))
    v_cell_expand = tk.StringVar(value=str(cfg.cell_expand))
    v_cell_method = tk.StringVar(value=str(cfg.cell_mask_method))

    v_fg_from_lyso = tk.BooleanVar(value=bool(cfg.fg_from_lyso))
    v_fg_lyso_pct = tk.StringVar(value=str(cfg.fg_lyso_pct))

    v_lyso_sigma = tk.StringVar(value=str(cfg.lyso_sigma))
    v_lyso_thresh = tk.StringVar(value=str(cfg.lyso_thresh))
    v_lyso_pct = tk.StringVar(value=str(cfg.lyso_pct))
    v_lyso_min_area = tk.StringVar(value=str(cfg.lyso_min_area))

    v_beta_invert = tk.BooleanVar(value=bool(cfg.beta_invert))
    v_beta_overlap = tk.StringVar(value=str(cfg.beta_overlap))
    v_beta_p_low = tk.StringVar(value=str(cfg.beta_p_low))
    v_beta_p_high = tk.StringVar(value=str(cfg.beta_p_high))
    v_beta_thr_pct = tk.StringVar(value=str(cfg.beta_thr_pct))
    v_beta_min_area = tk.StringVar(value=str(cfg.beta_min_area))
    v_beta_close_radius = tk.StringVar(value=str(cfg.beta_close_radius))
    v_zip_outputs = tk.BooleanVar(value=True)
    v_qc_box = tk.StringVar(value="260")

    # Debounce preview updates
    _after_id = {"id": None}

    def collect_cfg() -> Config:
        c = Config()
        c.p_low = float(v_p_low.get())
        c.p_high = float(v_p_high.get())
        c.dapi_sigma = float(v_dapi_sigma.get())
        c.nuc_min_area = int(float(v_nuc_min_area.get()))
        c.cell_expand = float(v_cell_expand.get())
        c.cell_mask_method = v_cell_method.get().strip()
        c.fg_from_lyso = bool(v_fg_from_lyso.get())
        c.fg_lyso_pct = float(v_fg_lyso_pct.get())
        c.lyso_sigma = float(v_lyso_sigma.get())
        c.lyso_thresh = float(v_lyso_thresh.get())
        c.lyso_pct = float(v_lyso_pct.get())
        c.lyso_min_area = int(float(v_lyso_min_area.get()))
        c.beta_invert = bool(v_beta_invert.get())
        c.beta_overlap = float(v_beta_overlap.get())
        c.beta_p_low = float(v_beta_p_low.get())
        c.beta_p_high = float(v_beta_p_high.get())
        c.beta_thr_pct = float(v_beta_thr_pct.get())
        c.beta_min_area = int(float(v_beta_min_area.get()))
        c.beta_close_radius = int(float(v_beta_close_radius.get()))
        return c

    def _compute_segmentation_for_current_state():
        """Compute segmentation products for the currently loaded state images (already preview-downsampled if applicable)."""
        if state.get("dapi") is None or state.get("lyso") is None:
            return None
        c = collect_cfg()
        dapi = state["dapi"].astype(np.float32)
        lyso = state["lyso"].astype(np.float32)

        d01 = norm01(dapi, c.p_low, c.p_high)
        l01 = norm01(lyso, c.p_low, c.p_high)

        nuclei = segment_nuclei(d01, c)
        cells = make_cell_masks(nuclei, l01, c, d01=d01)
        lyso_bw = segment_lyso(l01, cells, c)

        beta = state.get("beta")
        beta_bw = None
        beta_show = None
        if beta is not None:
            beta = beta.astype(np.float32)
            beta_show = norm01(beta, float(c.beta_p_low), float(c.beta_p_high))
            if bool(c.beta_invert):
                beta_show = 1.0 - beta_show
            beta_bw = segment_beta(beta, c)
            beta_bw &= (cells > 0)

        return {
            "cfg": c,
            "d01": d01,
            "l01": l01,
            "nuclei": nuclei,
            "cells": cells,
            "lyso_bw": lyso_bw,
            "beta_bw": beta_bw,
            "beta_show": beta_show,
        }

    def _cell_bbox_from_label(lbl: np.ndarray, cell_id: int):
        ys, xs = np.nonzero(lbl == int(cell_id))
        if ys.size == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        return y0, x0, y1, x1

    def _crop_center(img: np.ndarray, cy: int, cx: int, half: int):
        h, w = img.shape
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        return img[y0:y1, x0:x1]

    def load_images(for_preview: bool = True):
        """Load images from disk into state, harmonize shapes.

        For preview, large images are downsampled via stride slicing to keep the GUI responsive.
        Saving outputs always uses full resolution.
        """
        try:
            dapi_p = v_dapi.get().strip()
            lyso_p = v_lyso.get().strip()
            beta_p = v_beta.get().strip()
            if not dapi_p or not lyso_p:
                return

            plane = int(float(v_plane.get()))
            d = read_tiff_2d(Path(dapi_p), plane_index=plane)
            l = read_tiff_2d(Path(lyso_p), plane_index=plane)
            b = None
            if beta_p:
                b = read_tiff_2d(Path(beta_p), plane_index=plane)

            d, l, b = harmonize_shapes(d, l, b)

            scale = 1
            if for_preview:
                h, w = d.shape
                max_dim = 3000
                max_pixels = 20_000_000
                scale_dim = int(np.ceil(max(h / max_dim, w / max_dim, 1.0)))
                scale_pix = int(np.ceil(np.sqrt((h * w) / max_pixels))) if (h * w) > max_pixels else 1
                scale = int(max(1, scale_dim, scale_pix))
                if scale > 1:
                    d = d[::scale, ::scale]
                    l = l[::scale, ::scale]
                    if b is not None:
                        b = b[::scale, ::scale]

            state["dapi"], state["lyso"], state["beta"] = d, l, b
            state["preview_scale"] = int(scale)
            state["last_error"] = None
        except Exception as e:
            state["last_error"] = str(e)
            state["dapi"], state["lyso"], state["beta"] = None, None, None
            state["preview_scale"] = 1

    def schedule_update(*_):
        if _after_id["id"] is not None:
            try:
                root.after_cancel(_after_id["id"])
            except Exception:
                pass
        _after_id["id"] = root.after(150, update_preview)

    # ----------------------------
    # Layout
    # ----------------------------
    outer = ttk.Frame(root, padding=8)
    outer.pack(fill="both", expand=True)

    left = ttk.Frame(outer)
    left.pack(side="left", fill="y")

    right = ttk.Frame(outer)
    right.pack(side="right", fill="both", expand=True)

    # ----------------------------
    # Preview figure
    # ----------------------------
    fig = Figure(figsize=(10, 6), dpi=100)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.set_title("DAPI")
    ax2.set_title("Lyso")
    ax3.set_title("Nuclei labels")
    ax4.set_title("Cell masks")
    ax5.set_title("Lyso mask")
    ax6.set_title("Beta mask (optional)")

    # IMPORTANT: parent the canvas to the right frame, otherwise it may render blank on macOS.
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    status = tk.StringVar(value="Select DAPI and Lyso images.")
    ttk.Label(right, textvariable=status).pack(anchor="w", pady=(6, 0))

    def update_preview():
        try:
            load_images(for_preview=True)
            if state["last_error"]:
                status.set(state["last_error"])
                for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
                    ax.cla()
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle("")
                canvas.draw()
                return

            seg = _compute_segmentation_for_current_state()
            if seg is None:
                status.set("Select DAPI and Lyso images.")
                return

            c = seg["cfg"]
            d01 = seg["d01"]
            l01 = seg["l01"]
            nuclei = seg["nuclei"]
            cells = seg["cells"]
            lyso_bw = seg["lyso_bw"]
            beta_bw = seg["beta_bw"]
            beta_show = seg["beta_show"]
            beta = state.get("beta")

            # Boundaries
            nuc_b = find_boundaries(nuclei, mode="outer")
            cell_b = find_boundaries(cells, mode="outer")

            # Clear and draw
            for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
                ax.cla()
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.set_title("DAPI")
            ax1.imshow(d01, cmap="gray")

            ax2.set_title("Lyso")
            ax2.imshow(l01, cmap="gray")

            ax3.set_title("Nuclei labels")
            ax3.imshow(d01, cmap="gray")
            ax3.imshow(nuc_b.astype(float), alpha=0.9)

            ax4.set_title(f"Cell masks ({c.cell_mask_method})")
            ax4.imshow(l01, cmap="gray")
            ax4.imshow(cell_b.astype(float), alpha=0.9)

            ax5.set_title("Lyso mask")
            ax5.imshow(l01, cmap="gray")
            ax5.imshow(lyso_bw.astype(float), alpha=0.7)

            ax6.set_title("Beta (optional)")
            if beta is None:
                ax6.text(0.5, 0.5, "No Beta selected", ha="center", va="center")
            else:
                # Show normalized Beta image, overlay mask for context
                ax6.imshow(beta_show if beta_show is not None else beta.astype(np.float32), cmap="gray")
                if beta_bw is not None:
                    ax6.imshow(beta_bw.astype(float), alpha=0.35)

            # Simple QC text
            n_cells = int(cells.max())
            n_nuc = int(nuclei.max())
            scale = int(state.get("preview_scale", 1))
            extra = f" | Preview: 1/{scale}x" if scale > 1 else ""
            status.set(f"Nuclei: {n_nuc} | Cells: {n_cells} | Lyso pixels: {int(np.count_nonzero(lyso_bw))}{extra}")

            canvas.draw()
        except Exception as e:
            status.set(str(e))
    def on_qc_viewer():
        """Open a small QC window showing zoomed crops of random cells with overlays.

        Uses the current preview arrays (may be downsampled), so it stays responsive.
        Changing parameters in the main UI will automatically refresh these crops.
        """
        import random

        win = tk.Toplevel(root)
        win.title("QC cell zoom")
        win.geometry("820x720")

        top = ttk.Frame(win, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Crop size (px):").pack(side="left")
        ttk.Entry(top, textvariable=v_qc_box, width=8).pack(side="left", padx=(6, 14))

        info = tk.StringVar(value="")
        ttk.Label(top, textvariable=info).pack(side="left")

        # Figure
        qc_fig = Figure(figsize=(7.8, 6.4), dpi=110)
        qc_axes = [
            qc_fig.add_subplot(2, 2, 1),
            qc_fig.add_subplot(2, 2, 2),
            qc_fig.add_subplot(2, 2, 3),
            qc_fig.add_subplot(2, 2, 4),
        ]
        for ax in qc_axes:
            ax.set_xticks([])
            ax.set_yticks([])

        qc_canvas = FigureCanvasTkAgg(qc_fig, master=win)
        qc_canvas.get_tk_widget().pack(fill="both", expand=True)

        state_qc = {"cell_ids": []}

        def _pick_cells(seg_dict):
            cell_ids = np.unique(seg_dict["cells"])
            cell_ids = cell_ids[cell_ids != 0]
            cell_ids = [int(x) for x in cell_ids.tolist()]
            if not cell_ids:
                return []
            k = min(4, len(cell_ids))
            return random.sample(cell_ids, k=k)

        def resample():
            seg = _compute_segmentation_for_current_state()
            if seg is None:
                state_qc["cell_ids"] = []
                draw()
                return
            state_qc["cell_ids"] = _pick_cells(seg)
            draw()

        ttk.Button(top, text="Resample cells", command=resample).pack(side="right")

        def draw():
            seg = _compute_segmentation_for_current_state()
            if seg is None:
                info.set("No images loaded")
                for ax in qc_axes:
                    ax.cla()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, "Load images", ha="center", va="center")
                qc_canvas.draw()
                return

            cells = seg["cells"]
            nuclei = seg["nuclei"]
            d01 = seg["d01"]
            l01 = seg["l01"]
            lyso_bw = seg["lyso_bw"]

            try:
                box = int(float(v_qc_box.get()))
            except Exception:
                box = 260
            box = int(max(120, min(900, box)))
            half = box // 2

            # Ensure we have 4 cell IDs
            ids = state_qc.get("cell_ids") or []
            # If ids are missing or no longer present, resample
            if len(ids) == 0 or any(_cell_bbox_from_label(cells, cid) is None for cid in ids):
                ids = _pick_cells(seg)
                state_qc["cell_ids"] = ids

            info.set(f"Showing {len(ids)} cells | Total cells: {int(cells.max())} | Total nuclei: {int(nuclei.max())}")

            for ax_i, ax in enumerate(qc_axes):
                ax.cla()
                ax.set_xticks([])
                ax.set_yticks([])

                if ax_i >= len(ids):
                    ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
                    continue

                cid = int(ids[ax_i])
                bb = _cell_bbox_from_label(cells, cid)
                if bb is None:
                    ax.text(0.5, 0.5, f"cell {cid} missing", ha="center", va="center")
                    continue

                y0, x0, y1, x1 = bb
                cy = int((y0 + y1) / 2)
                cx = int((x0 + x1) / 2)

                l_crop = _crop_center(l01, cy, cx, half)
                d_crop = _crop_center(d01, cy, cx, half)
                c_crop = _crop_center(cells, cy, cx, half)
                n_crop = _crop_center(nuclei, cy, cx, half)
                ly_crop = _crop_center(lyso_bw.astype(np.uint8), cy, cx, half)

                # Boundaries for overlays
                cb = find_boundaries(c_crop, mode="outer")
                nb = find_boundaries(n_crop, mode="outer")

                ax.set_title(f"cell_id={cid}")
                ax.imshow(l_crop, cmap="gray")
                ax.imshow(cb.astype(float), alpha=0.85)
                ax.imshow(nb.astype(float), alpha=0.65)
                ax.imshow(ly_crop.astype(float), alpha=0.25)

            qc_fig.tight_layout()
            qc_canvas.draw()

        # Initial pick and draw
        resample()

        # Keep QC window refreshed when parameters change
        def _tick_refresh():
            try:
                if win.winfo_exists():
                    draw()
                    win.after(350, _tick_refresh)
            except Exception:
                pass

        win.after(350, _tick_refresh)

    # ----------------------------
    # Layout
    # ----------------------------

    # File selection
    frm_files = ttk.LabelFrame(left, text="Inputs", padding=8)
    frm_files.pack(fill="x", pady=(0, 8))

    def browse_into(var: tk.StringVar):
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("TIFF", "*.tif *.tiff *.TIF *.TIFF"),
                ("All files", "*.*"),
            ],
        )
        if p:
            var.set(p)
            schedule_update()

    ttk.Label(frm_files, text="DAPI TIFF").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm_files, textvariable=v_dapi, width=38).grid(row=1, column=0, sticky="we", pady=(0, 6))
    ttk.Button(frm_files, text="Browse", command=lambda: browse_into(v_dapi)).grid(row=1, column=1, padx=(6, 0))

    ttk.Label(frm_files, text="Lyso TIFF").grid(row=2, column=0, sticky="w")
    ttk.Entry(frm_files, textvariable=v_lyso, width=38).grid(row=3, column=0, sticky="we", pady=(0, 6))
    ttk.Button(frm_files, text="Browse", command=lambda: browse_into(v_lyso)).grid(row=3, column=1, padx=(6, 0))

    ttk.Label(frm_files, text="Beta TIFF (optional)").grid(row=4, column=0, sticky="w")
    ttk.Entry(frm_files, textvariable=v_beta, width=38).grid(row=5, column=0, sticky="we", pady=(0, 6))
    ttk.Button(frm_files, text="Browse", command=lambda: browse_into(v_beta)).grid(row=5, column=1, padx=(6, 0))

    ttk.Label(frm_files, text="TIFF plane index").grid(row=6, column=0, sticky="w")
    ttk.Entry(frm_files, textvariable=v_plane, width=8).grid(row=7, column=0, sticky="w")

    # Settings
    frm_set = ttk.LabelFrame(left, text="Settings", padding=8)
    frm_set.pack(fill="x", pady=(0, 8))

    r = 0
    ttk.Label(frm_set, text="Normalize p_low").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_p_low, width=10).grid(row=r, column=1, sticky="w")
    r += 1
    ttk.Label(frm_set, text="Normalize p_high").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_p_high, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="DAPI sigma").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_dapi_sigma, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Nuc min area").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_nuc_min_area, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Cell expand (expand mode)").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_cell_expand, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Cell mask method").grid(row=r, column=0, sticky="w")
    ttk.Combobox(frm_set, textvariable=v_cell_method, values=["expand", "watershed"], state="readonly", width=12).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Checkbutton(frm_set, text="Use Lyso foreground", variable=v_fg_from_lyso).grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 0))

    r += 1
    ttk.Label(frm_set, text="FG Lyso percentile").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_fg_lyso_pct, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Lyso sigma").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_lyso_sigma, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Lyso thresh (0-1 fixed)").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_lyso_thresh, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Lyso percentile (if thresh>=1)").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_lyso_pct, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Lyso min area").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_lyso_min_area, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Checkbutton(frm_set, text="Invert Beta mask", variable=v_beta_invert).grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 0))

    r += 1
    ttk.Label(frm_set, text="Beta p_low").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_beta_p_low, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Beta p_high").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_beta_p_high, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Beta thr percentile").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_beta_thr_pct, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Beta min area").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_beta_min_area, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Beta close radius").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_beta_close_radius, width=10).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(frm_set, text="Beta overlap frac").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm_set, textvariable=v_beta_overlap, width=10).grid(row=r, column=1, sticky="w")

    # Buttons
    frm_btn = ttk.Frame(left)
    frm_btn.pack(fill="x")

    def on_refresh():
        schedule_update()

    def on_save():
        load_images(for_preview=False)
        if state["last_error"]:
            messagebox.showerror("Error", state["last_error"])
            return
        if state["dapi"] is None or state["lyso"] is None:
            messagebox.showwarning("Missing inputs", "Select at least DAPI and Lyso images.")
            return

        out_dir = filedialog.askdirectory(title="Select output folder")
        if not out_dir:
            return

        try:
            from datetime import datetime
            c = collect_cfg()
            d01 = norm01(state["dapi"].astype(np.float32), c.p_low, c.p_high)
            l01 = norm01(state["lyso"].astype(np.float32), c.p_low, c.p_high)
            nuclei = segment_nuclei(d01, c)
            cells = make_cell_masks(nuclei, l01, c, d01=d01)
            lyso_bw = segment_lyso(l01, cells, c)

            beta_bw = None
            if state["beta"] is not None:
                beta_bw = segment_beta(state["beta"].astype(np.float32), c)
                beta_bw &= (cells > 0)

            rows = compute_cell_table(cells, l01, lyso_bw, beta_bw, c)

            out_dir = Path(out_dir)

            stem = _safe_stem(Path(v_dapi.get().strip()).stem if v_dapi.get().strip() else "sentracklite")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{stem}_{ts}"

            files_written: list[Path] = []

            # Save figure
            fig_path = out_dir / f"{base}_preview.png"
            fig.savefig(fig_path, dpi=300)
            files_written.append(fig_path)

            # Save masks
            tiff_cells = out_dir / f"{base}_cells.tif"
            tiff_nuc = out_dir / f"{base}_nuclei.tif"
            tiff_lyso = out_dir / f"{base}_lyso_mask.tif"
            tiff_beta = out_dir / f"{base}_beta_mask.tif"

            tiff.imwrite(str(tiff_cells), cells.astype(np.int32))
            tiff.imwrite(str(tiff_nuc), nuclei.astype(np.int32))
            tiff.imwrite(str(tiff_lyso), lyso_bw.astype(np.uint8) * 255)
            files_written.append(tiff_cells)
            files_written.append(tiff_nuc)
            files_written.append(tiff_lyso)
            if beta_bw is not None:
                tiff.imwrite(str(tiff_beta), beta_bw.astype(np.uint8) * 255)
                files_written.append(tiff_beta)

            # Save CSV
            csv_path = out_dir / f"{base}_per_cell.csv"
            import csv
            fieldnames = [
                "cell_id",
                "cell_area_px",
                "lyso_area_px",
                "lyso_mean",
                "lyso_integrated",
                "beta_overlap_frac",
                "label_sen",
            ]
            if rows:
                fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r0 in rows:
                    w.writerow(r0)
            files_written.append(csv_path)

            summ = _summarize_well(rows)
            summ_path = out_dir / f"{base}_summary.csv"
            with open(summ_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summ.keys()))
                w.writeheader()
                w.writerow(summ)
            files_written.append(summ_path)

            inputs = {
                "dapi": v_dapi.get().strip(),
                "lyso": v_lyso.get().strip(),
                "beta": v_beta.get().strip(),
                "plane_index": int(float(v_plane.get())),
            }
            manifest_path = _write_run_manifest(
                out_dir=out_dir,
                base=base,
                cfg=c,
                inputs=inputs,
                preview_scale=int(state.get("preview_scale", 1)),
                files_written=files_written,
            )
            files_written.append(manifest_path)

            zip_path = None
            if bool(v_zip_outputs.get()):
                zip_path = _maybe_zip_outputs(out_dir, base, files_written)

            msg_lines = [
                f"Preview: {fig_path}",
                f"Cells mask: {tiff_cells}",
                f"Per-cell CSV: {csv_path}",
                f"Summary CSV: {summ_path}",
                f"Run manifest: {manifest_path}",
            ]
            if zip_path is not None:
                msg_lines.append(f"ZIP: {zip_path}")
            messagebox.showinfo("Saved", "Saved:\n" + "\n".join(msg_lines))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(frm_btn, text="Refresh preview", command=on_refresh).pack(fill="x", pady=(0, 6))
    ttk.Button(frm_btn, text="Save outputs", command=on_save).pack(fill="x", pady=(0, 6))
    ttk.Checkbutton(frm_btn, text="Zip outputs", variable=v_zip_outputs).pack(anchor="w", pady=(0, 6))

    def on_assist_plots():
        win = tk.Toplevel(root)
        win.title("Assist plots")
        win.geometry("560x260")

        v_low = tk.StringVar(value="")
        v_high = tk.StringVar(value="")
        v_plate = tk.StringVar(value="")
        v_feat = tk.StringVar(value="lyso_mean")
        v_mode = tk.StringVar(value="auto")
        v_thr = tk.StringVar(value="0.5")

        frm = ttk.Frame(win, padding=10)
        frm.pack(fill="both", expand=True)

        def pick_dir(var: tk.StringVar):
            p = filedialog.askdirectory(title="Select folder")
            if p:
                var.set(p)

        def pick_save_png(var: tk.StringVar):
            p = filedialog.asksaveasfilename(
                title="Save plots PNG",
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
            )
            if p:
                var.set(p)

        r = 0
        ttk.Label(frm, text="Folder A: per-cell CSVs").grid(row=r, column=0, sticky="w")
        r += 1
        ttk.Entry(frm, textvariable=v_low, width=60).grid(row=r, column=0, sticky="we")
        ttk.Button(frm, text="Browse", command=lambda: pick_dir(v_low)).grid(row=r, column=1, padx=(6, 0))
        r += 1

        ttk.Label(frm, text="Folder B: per-cell CSVs").grid(row=r, column=0, sticky="w", pady=(8, 0))
        r += 1
        ttk.Entry(frm, textvariable=v_high, width=60).grid(row=r, column=0, sticky="we")
        ttk.Button(frm, text="Browse", command=lambda: pick_dir(v_high)).grid(row=r, column=1, padx=(6, 0))
        r += 1

        ttk.Label(frm, text="Plate folder: per-well per-cell CSVs").grid(row=r, column=0, sticky="w", pady=(8, 0))
        r += 1
        ttk.Entry(frm, textvariable=v_plate, width=60).grid(row=r, column=0, sticky="we")
        ttk.Button(frm, text="Browse", command=lambda: pick_dir(v_plate)).grid(row=r, column=1, padx=(6, 0))
        r += 1

        opts = ttk.Frame(frm)
        opts.grid(row=r, column=0, columnspan=2, sticky="we", pady=(10, 0))

        ttk.Label(opts, text="Feature column").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=v_feat, values=["lyso_mean", "lyso_integrated", "lyso_area_px", "cell_area_px"], state="readonly", width=18).grid(row=0, column=1, padx=(6, 14), sticky="w")

        ttk.Label(opts, text="Burden mode").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=v_mode, values=["auto", "label", "prob", "feature"], state="readonly", width=12).grid(row=0, column=3, padx=(6, 14), sticky="w")

        ttk.Label(opts, text="Threshold (prob or feature)").grid(row=0, column=4, sticky="w")
        ttk.Entry(opts, textvariable=v_thr, width=8).grid(row=0, column=5, padx=(6, 0), sticky="w")

        r += 1
        v_out = tk.StringVar(value="")
        ttk.Label(frm, text="Output PNG path").grid(row=r, column=0, sticky="w", pady=(10, 0))
        r += 1
        ttk.Entry(frm, textvariable=v_out, width=60).grid(row=r, column=0, sticky="we")
        ttk.Button(frm, text="Save As", command=lambda: pick_save_png(v_out)).grid(row=r, column=1, padx=(6, 0))

        frm.columnconfigure(0, weight=1)

        def run_make():
            try:
                if not v_out.get().strip():
                    raise ValueError("Choose an output PNG path.")
                make_assist_plots(
                    out_png=Path(v_out.get().strip()),
                    folder_a=Path(v_low.get().strip()),
                    folder_b=Path(v_high.get().strip()),
                    plate_folder=Path(v_plate.get().strip()),
                    feature_col=v_feat.get().strip(),
                    burden_mode=v_mode.get().strip(),
                    prob_threshold=float(v_thr.get()),
                )
                messagebox.showinfo("Saved", f"Saved plots to:\n{v_out.get().strip()}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(frm, text="Generate plots", command=run_make).grid(row=r + 1, column=0, columnspan=2, sticky="we", pady=(12, 0))

    # Right: preview canvas and status

    # Variable traces
    for var in (
        v_dapi, v_lyso, v_beta, v_plane,
        v_p_low, v_p_high, v_dapi_sigma, v_nuc_min_area,
        v_cell_expand, v_cell_method,
        v_fg_lyso_pct, v_lyso_sigma, v_lyso_thresh, v_lyso_pct, v_lyso_min_area,
        v_beta_overlap,
        v_beta_p_low, v_beta_p_high, v_beta_thr_pct, v_beta_min_area, v_beta_close_radius,
    ):
        var.trace_add("write", schedule_update)

    for var in (v_fg_from_lyso, v_beta_invert):
        var.trace_add("write", schedule_update)

    # Small bottom-right affordance for optional plotting tools (kept out of the main workflow)
    assist_canvas = tk.Canvas(root, width=140, height=34, highlightthickness=0, bd=0)
    assist_canvas.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

    assist_canvas.create_oval(4, 6, 24, 26, fill="#dddddd", outline="#888888")
    assist_canvas.create_text(30, 16, text="Figure assist mode", anchor="w", font=("TkDefaultFont", 9))

    def _open_assist(_evt=None):
        on_assist_plots()

    assist_canvas.bind("<Button-1>", _open_assist)
    assist_canvas.bind("<Enter>", lambda _e: assist_canvas.configure(cursor="hand2"))
    assist_canvas.bind("<Leave>", lambda _e: assist_canvas.configure(cursor=""))

    # Small bottom-left affordance for QC zoom (random cells)
    qc_canvas_btn = tk.Canvas(root, width=130, height=34, highlightthickness=0, bd=0)
    qc_canvas_btn.place(relx=1.0, rely=1.0, anchor="se", x=-160, y=-10)

    qc_canvas_btn.create_oval(4, 6, 24, 26, fill="#dddddd", outline="#888888")
    qc_canvas_btn.create_text(30, 16, text="QC zoom", anchor="w", font=("TkDefaultFont", 9))

    def _open_qc(_evt=None):
        on_qc_viewer()

    qc_canvas_btn.bind("<Button-1>", _open_qc)
    qc_canvas_btn.bind("<Enter>", lambda _e: qc_canvas_btn.configure(cursor="hand2"))
    qc_canvas_btn.bind("<Leave>", lambda _e: qc_canvas_btn.configure(cursor=""))

    # Initial draw
    schedule_update()
    root.mainloop()


# --- Run GUI if main ---

if __name__ == "__main__":
    gui_main()
