# SenTrack

SenTrack is a Python tool for live-cell assessment of cellular senescence using lysosomal profiling. It performs single-cell image analysis from DAPI and LysoTracker images, with optional SA-β-Gal support for training, validation, and QC workflows.

## What’s in this repo

### SenTrackLite (GUI, analysis-only)
A lightweight, user-guided segmentation and measurement app for quick senescence assessment and figure-ready outputs.

### SenTrack (full pipeline)
A more comprehensive toolkit intended for training, fine-tuning parameters, validation, and additional QC utilities.

## Requirements
- Python 3.9+ (recommended: 3.10 or 3.11)
- macOS, Linux, or Windows
- Imaging inputs: DAPI + LysoTracker (SA-β-Gal optional)

## Installation

1) Clone the repo
```
git clone https://github.com/goldbader-hub/SenTrack.git
cd SenTrack
```

2) Create and activate a virtual environment (recommended)

macOS / Linux
```
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3) Install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (only needed for some compressed TIFFs like LZW)
```
pip install imagecodecs
```

## Quick start

### Run SenTrackLite (GUI)
```
python SenTrackLite.py
```

In the GUI you can:
- Select DAPI, LysoTracker, and optional SA-β-Gal images manually
- Live-preview segmentation and masks
- Adjust thresholds and segmentation parameters
- Export per-cell measurements, masks, and preview figures

### Run the full SenTrack pipeline (CLI)
If this repo includes a CLI script (for example `Sentrackfigures.py`), you can check available commands with:
```
python Sentrackfigures.py --help
```

## Outputs (SenTrackLite)
SenTrackLite can export:
- Per-cell CSV with lysosomal features
- Segmentation masks (nuclei, cells, lysosomes, optional SA-β-Gal mask)
- Preview PNG suitable for figures and QC

## Notes on TIFF compatibility
Some microscope TIFFs are compressed (often LZW). If loading fails with an error mentioning `imagecodecs`, install:
```
pip install imagecodecs
```
If you prefer not to install extra codecs, convert TIFFs to uncompressed in Fiji/ImageJ (Save As TIFF with Compression: None).

## License
MIT License (see `LICENSE`).

## Citation
If you use SenTrack in academic work, please cite the accompanying paper.
