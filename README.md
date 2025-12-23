# SenTrack

SenTrack is a Python-based tool for live-cell assessment of cellular senescence using lysosomal profiling. It performs single-cell image analysis from DAPI and LysoTracker images, with optional SA-β-Gal–based training, validation, and quality control workflows.

## Quick start

SenTrack provides two versions:

- **SenTrack Lite** – streamlined, GUI-based analysis for rapid senescence assessment
- **SenTrack (full)** – extended tools for model training, parameter fine-tuning, and quality control

This repository currently focuses on **SenTrack Lite**, which is sufficient for most experimental analyses.

## Requirements

- Python 3.9 or newer  
- macOS, Linux, or Windows  
- Fluorescence imaging data (DAPI + LysoTracker; SA-β-Gal optional)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/goldbader-hub/SenTrack.git
cd SenTrack
pip install -r requirements.txt

Running SenTrack Lite (analysis only)

The Lite version is designed for fast, user-guided analysis and visualization.

python SenTrackLite.py

This launches a graphical interface that allows:
	•	Manual selection of DAPI, LysoTracker, and optional SA-β-Gal images
	•	Live preview of nuclei, cell, lysosomal, and SA-β-Gal segmentation
	•	Interactive adjustment of thresholding and segmentation parameters
	•	Export of per-cell measurements, masks, and summary figures

Output

SenTrack Lite generates:
	•	Per-cell CSV files containing lysosomal features
	•	Segmentation masks for nuclei, cells, lysosomes, and SA-β-Gal signal
	•	Preview figures suitable for downstream analysis and visualization

SenTrack (full version)

The full SenTrack framework extends beyond analysis to include:
	•	Classifier training using SA-β-Gal–labeled ground truth
	•	Feature selection and model evaluation
	•	Analysis quality-control tools
	•	Batch processing across experiments and plates

These components are described in detail in the accompanying manuscript and will be released separately.

License

This project is released under the MIT License.
- Create a **minimal `requirements.txt`** for SenTrack Lite  
- Help you write the **short GitHub description (≤350 chars)**

You’re doing this exactly right.
