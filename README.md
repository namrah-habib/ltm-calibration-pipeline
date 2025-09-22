# LTM Calibration Pipeline

Python-based calibration pipeline for Lunar Thermal Mapper (LTM) aboard Lunar Trailblazar. The calibration pipeline takes the raw LTM data and performs a 4-point radiometric calibration on the data to compute the calibrated radiance and temperature. The pipeline inputs the raw LTM HDF5 files, applies row/gain/stray‑light corrections, computes calibrated radiance via an instrument model, and writes analysis‑friendly HDF5 outputs along with quicklook images.

---

## Features

* Load LTM **observation** HDF5 files and extract per‑look cubes (`Int`, `Cold`, `Hot`). 
* Apply **row & offset correction** and **stray‑light correction**. 
* Compute **calibrated radiance** using the **LTM instrument model** and per‑look metadata. 
* Write an output **HDF5** file with calibrated dataset  
* Generate **quicklook** PDF images of calibrated radiance
---

## Directory Layout 

```
ltm-cal-pipeline/
├─ calibration-tables/            # Required model correction tables 
│  ├─ TemperatureGain_tdi.pkl
│  ├─ StrayLightCorr.pkl
│  ├─ StrayLightCorr_tdi.pkl
│  ├─ RadCalIndex.mat
│  ├─ LTM_MODEL_FilterBandpasses.pkl
│  ├─ LTM_MODEL_MirrorReflectivity.pkl
│  └─ LTM_MODEL_DetectorResponse.pkl
├─ ltm-raw-data/                  # Directory for input (raw) HDF5 LTM data
├─ ltm-calibrated-data/           # HDF5 outputs written by the pipeline
├─ ltm-quicklooks/                # Quicklook images of calibrated radiance
├─ src/
│  ├─ ltm_radcal.py               # RadCal Class (code that computes radiometric calibraiton)
│  ├─ ltm_model.py                # Instrument radiance model
│  └─ preview_frame.py            # Plotting function
├─ ltm-calibration-pipeline.py    # Calibration python (THIS IS THE CODE TO RUN)
├─ README.md
├─ requirements.txt
└─ environment.yml
```
---

## Installation

Use either the Conda **environment.yml** or the pip **requirements.txt**.

### Option 1 — Conda (recommended)

```bash
conda env create -f environment.yml -n ltm
conda activate ltm
```

Update an existing env:

```bash
conda env update -f environment.yml -n ltm --prune
```

### Option 2 — pip + venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
---

## Quick Start

Run the end‑to‑end pipeline over all files in `ltm-raw-data/`:

```bash
python ltm-calibration-pipeline.py
```
---

## Output HDF5 layout (flat, per‑look)

The output HDF5 file contains, for each LTM view (`Internal Calibration View`, `Cold Calibration View`, `Scene`):

* **Raw Data** — original frames
* **Row Corrected Frames** — row/offset corrected
* **Offset Average** — per‑row masked average
* **Corrected Frames** — stabilized corrected frames
* **Corrected Frames with Stray Light Removed** — stabilized + stray‑light‑corrected
* **Calibrated Radiance** — final product
* Attributes: `time`, `temperature`, `tdi_flg`

All names and attribute keys are set in `save_calibrated_data(...)`. 

---

## License

authors @ N.Habib, R. Evans

Copyright (c) [2025] [Atmospheric, Oceanic & Planetary Physics, University of Oxford]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

---

## Citation

If you use this code in a publication, please cite the repository and describe the LTM calibration approach as implemented here.

---

## Changelog

* Initial public pipeline and documentation.
