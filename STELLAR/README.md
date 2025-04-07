# STELLAR: **S**olar-**T**yp**e** osci**LL**ation **A**nalyse**R**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![UltraNest](https://img.shields.io/badge/dependency-UltraNest-orange)](https://johannesbuchner.github.io/UltraNest/)

A Python class for **automated asteroseismic analysis** of solar-type oscillators (MS to AGB stars). Extracts global oscillation parameters (`fmax`, `dnu`, `dnu02`,...), evolutionary stage, and individual modes (`l=0-3`) from power density spectra (Kepler, TESS, etc.). Optimized for speed—completes analysis in **minutes** on an Apple M2 CPU.

---

## 🚀 Installation
1. Install [UltraNest](https://johannesbuchner.github.io/UltraNest/):
   ```bash
   pip install ultranest
   ```
2. Download `stellar.py` and `RFmodel_dnu.pkl` to your working directory.

---

## 🛠️ Quick Start
```python
from stellar import stellar

# Initialize with a power density spectrum (PDS)
star = stellar(
    ID="MyStar",                      # Project name (output folder prefix)
    pds=pd.DataFrame(columns=["f", "p"]),  # PDS: frequency (µHz) and power (ppm²/µHz)
    path="files",                     # Output directory (optional)
    f_nyg=4165.2,                     # Nyquist frequency (default: TESS 2-min cadence)
    verbose=True                      # Toggle UltraNest output (optional)
)
```

---

## 🔍 Class Methods (Execute in Order)

### 1. **Granulation Background Fit**
```python
star.fmax_fitter(fmax_guess=False, plot=False)
```
- **Input**: Automatically locates power excess (`fmax`) or uses `fmax_guess` if provided.
- **Model**: Follows [Kallinger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract).
- **Output**: 
  - Best-fit params → `<ID>.bg_par.dat`
  - Fit components → `<ID>.bg.fit.pds`
  - Plot (if `plot=True`) → `<ID>.pdf`

---

### 2. **Central Large Frequency Separation (`dnu`)**
```python
star.dnu_fitter(dnu_guess=False, flip_fc=False, plot=False)
```
- **Input**: Uses a Random Forest regressor (trained on 6000+ Kepler red giants) for initial `dnu`/`dnu02` guesses (accurate to ~2%).  
  - For `fmax > 280 µHz`, provide `dnu_guess`.
- **Output**: 
  - Evolutionary stage (MS/RGB/RC/AGB) via [Kallinger et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..51K/abstract).
  - Params → `<ID>.dnu_par.dat`
  - Plot (if `plot=True`) → `<ID>.pdf`

---

### 3. **Peakbagging: `l=0` & `l=2` Modes**
```python
star.peakbag_02(
    alpha=None,                      # Curvature parameter (auto-estimated if None)
    l1_threshold=8,                  # Threshold for dipole-mode suppression
    odds_ratio_limit=5,              # Mode significance cutoff
    rotation=False,                  # Enable rotational splitting for l=2
    incl_prior=None,                 # Gaussian prior for inclination [mean, σ]
    plot=False,
    log=False
)
```
- **Steps**:
  1. Fits Lorentzians to radial (`l=0`) and quadrupole (`l=2`) modes.
  2. Checks for narrow dipole-mode interference (if `l1_threshold` is set).
  3. Prewhitens significant modes (`odds_ratio > limit`).
  4. Fits curvature via [Kallinger et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.104K/abstract).
- **Output**: Mode params → `<ID>.modes.dat`

---

### 4. **Peakbagging: `l=1` & `l=3` Modes**
```python
star.peakbag_13(
    ev_limit=0.8,                    # Evidence threshold for mode acceptance
    snr_limit=7.5,                   # Signal-to-Background cutoff
    iter_limit=150,                  # Max iterations
    plot=False,
    rotation=False,                  # Experimental: rotational splitting for l=1 (MS/subgiants only)
    incl_prior=None
)
```
- **Notes**:
  - Works only for `fmax > 30 µHz` (hard limit).
  - **Experimental**: Mixed dipole modes in red giants remain challenging to automate.
- **Output**: Updated `<ID>.modes.dat` and plots (if `plot=True`).

---

## 🌟 Planned Features
- **Core rotation rate** for red giants (via mixed dipole modes; Schauer & Kallinger, in prep.).
- **Automated `dP1` identification**.

---

## 📜 References
- Background modeling: [Kallinger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract)
- `dnu`/`dnu02`: [Kallinger et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...509A..77K/abstract)
- Evolutionary stage: [Kallinger et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..51K/abstract)
- Peakbagging: Based on [ABBA (Kallinger 2019)](https://ui.adsabs.harvard.edu/abs/2019arXiv190609428K/abstract)

---

For further informations please contact tkallinger@me.com
