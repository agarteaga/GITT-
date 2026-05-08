# GITT_V2 — Galvanostatic Intermittent Titration Technique Analysis GUI

A multi-sample desktop application for analysing GITT electrochemical data. Implements both the Weppner & Huggins (1977) conventional method and the Kang & Chueh (2021) corrected method for extracting solid-state diffusion coefficients, alongside a full suite of visualisation and diagnostic tabs.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| tkinter | bundled with Python (see Installation) |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| matplotlib | ≥ 3.6 |

---

## Installation

```bash
pip install -e .
```

### Linux (Ubuntu / Debian)
```bash
sudo apt-get install python3-tk
pip install -e .
```

### macOS
```bash
brew install python-tk   # only if using Homebrew Python
pip install -e .
```

### Windows
tkinter ships with the standard python.org installer — just run:
```bash
pip install -e .
```

---

## Running

```bash
# Open the GUI with no data pre-loaded:
python GITT_V2.py

# Pre-load a data directory on startup:
python GITT_V2.py --data_dir /path/to/your/data
```

---

## Data Format

The app reads BioLogic-style `.dat` files. Place both file types in a folder — the app detects them automatically by filename keywords.

### Time file  (`*time*.dat`)
Tab-separated. Header must contain `Potential` (or `Ewe`/`Voltage`), `Current` (or `Iwe`), and `Time` (or `Elapsed`) — all on the same line. Comma or period decimals are both accepted.

```
Potential (V)    Current (A)    Elapsed Time (s)
1.959867         -7.46483E-05   231930.99
...
```

### Capacity file  (`*cap*.dat`)
Tab-separated. Header must contain a capacity keyword (`Capacity`, `mAh`, `mA.h`, `Charge`) and a voltage keyword (`Ecell`, `Ewe`, `Potential`, `Voltage`).

```
Capacity/mA.h    Ecell/V
4.147E-08        1.95987
...
```

Units `mAh/g` are detected automatically from the capacity column header and handled correctly throughout.

### Multi-file experiments
Split files (e.g. `G170_time_1.dat`, `G170_time_2.dat`) are sorted and concatenated automatically.

### Folder structure
```
data/
├── SampleA/
│   ├── SampleA_time_1.dat
│   ├── SampleA_cap_1.dat
│   └── ...
├── SampleB/
│   └── ...
└── *.dat   ← files directly in the root are shown as "(this folder)"
```
Point the app at `data/` and each subfolder becomes a named sample.

---

## Interface Overview

### Parameter bar (top)

| Parameter | Default | Description |
|---|---|---|
| **tau (s)** | auto | GITT pulse duration in seconds |
| **Fit start (s)** | 60 | Start of the sqrt(t) / xi fit window |
| **Fit end (s)** | auto (= tau) | End of the fit window |
| **V\_lo / V\_hi** | auto | Voltage range filter applied to D plots |
| **Min R²** | 0.85 | Minimum R² to include a block in D calculations |
| **delta\_e** | 1 | Number of formula units per unit cell (δe) |
| **sigma-clip** | 0 (off) | Robust MAD-based outlier rejection on log₁₀(D) |
| **KC xi\_min / xi\_max** | 0 / 99 | Fit-window limits for the K&C ξ axis |
| **Pulse #** | 1 | Pulse to show in single-pulse fit and relaxation tabs |
| **m for Rtot (mg)** | 7.05 | Active electrode mass for normalised R\_tot output |

Parameters are auto-detected from the first loaded sample (τ, V range) and update plots live.

---

## Tabs

### 🔵 Raw Data
| Tab | Content |
|---|---|
| **GITT curve** | Full V(t) and I(t) traces with the fit window highlighted |

### 🟣 Analysis
| Tab | Content |
|---|---|
| **OCV** | Top: V\_eq (relaxation endpoints) vs SOD/SOC %; bottom: full E vs SOD/SOC from cap file |
| **Relaxation curves** | All relaxation V(t) overlaid, coloured by V\_eq. Top: absolute E; bottom: E − V\_eq (mV) showing convergence to equilibrium |
| **Relaxation ΔV** | Total voltage recovery per pulse (V\_eq − V\_relax,start) vs SOD/SOC % |
| **Overpotential curves** | All pulse η(t) = V\_pulse(t) − V\_eq,prev overlaid, coloured by V\_eq |
| **Overpotential & Rtot** | η (mV) and R\_tot × mass (Ω·g) vs SOD/SOC % |

### 🟩 Weppner & Huggins (Conventional)
| Tab | Content |
|---|---|
| **All pulses (sqt)** | All ΔV vs √t overlaid, coloured by V\_eq gradient, with fit lines |
| **sqrt(t) fits** | Single-pulse ΔV vs √t for the selected Pulse # with fit line and R² |
| **Slopes & dE/dx** | \|dV/d√t\| and \|dE/dx\| vs V\_eq and vs SOD/SOC % |
| **D coefficient** | D̃\_conv vs V\_eq (top) and vs SOD/SOC % (bottom) |

### 🟥 Kang & Chueh (K&C)
| Tab | Content |
|---|---|
| **All xi pulses** | All V vs ξ = √(t+τ)−√t overlaid, coloured by V\_eq gradient, with fit lines |
| **xi fits** | Single-pulse V vs ξ for the selected Pulse # with fit and R² |
| **xi slopes** | \|s\_KC\| vs V\_eq (top) and vs SOD/SOC % (bottom) |
| **D coefficient** | D̃\_KC vs V\_eq (top) and vs SOD/SOC % (bottom) |

### ⚫ Documentation
Built-in equations reference with rendered LaTeX, method descriptions, and parameter guidance.

---

## Physics

### SOD / SOC normalisation

State of Discharge/Charge is computed from the time file by trapezoidal integration of |I|·dt, normalised by the **total pulse charge including the last dropped pulse**:

```
x_block[n] = Σ_{k=1..n} ∫|I|dt_k  /  Σ_{k=1..N_all} ∫|I|dt_k
```

Discharge and charge are normalised independently from their own first pulse start. This ensures OCV dots and the E vs SOD/SOC curve share a physically consistent x-axis.

### Conventional GITT — Weppner & Huggins (1977)

```
D̃_conv · (S/Vm)² = (4 I² / π F² δe²) · (dVeq/dx / s_√t)²
```

s\_√t is the slope of ΔV vs √t in the user-defined fit window [Fit start, Fit end].

### K&C GITT — Kang & Chueh (2021)

```
D̃_KC · (S/Vm)² = (4 I² / π F² δe²) · (dVeq/dx / s_ξ)²
```

s\_ξ is the slope of V\_relax vs ξ = √(t + τ) − √t over the full relaxation, windowed by KC xi\_min / xi\_max.

Both methods report the lumped product D̃·(S/Vm)² in mol² s⁻¹. To extract D alone you need the electrode geometry (S = electroactive area) and molar volume (Vm).

---

## Exporting

| Button | Output |
|---|---|
| **Export figures** | PNG images (150 dpi) of all visible plots → `<data_dir>/results/` |
| **Export curves** | Origin-ready CSV files, one X,Y pair per dataset → same folder |

---

## Keyboard & UI tips

- Changing **Pulse #** updates the sqrt(t) fits, xi fits, and single-pulse views simultaneously with a single redraw.
- All spinners update plots live — no need to press **Replot** unless reloading data.
- Check/uncheck sample boxes to overlay or hide individual samples.
- Adjust **V\_lo** and **V\_hi** to exclude bad voltage regions from D plots without affecting fits.
- **Sigma-clip** (e.g. 2.5) applies robust outlier rejection on log₁₀(D) using the MAD estimator; set to 0 to disable.

---

## References

1. Weppner, W. & Huggins, R. A. *Determination of the kinetic parameters of mixed-conducting electrodes and application to the system Li₃Sb.* J. Electrochem. Soc. **124**, 1569–1578 (1977).
2. Kang, S. & Chueh, W. C. *Galvanostatic Intermittent Titration Technique Reinvented: Part I. A Critical Review.* J. Electrochem. Soc. **168**, 120504 (2021).
